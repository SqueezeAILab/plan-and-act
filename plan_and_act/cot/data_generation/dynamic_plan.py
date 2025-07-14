from __future__ import annotations

import argparse
import asyncio
import json
import os

from plan_and_act.cot.models import (
    LLM,
    ActionStep,
    AsyncDataGenerationJobEngine,
    CotPlanAnnotatorDataGeneratorData,
    DynamicPlanAnnotatorDataGeneratorData,
    Plan,
    PlanAnnotatorOutput,
    PlanHistory,
    ProcessedData,
    ReplanDecision,
    ReplanStep,
    ServerMessage,
    WebArenaLiteAction,
)
from plan_and_act.cot.utils import (
    clean_html,
    format_action_steps,
    format_last_exit_action,
    get_html_from_action,
    preprocess_webarena_data,
)

# This file implements the dynamic replanning system, which enhances the original
# Plan-and-Act system with the ability to adapt plans during execution.
#
# It contains two main components:
# 1. DynamicCoTPlanAnnotator: Generates adaptive replans based on existing initial plans
# 2. DynamicCoTPlanAnnotatorDataGenerator: Orchestrates replan generation for multiple tasks


class DynamicCoTPlanAnnotator:
    """
    Enhanced plan annotator that generates adaptive replans.

    This annotator is used to generate training data for a dynamic planning model.
    It has access to the full ground truth action trajectory and creates replans
    as if it were the actual planner model making predictions.

    The annotator performs two main functions:
    1. Use the existing initial plan and create replans based on current state
    2. Generate reasoning for these replans as if it were the planner model itself
    """

    _MAX_RETRY = 3
    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool = False) -> None:
        """
        Initialize the dynamic plan annotator.

        Args:
            llm: The language model to use for generating replans
            verbose: Whether to print verbose output
        """
        self._llm = llm
        self._verbose = verbose

    async def annotate(
        self,
        plan_data: CotPlanAnnotatorDataGeneratorData,
        processed_data: ProcessedData,
    ) -> DynamicPlanAnnotatorDataGeneratorData:
        """
        Generate potential replans for all actions in the task based on existing initial plan.

        This method:
        1. Uses the existing action descriptions from plan_data
        2. Uses the existing initial plan from plan_data
        3. For each action, generates replans considering all future actions
        4. Returns a complete history of plans and replanning decisions

        Args:
            plan_data: The existing initial plan data
            processed_data: The processed data for a task containing actions and initial state

        Returns:
            A DynamicPlanAnnotatorDataGeneratorData object containing the task and plan history
        """
        # Sort the actions by the index within task
        actions = sorted(
            processed_data["actions"], key=lambda x: x["index_within_task"]
        )

        last_exit_action = actions[-1]

        # Get the initial plan and action descriptions from the provided plan data
        # The data field is a tuple with structure: [list[ActionStep], PlanAnnotatorOutput, PlanTrainingData]
        data_tuple = plan_data["data"]

        # Extract the action descriptions from the first element of the tuple
        action_step_descriptions = sorted(
            data_tuple[0], key=lambda x: x["action"]["index_within_task"]
        )

        # Extract the third element (PlanTrainingData) and get the plan from it
        plan_training_data = data_tuple[2]
        initial_plan_data = plan_training_data["plan"]

        # Create the Plan object
        initial_plan = Plan(
            reasoning=initial_plan_data["reasoning"],
            plan=initial_plan_data["plan"],
        )
        initial_html_state = plan_training_data["initial_html_state"]

        if self._verbose:
            print(
                f"\n=== Using {len(action_step_descriptions)} existing action step descriptions ==="
            )
            print(f"\n=== Initial Plan ===\n{initial_plan['plan']}\n==================")

        # Process each action step to generate replans
        replan_steps = []
        current_plan = initial_plan

        for idx in range(len(actions) - 1):
            if self._verbose:
                print(f"\n=== Processing action {idx} ===")

            # Skip the last action (which is usually just the exit action)
            if idx == len(actions) - 1:
                continue

            # Get HTML state after this action
            next_action = actions[idx + 1]
            html_state = clean_html(
                get_html_from_action(next_action),
                prettify=True,
                strip_annotation=True,
            )

            # Get remaining action steps for future actions
            remaining_action_steps = (
                action_step_descriptions[idx + 1 :]
                if idx + 1 < len(action_step_descriptions)
                else []
            )
            previous_action_steps = action_step_descriptions[: idx + 1]

            # Generate replan with the knowledge of all future actions
            replan_output = await self._generate_replan(
                processed_data["task"],
                html_state,
                current_plan,
                previous_action_steps,
                remaining_action_steps,
                last_exit_action,
            )

            # Update the current plan with the new plan
            current_plan = Plan(
                reasoning=replan_output["reasoning"],
                plan=replan_output["plan"],
            )

            if self._verbose:
                print(f"\n=== New Plan ===\n{current_plan['plan']}\n===============")

            # Create a replan decision based on the reasoning
            replan_decision = ReplanDecision(
                needs_replan=True,  # We always generate a replan in this data generation
                reasoning=replan_output["plan_generation_reasoning"],
            )

            replan_step = ReplanStep(
                index_within_task=next_action["index_within_task"],
                plan=current_plan,
                html_state=get_html_from_action(next_action),
                previous_executor_actions=actions[: idx + 1],
                replan_decision=replan_decision,
            )
            replan_steps.append(replan_step)

        return DynamicPlanAnnotatorDataGeneratorData(
            task=processed_data["task"],
            data=PlanHistory(
                initial_plan=initial_plan,
                initial_html_state=initial_html_state,
                replan_steps=replan_steps,
            ),
        )

    async def _generate_replan(
        self,
        task: str,
        current_html_state: str,
        current_plan: Plan,
        previous_action_steps: list[ActionStep],
        remaining_action_steps: list[ActionStep],
        last_exit_action: WebArenaLiteAction,
    ) -> PlanAnnotatorOutput:
        """
        Generate a new plan based on the current HTML state, previous actions,
        and remaining actions (all future actions are shown for data generation).

        This method follows a two-stage process:
        1. Generate a new plan with knowledge of future actions
        2. Generate reasoning as if the model didn't have knowledge of future actions

        Args:
            task: The task description
            current_html_state: The current HTML state (cleaned and prettified)
            current_plan: The current plan
            previous_action_steps: Previous action step descriptions
            remaining_action_steps: Remaining action step descriptions (ground truth future actions)

        Returns:
            A PlanAnnotatorOutput object containing the new plan and reasoning
        """
        # Format previous and remaining actions using the helper method
        formatted_previous_steps = format_action_steps(
            previous_action_steps,
            "Previous Action",
            include_html_snippet=False,
            use_simple_html=True,
        )
        formatted_remaining_steps = format_action_steps(
            remaining_action_steps,
            "Remaining Action",
            include_html_snippet=False,
            use_simple_html=True,
        )

        # Format the last exit action if remaining_action_steps is not empty
        last_exit_section = ""
        try:
            # Format the last exit action section for the prompt
            last_exit_section = format_last_exit_action(
                last_exit_action,
                last_exit_action_prefix="Last Remaining Action",
                include_html_snippet=False,
            )
        except (IndexError, KeyError):
            # If there's an error, continue without the exit action
            pass

        # Stage 1: Generate a new plan with knowledge of future actions
        prompt = f"""## Your Role: The ExecutionReplanner

You are an ExecutionReplanner helping a user complete a web task. Your job is to update an existing plan based on what's already happened and what the user is currently seeing.

## What is Replanning?

Replanning is the process of updating an existing plan based on:
- Actions that have already been taken
- The current state of the webpage
- New information that has been revealed
- Changes in the environment since the original plan was created

Replanning becomes necessary when:

1. The initial plan was incorrect or incomplete:
   - The plan didn't accurately predict page changes
   - Something unexpected happened (e.g., no search results after submitting)
   - Elements mentioned in the plan don't exist on the page
   - The plan didn't divide the task into appropriate high-level goals

2. The plan needs more specificity:
   - The initial plan was necessarily vague about dynamic content
   - Now that execution has progressed, specific information is available
   - For example, "Find the top contributor" becomes "Click on JohnDoe123" once usernames are visible

3. The execution environment has changed:
   - Elements that were supposed to be visible aren't (requiring scrolling)
   - The website structure differs from what was expected

## The Timeline of This Task

Let me explain where we are in the process:

1. **Original Plan**: The user started with an initial plan for completing their task.

2. **Previous Actions**: The user has already performed some actions following that plan. These are in the past.

3. **Current Moment**: Right now, the user is looking at a webpage. The HTML I'm showing you is exactly what they're seeing at this moment after their most recent action.

4. **Next Action**: The user will take ONE immediate action on this current HTML. After that action, the webpage will change.

5. **Future Actions**: After the first action, the webpage will change, and then another action will be taken on the NEW HTML (which you can't see yet), and so on.

## Important: Understanding the Current HTML

The current HTML snapshot I'm showing you is ONLY relevant for deciding the NEXT IMMEDIATE action. It becomes invalid after that action is taken because the webpage will change.

For example:
- If the user has entered text in a search box but hasn't clicked "Search" yet, the current HTML won't show search results
- If the user is about to click on a product listing, the current HTML won't show the product details page that will appear after clicking

**Important Note on Final Answers**: Do NOT include the final answer (like a phone number, address, or price) in your plan UNLESS:
1. The only remaining action is the exit action, AND
2. The necessary information to exit with the appropriate answer is visible in the current HTML

Otherwise, your plan should describe how to FIND the information, not what the information will be, since it may not be visible yet.

## Your Task: Generate an Updated Plan

You need to update the existing plan based on:
- What has already been accomplished (previous actions)
- What the user is seeing right now (current HTML)
- What needs to happen next (remaining actions)

Since this is for data generation, I'm providing you with the ground truth future actions that will be taken. Your updated plan should align with these actions, but express them as high-level goals and steps.

## How to Create Your Updated Plan

1. First, carefully analyze which parts of the previous plan have already been completed by examining the previous actions list.

2. Next, examine the current HTML to understand exactly what the user is seeing right now.

3. Then, review the remaining actions that will be taken. Remember:
   - The FIRST remaining action will be performed on the CURRENT HTML
   - Each SUBSEQUENT action will be performed on a NEW HTML state that will appear after the previous action

4. Determine whether the previous plan:
   - Can continue as is (if it's still valid)
   - Needs updates with new information
   - Contains incorrect assumptions that need correction
   - Should be completely replaced

5. Write out your detailed thinking process first, including your analysis of:
   - Which exact actions from the previous plan have been completed
   - What the current HTML shows and what can be done now
   - What the next immediate action should be
   - How you expect the page to change after each action

6. Finally, create a comprehensive updated plan between "[Start of Plan]" and "[End of Plan]" tags.

## The Information You Have

**Task Query:**
{task}

**Previous Plan (that needs updating):**
```
{current_plan["plan"]}
```

**Actions Already Taken:**
{formatted_previous_steps}

**Current HTML State (what the user sees RIGHT NOW):**
{current_html_state}

**Ground Truth Future Actions (in sequence, each changing the HTML after it's taken):**
{formatted_remaining_steps}
{last_exit_section}

Based on this information, please provide your step-by-step analysis followed by an updated execution plan that will guide the user from their current state to successful completion of the task.
"""

        if self._verbose:
            print(f"@@@@ Stage 1 prompt:\n\n{prompt}\n\n@@@@")

        messages: list[ServerMessage] = [
            {"role": "user", "content": prompt},
        ]

        plan_generation_reasoning = None
        plan = None
        for _ in range(self._MAX_RETRY):
            try:
                response = await self._llm.acall_model(messages)
                if response is None:
                    raise ValueError("Failed to generate replan.")

                # Check if the response contains the plan.
                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    # Extract the reasoning from the response (any text before the plan)
                    plan_generation_reasoning = response.split("</think>")[0].strip()
                    plan = (
                        response.split("[Start of Plan]")[1]
                        .split("[End of Plan]")[0]
                        .strip()
                    )
                    break

                # If the response does not contain the plan, retry.
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response,
                        },
                        {
                            "role": "user",
                            "content": "You need to ensure that you enclose the final plan between the '[Start of Plan]' and '[End of Plan]' tags.",
                        },
                    ]
                )

            except Exception as e:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}",
                    },
                )

        if plan is None or plan_generation_reasoning is None:
            raise ValueError("Failed to generate replan.")

        if self._verbose:
            print(f"@@@@ Stage 1 response:\n\n{response}\n\n@@@@")

        # Stage 2: Generate reasoning as if the model didn't have knowledge of future actions
        prompt = f"""Now, I want you to be an annotator LLM helping to generate training data for a dynamic execution replanner model. You have already generated a perfect updated plan for the given task in the previous conversation; however, the replanner model only has access to the task query, the current HTML state, the actions taken so far, and the previous plan (NOT the remaining actions).

Your task is to generate reasoning and a plan by pretending that you are the replanner that has only access to the task query, the current HTML state, the actions taken so far, and the previous plan. This replanner must be an expert in web-based task planning, creating execution plans that are as comprehensive as the one you previously generated (as if it can "see the future", as if it is a "world model" of this particular website). Its reasoning should:

- Demonstrate expert-level understanding of web navigation and DOM manipulation
- Show how to analyze the current HTML state to confidently identify the correct elements to interact with
- Predict what changes the actions will cause to the webpage state
- Explain why the updated plan aligns with the current state of the task
- Act as a "world model" by anticipating how the webpage will respond to future actions
- Pretend as if you are the replanner itself. Hence, you MUST always talk in "I" or "we" instead.

Important: The replanner understands that:
- The current HTML state represents the webpage AFTER the most recent action
- This HTML state is just a snapshot of the current page, not the final state
- Each action will change the webpage, requiring anticipation of these changes
- The plan must integrate observations from the current HTML to help with the immediate next steps
- The plan should be flexible enough to adapt as the webpage changes after each action

The replanner should consider several scenarios where replanning might be necessary:
1. When the initial plan was incorrect or incomplete:
   - The plan wasn't able to correctly predict page changes
   - Something unexpected happened (e.g., no search results after submitting a query)
   - Elements mentioned in the plan don't exist on the page
   - The plan didn't accurately divide the task into correct high-level goals

2. When the plan needs more specificity:
   - The initial plan might have been necessarily vague about dynamic content
   - Now that execution has progressed, specific information is available
   - For example, an initial plan might say "Identify the top contributor and follow them" but after navigating to the contributors section, the plan can be updated to "Follow UserX" with the actual username

3. When the execution environment has changed:
   - Elements that were supposed to be visible aren't (requiring scrolling or navigation)
   - The website structure has changed from what was expected

... And more.

Here is the information that the replanner will use to generate the updated plan:
        
Task Query:
"{task}"

Current HTML State (after the most recent action):
{current_html_state}

Previous Plan (that was being followed):
{current_plan["plan"]}

Actions Taken So Far (according to the previous plan):
{formatted_previous_steps}

Some special notes:
- Since the replanner only has access to the current HTML state, it must handle dynamic content carefully. It should not assume specific search results or values.
  * Example: When searching for gas stations, explain how to analyze results rather than assuming specific stations
  * Example: When finding order #178, explain how to locate and verify the order rather than assuming its position

- Similarly, if the task requires the analysis of some page and then give the final answer, the replanner should 
not assume the final answer but instead should explain how to analyze the page and extract an answer from it.

- Guidelines for handling dynamic/unknown content:
  * Describe how to analyze search results
  * Explain methods to extract information dynamically
  * Show how to make decisions based on available content
  * Avoid hardcoding any specific values or answers

Remember:
- You are generating training data to teach another model how to reason about updating plans
- The reasoning should be very detailed, comprehensive, and thorough about the expected outcomes, webpage state changes, and how to achieve the high level goals.
- Focus on the high-level goals and explain how the webpage will look like after each step like a "world model". 
- Explain how to handle dynamic content without assuming specific values
- Output your final plan between [Start of Plan] and [End of Plan] tags.
- Avoid mentioning the "replanner" in your reasoning. Always talk in "I" or "we" instead.
"""

        # Append our new prompt to the existing conversation
        messages.append({"role": "user", "content": prompt})

        if self._verbose:
            print(f"@@@@ Stage 2 prompt:\n\n{prompt}\n\n@@@@")

        response = None
        reasoning = None
        updated_plan = None
        for _ in range(self._MAX_RETRY):
            try:
                response = await self._llm.acall_model(messages)
                if response is None:
                    raise ValueError(
                        "Failed to generate stage 2 plan reasoning. Response is None."
                    )

                # Check if reasoning and plan are present in the response.
                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    reasoning = response.split("</think>")[0].strip()
                    updated_plan = (
                        response.split("[Start of Plan]")[1]
                        .split("[End of Plan]")[0]
                        .strip()
                    )
                    break

                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response,
                        },
                        {
                            "role": "user",
                            "content": "You need to ensure that you enclose the final plan between the '[Start of Plan]' and '[End of Plan]' tags.",
                        },
                    ]
                )
            except Exception as e:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}",
                    }
                )

        if reasoning is None or updated_plan is None:
            if response is None:
                raise ValueError(
                    "Failed to generate stage 2 plan reasoning. Reasoning or updated plan is None."
                )
            else:
                reasoning = response
                updated_plan = None

        if self._verbose:
            print(f"@@@@ Stage 2 response:\n\n{response}\n\n@@@@")

        return PlanAnnotatorOutput(
            task=task,
            plan_generation_reasoning=plan_generation_reasoning,
            plan=plan if updated_plan is None else updated_plan,
            reasoning=reasoning,
        )


class DynamicCoTPlanAnnotatorDataGenerator:
    """
    Data generator for the Dynamic CoT Plan Annotator.

    This class orchestrates the generation of replans for a set of tasks based on existing initial plans,
    saving the results to disk for later use by the Dynamic Executor Annotator.
    """

    _dynamic_cot_planner: DynamicCoTPlanAnnotator

    # Input data
    _paired_data: list[tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]]
    _already_processed: set[str]

    # Output paths
    _output_path: str

    # Results aggregator
    _results: list[DynamicPlanAnnotatorDataGeneratorData]

    def __init__(
        self,
        dynamic_cot_planner: DynamicCoTPlanAnnotator,
        paired_data: list[tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]],
        output_path: str,
    ) -> None:
        """
        Initialize the dynamic plan annotator data generator.

        Args:
            dynamic_cot_planner: The dynamic plan annotator
            paired_data: List of paired plan data and processed data
            output_path: Path to save results
        """
        self._dynamic_cot_planner = dynamic_cot_planner
        self._paired_data = paired_data
        self._output_path = output_path
        self._already_processed = self._get_already_processed_tasks(output_path)
        self._results = []

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load any existing results
        if os.path.exists(output_path):
            try:
                with open(output_path, "r") as f:
                    self._results = [json.loads(line) for line in f]
            except Exception as e:
                print(f"Error loading existing results: {e}")

    async def run(self) -> None:
        """
        Run the dynamic plan annotator on all tasks.

        This method:
        1. Filters out already processed tasks
        2. Uses AsyncDataGenerationJobEngine to process tasks in parallel
        3. Saves results periodically
        """
        # Filter out already processed tasks
        tasks_to_process = []
        for plan_data, processed_data in self._paired_data:
            if processed_data["task"] not in self._already_processed:
                tasks_to_process.append((plan_data, processed_data))
            else:
                print(f"Skipping already processed task: {processed_data['task']}")

        print(f"Processing {len(tasks_to_process)} tasks")

        # Use AsyncDataGenerationJobEngine to process tasks
        # The identity_fn receives the output of task_fn, which is DynamicPlanAnnotatorDataGeneratorData
        job_engine = AsyncDataGenerationJobEngine(
            data_to_process=tasks_to_process,
            task_fn=self._task_fn,
            save_fn=self.save_results,
            identity_fn=lambda result: result["task"],  # Extract task from result
            already_processed=self._already_processed,
            concurrency=4,  # Adjust based on your resources
            save_interval=60.0,  # Save every minute
            progress_interval=5.0,  # Show progress every 5 seconds
        )

        await job_engine.run()

    def save_results(
        self, new_results: list[DynamicPlanAnnotatorDataGeneratorData]
    ) -> None:
        """
        Save results to disk.

        Args:
            new_results: List of new results to save
        """
        # Append results to file
        with open(self._output_path, "a", encoding="utf-8") as f:
            for result in new_results:
                json_line = json.dumps(result)
                f.write(json_line + "\n")

    async def _task_fn(
        self, data_tuple: tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]
    ) -> DynamicPlanAnnotatorDataGeneratorData:
        """
        Process a single task.

        Args:
            data_tuple: Tuple of (plan_data, processed_data) for a task

        Returns:
            A DynamicPlanAnnotatorDataGeneratorData object
        """
        plan_data, processed_data = data_tuple
        return await self._dynamic_cot_planner.annotate(plan_data, processed_data)

    @staticmethod
    def _get_already_processed_tasks(output_path: str) -> set[str]:
        """
        Get the set of already processed tasks.

        Args:
            output_path: Path to results file

        Returns:
            Set of task IDs that have already been processed
        """
        if not os.path.exists(output_path):
            return set()

        try:
            with open(output_path, "r") as f:
                return {json.loads(line)["task"] for line in f}
        except Exception as e:
            print(f"Error loading already processed tasks: {e}")
            return set()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate dynamic replans for WebArena Lite data using existing initial plans"
    )
    parser.add_argument(
        "--input_data_path",
        type=str,
        required=True,
        help="Path to the input WebArena Lite data",
    )
    parser.add_argument(
        "--plan_data_path",
        type=str,
        required=True,
        help="Path to the initial plan data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the generated replans",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "--base_url", type=str, default=None, help="Base URL for the API"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start index for processing data"
    )
    parser.add_argument(
        "--end_index", type=int, default=None, help="End index for processing data"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def load_plan_data_and_pair_with_processed_data(
    plan_data_path: str, input_data_path: str
) -> list[tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]]:
    """
    Loads the plan data and the processed data and then pairs the ones that are for the same task together.

    Args:
        plan_data_path: Path to the initial plan data
        input_data_path: Path to the input WebArena Lite data

    Returns:
        List of paired plan data and processed data
    """
    # 1) Load the processed data
    with open(input_data_path, "r") as f:
        unprocessed_data = json.load(f)
    processed_data = sorted(
        preprocess_webarena_data(unprocessed_data),
        key=lambda x: x["task"],
    )

    print(f"Loaded {len(processed_data)} processed data items.")

    # 2) Load the plan data (.jsonl)
    plan_data: list[CotPlanAnnotatorDataGeneratorData] = []
    with open(plan_data_path, "r") as f:
        for line in f.readlines():
            plan_data.append(json.loads(line))
    plan_data = sorted(plan_data, key=lambda x: x["task"])

    print(f"Loaded {len(plan_data)} plan data items.")

    # 3) Pair the ones that are for the same task together
    paired_data = []
    for plan_data_item in plan_data:
        for processed_data_item in processed_data:
            if plan_data_item["task"] == processed_data_item["task"]:
                paired_data.append((plan_data_item, processed_data_item))
                break

    print(f"Paired {len(paired_data)} items.")
    return paired_data


async def main() -> None:
    """Main entry point for the script.
    
    Example usage:
    python3 plan_and_act/cot/data_generation/dynamic_plan.py --input_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/raw_all_2036.json" \
        --plan_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/plan_annotation_data_all_2036_DeepSeek-R1-Distill-Llama-70B.jsonl" \
        --output_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/dynamic_plan_annotation_data_all_2036.jsonl" \
        --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
        --start_index 0 \
        --end_index 5 \
        --base_url "http://localhost:8000/v1"
    """
    args = parse_args()

    # Load and pair plan data with processed data
    paired_data = load_plan_data_and_pair_with_processed_data(
        args.plan_data_path, args.input_data_path
    )

    # Apply range filtering if specified
    if args.start_index is not None or args.end_index is not None:
        paired_data = paired_data[args.start_index : args.end_index]

    print(
        f"Processing {len(paired_data)} tasks. From {args.start_index} to {args.end_index if args.end_index else 'end'}"
    )

    # Customize output path with range and model information
    output_path = args.output_path.replace(
        ".jsonl",
        f"_{args.start_index}_{args.end_index if args.end_index else 'end'}_{args.model_name.split('/')[-1]}.jsonl",
    )

    # Initialize LLM
    llm = LLM(
        model_name=args.model_name,
        max_tokens=16384,
        max_length=128000,
        temperature=0.6,  # Recommended for CoT by deepseek
        base_url=args.base_url,
    )

    # Initialize dynamic plan annotator
    dynamic_cot_planner = DynamicCoTPlanAnnotator(llm, verbose=args.verbose)

    # Initialize dynamic plan annotator data generator
    generator = DynamicCoTPlanAnnotatorDataGenerator(
        dynamic_cot_planner,
        paired_data,
        output_path,
    )

    # Run generator
    await generator.run()


if __name__ == "__main__":
    asyncio.run(main())
