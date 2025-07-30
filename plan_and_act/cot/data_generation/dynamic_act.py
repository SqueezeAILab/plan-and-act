import argparse
import asyncio
import copy
import fcntl
import json
import os

from browser_env.actions import ActionTypes
from typing_extensions import TypedDict

from plan_and_act.cot.models import (
    LLM,
    AsyncDataGenerationJobEngine,
    DynamicActAnnotatorDataGeneratorData,
    DynamicExecutorTrainingData,
    DynamicPlanAnnotatorDataGeneratorData,
    ExecutorAction,
    ProcessedData,
    ServerMessage,
    WebArenaLiteAction,
)
from plan_and_act.cot.utils import (
    clean_html,
    get_action_information,
    get_html_from_action,
    get_previous_round_html_snippet,
    get_raw_action_string_from_action,
    prepare_completions_prompt_for_reasoning_models,
    preprocess_webarena_data,
)


class ExecutorAnnotationPrompt(TypedDict):
    """
    For this annotation process, each action should return the following:
        - Current prompt: This is the prompt that will be used in the current round of execution to generate the next action. This is just for the current round representation.
        - Global prompt: Since the current prompt cannot contain all of the previous HTMLs (because of context length limitations), for the future rounds, we will need to provide the global prompt which contains the action taken in that round and only a representation of the HTML state.
    """

    current_prompt: str
    global_prompt: str


class DynamicActAnnotator:
    """
    Annotator for dynamic executor actions. This class is responsible for generating
    reasoning for each action step based on dynamic plans.
    """

    _MAX_RETRY = 3
    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool) -> None:
        self._llm = llm
        self._verbose = verbose

    async def annotate(
        self,
        dynamic_plan_data: DynamicPlanAnnotatorDataGeneratorData,
        processed_data: ProcessedData,
    ) -> DynamicActAnnotatorDataGeneratorData:
        """
        This annotator is a dynamic CoT annotator for WebArena Lite actions for our executor model.

        Unlike the previous static plan annotator, this annotator considers that each round has an
        updated plan based on the current state of the task. The executor model takes in the
        current dynamic plan and the current state of the task (which includes the previous actions,
        the HTML state, and the task query) and generates reasoning for the next immediate action
        as well as the action itself as a function call.

        Parameters:
            - dynamic_plan_data: The dynamic plan data that was generated from dynamic_plan.py
            - processed_data: The processed data object which includes all actions belonging to this task

        Returns:
            A DynamicActAnnotatorDataGeneratorData that contains the dynamic plans and executor actions
        """

        actions = sorted(
            processed_data["actions"], key=lambda x: x["index_within_task"]
        )

        if len(actions) == 0:
            raise ValueError("No actions found in the processed data.")

        # Initialize the output list
        output_executor_actions: list[ExecutorAction] = []

        # Initialize the global messages list
        global_messages: list[ServerMessage] = []

        # Extract plan history from the input data
        plan_history = dynamic_plan_data["data"]

        if self._verbose:
            initial_plan = (
                plan_history["initial_plan"]["plan"]
                if "initial_plan" in plan_history
                else "No initial plan"
            )
            print(f"=========\n\n### Initial Plan:\n\n{initial_plan}\n\n=========")

        # Iterate over all actions
        for i, webarena_lite_action in enumerate(actions):
            # Check if this is a STOP action
            is_action_exit = (
                get_action_information(webarena_lite_action)["action"]["action_type"]
                == ActionTypes.STOP
            )

            if self._verbose:
                print(
                    f">>> Action {i}: {get_raw_action_string_from_action(webarena_lite_action)}"
                )

            # Get the current dynamic plan for this round
            current_plan = ""
            if i == 0:
                current_plan = plan_history["initial_plan"]["plan"]
            elif i > 0 and i - 1 < len(plan_history["replan_steps"]):
                current_plan = plan_history["replan_steps"][i - 1]["plan"]["plan"]
            else:
                current_plan = plan_history["replan_steps"][-1]["plan"]["plan"]

            if self._verbose:
                print(f">>> Dynamic Plan for Round {i}:\n{current_plan}")

            if i == 0 and not is_action_exit:
                # First action - include initial setup
                prompts = self._prepare_first_action_prompt(
                    processed_data,
                    webarena_lite_action,
                    current_plan,
                )
            elif i == len(actions) - 1:
                # Last action - special prompt for exit
                prompts = self._prepare_last_action_prompt(
                    webarena_lite_action,
                    current_plan,
                )
            else:
                # Intermediate action with no action description since we don't have it in PlanHistory
                prompts = self._prepare_intermediate_action_prompt(
                    webarena_lite_action,
                    current_plan,
                    "",  # We don't have action descriptions in PlanHistory
                )

            if self._verbose:
                print(f"Current html: {get_html_from_action(webarena_lite_action)}")

            # Call the model to get reasoning and action
            reasoning = None
            action = None

            # Keep track of internal messages list in case of errors
            internal_messages: list[ServerMessage] = copy.deepcopy(global_messages)
            internal_messages.append(
                {
                    "role": "user",
                    "content": prompts["current_prompt"],
                }
            )

            raw_action_string = get_raw_action_string_from_action(webarena_lite_action)

            for _ in range(self._MAX_RETRY):
                try:
                    prompt = prepare_completions_prompt_for_reasoning_models(
                        internal_messages,
                        self._llm._tokenizer,  # type: ignore
                    )
                    response = await self._llm.acall_completions(prompt)
                    if response is None:
                        raise ValueError("No response from the model.")

                    # Check if the action is there
                    if (
                        "[Start of Action]" in response
                        and "[End of Action]" in response
                    ):
                        # Extract reasoning and action
                        reasoning = response.split("</think>")[0].strip()
                        action = (
                            response.split("[Start of Action]")[1]
                            .split("[End of Action]")[0]
                            .strip()
                        )
                        break

                    internal_messages.append(
                        {
                            "role": "assistant",
                            "content": "You need to ensure that you enclose the action between the '[Start of Action]' and '[End of Action]' tags.",
                        }
                    )
                except Exception as e:
                    internal_messages.append(
                        {
                            "role": "user",
                            "content": f"Error: {e}",
                        }
                    )
                    print(f"Error during action generation: {e}")

            if reasoning is None or action is None:
                raise ValueError(
                    f"Failed to generate action. reasoning: {reasoning}, action: {action}"
                )

            if self._verbose:
                print(f">>> Reasoning {i}: {reasoning}")
                print(f">>> Action {i}: {raw_action_string}")

            # Append new round messages to global messages list
            global_messages.extend(
                [
                    {
                        "role": "user",
                        "content": prompts["global_prompt"],
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>\n{reasoning}\n</think>\n\n[Start of Action]\n{action}\n[End of Action]",
                    },
                ]
            )

            output_executor_actions.append(
                ExecutorAction(
                    action=webarena_lite_action,
                    reasoning=reasoning,
                )
            )

        return DynamicActAnnotatorDataGeneratorData(
            task=processed_data["task"],
            data=DynamicExecutorTrainingData(
                task=processed_data["task"],
                initial_html_state=processed_data["initial_html_state"],
                plan_history=plan_history,
                executor_actions=output_executor_actions,
            ),
        )

    def _prepare_first_action_prompt(
        self,
        processed_data: ProcessedData,
        webarena_lite_action: WebArenaLiteAction,
        current_plan: str,
    ) -> ExecutorAnnotationPrompt:
        """
        This first action prompt includes:
            - System instructions explaining the training data generation task
            - The user task and CURRENT dynamic plan
            - The current HTML state
            - The ground truth action and resulting HTML state (for training data generation)
        """

        prefix = """# Goal
You are an annotator LLM helping to generate training data for an executor agent model with dynamic planning capabilities. The executor agent is a powerful model that can complete complex web navigation tasks by issuing web actions such as clicking, typing, selecting, and more. The agent takes the current HTML state, user's task, and the CURRENT DYNAMIC PLAN, then decides what immediate action to take.

For this training data generation task, for each round of execution, you will be given:
1. The user's task and the CURRENT DYNAMIC PLAN for this round (plans are updated every round based on the current state)
2. The previous rounds of actions that were taken (as previous conversations)
3. The current HTML state
4. The ground truth action that the executor agent needs to take at this round
5. The description of the ground truth action

Your job is to generate expert-level reasoning that the executor agent should use when deciding its next action (which is the ground truth action). The executor agent itself will only have access to the task, CURRENT dynamic plan, and current HTML state - it won't know the ground truth action or future HTML state. Your reasoning should demonstrate the analytical process of discovering and choosing this action through careful examination of the page and understanding of the task.

## Action Formats
The executor agent will be trained to output actions in these formats:
- Actions:
1. Click: `do(action="Click", element="<element_id>")`
2. Type: `do(action="Type", element="<element_id>", argument="<text>")`
3. Scroll: `do(action="Scroll Down")` or `do(action="Scroll Up")`
4. Hover: `do(action="Hover", element="<element_id>")`
5. Select: `do(action="Select Dropdown Option", element="<element_id>", argument="<value>")`
6. Search: `do(action="Search", element="<element_id>", argument="<text>")`
7. Back: `go_backward()`
8. Exit: `exit(message="<answer>")`

- Comments:
Before outputting the action, it the executor agent will also output a comment `# Note:` and/or `# Element:` to provide a very brief description of the targeted element or a stored note and follow with the action string. Especially, the '# Note:' comment is very very important since it is used to store important information that the executor agent will need to remember for later steps in the current task.

- Examples:
```
# Element: the 'Next Page' link
do(action="click", element="17")
```

```
# Note: These places may satisfy the task requirement:
# 1: Ithaca Tompkins Regional Airport, Neimi Road, Town of Lansing, Tompkins County, New York, 13068, United States
# Element: the direction sign on the right side of the Go icon
do(action="click", element="15")
```

# Task
Here is the task the agent needs to complete:

{task}"""

        postfix = """
        
---

Let's generate the reasoning for the first action. 

## Round 1

Current HTML state:

{current_html}

Action to be taken by the executor agent in this round (you also need to generate this exact action - including the comments - in your response) for which you are going to generate the reasoning:

{action}

Description of the action:

{action_description}

Based on the above, generate expert reasoning that shows how the executor agent would analyze the current situation and arrive at this action. The executor agent itself won't know the ground truth action or the description of the action - your reasoning should demonstrate the analytical process of discovering and choosing this action through careful examination of the page and understanding of the task. Hence, you should:

- Demonstrate expert-level understanding of web navigation and DOM manipulation
- Show how to analyze the current HTML state to confidently identify the correct element to interact with by including any relevant element attributes, text content, or positioning that will help locate the target
- Predict what changes the action will cause to the webpage state (how does the action change the current HTML state to the next HTML state)
- Explain why this action aligns with the current step in the DYNAMIC plan
- Act as a "world model" by anticipating how the webpage will respond to the action
- Pretend as if you are the executor agent itself. Hence, you MUST always talk in "I" or "we" instead. Avoid mentioning the "executor agent" or "agent" in your reasoning.

After your thinking, output the action enclosed in '[Start of Action]' and '[End of Action]' tags."""

        current_prompt = (
            prefix
            + """

# Current Dynamic Plan (for this round)
Here is the current dynamic plan the agent should follow for this round:

{plan}"""
            + postfix
        )

        global_prompt = prefix + postfix

        # Format the ground truth action string
        action_str = get_raw_action_string_from_action(webarena_lite_action)

        # Current Prompt
        ## Get current state
        current_html = clean_html(
            get_html_from_action(webarena_lite_action),
            prettify=True,
            strip_annotation=False,
            dont_remove_ids={
                get_action_information(webarena_lite_action)["action"]["element_id"]
            },
        )

        # Global Prompt
        ## Get current state
        current_html_snippet = get_previous_round_html_snippet(
            webarena_lite_action, strip_annotation=False
        )

        # We don't have action descriptions in the PlanHistory structure
        action_description_text = ""

        return ExecutorAnnotationPrompt(
            current_prompt=current_prompt.format(
                task=processed_data["task"],
                plan=current_plan,
                current_html=current_html,
                action=action_str,
                action_description=action_description_text,
            ),
            global_prompt=global_prompt.format(
                task=processed_data["task"],
                current_html=current_html_snippet,
                action=action_str,
                action_description=action_description_text,
            ),
        )

    def _prepare_intermediate_action_prompt(
        self,
        webarena_lite_action: WebArenaLiteAction,
        current_plan: str,
        action_description: str = "",
    ) -> ExecutorAnnotationPrompt:
        """
        This intermediate action prompt includes:
            - The current round number
            - The UPDATED dynamic plan for this round
            - The current HTML state
            - The ground truth action that was taken
        """

        prefix = """Let's continue with the next round of execution. You will be given the current HTML of the webpage, the UPDATED dynamic plan for this round, the ground truth action that was taken, and the description of the action.

## Round {round_number}"""

        postfix = """

Current HTML state:

{current_html}

Action to be taken by the executor agent in this round (you also need to generate this exact action - including the '# Note:' comment - in your response) for which you are going to generate the reasoning:

{action}

Description of the action:

{action_description}

Based on the above, generate expert reasoning that shows how the executor agent would analyze the current situation and arrive at this action. The executor agent itself won't know the ground truth action or the description of the action - your reasoning should demonstrate the analytical process of discovering and choosing this action through careful examination of the page and understanding of the task. Hence, you should:

- Reflect on the previous rounds of actions and the changes they made to the HTML state
- Consider the UPDATED dynamic plan for this round, which may have changed based on the current state
- Demonstrate expert-level understanding of web navigation and DOM manipulation
- Show how to analyze the current HTML state to confidently identify the correct element to interact with by including any relevant element attributes, text content, or positioning that will help locate the target
- Predict what changes the action will cause to the webpage state (how does the action change the current HTML state to the next HTML state)
- Explain why this action aligns with the current step in the UPDATED dynamic plan
- Act as a "world model" by anticipating how the webpage will respond to the action
- Pretend as if you are the executor agent itself. Hence, you MUST always talk in "I" or "we" instead. Avoid mentioning the "executor agent" or "agent" in your reasoning.

After your thinking, output the action enclosed in '[Start of Action]' and '[End of Action]' tags."""

        current_prompt = (
            prefix
            + """

# Updated Dynamic Plan (for this round)
Here is the updated dynamic plan the agent should follow for this round:

{current_plan}"""
            + postfix
        )

        global_prompt = prefix + postfix

        round_number = webarena_lite_action["index_within_task"]

        if self._verbose:
            print(f">> Round {round_number}")

        # Format the ground truth action string
        action_str = get_raw_action_string_from_action(webarena_lite_action)

        # Get current HTML state
        current_html = clean_html(
            get_html_from_action(webarena_lite_action),
            prettify=True,
            strip_annotation=False,
            dont_remove_ids={
                get_action_information(webarena_lite_action)["action"]["element_id"]
            },
        )

        # For global prompt, use HTML snippets instead of full HTML
        current_html_snippet = get_previous_round_html_snippet(
            webarena_lite_action, strip_annotation=False
        )

        # Use the description of the action if provided
        action_description_text = (
            action_description.get("action_description", "")
            if isinstance(action_description, dict)
            else action_description
        )

        return ExecutorAnnotationPrompt(
            current_prompt=current_prompt.format(
                round_number=round_number,
                current_plan=current_plan,
                current_html=current_html,
                action=action_str,
                action_description=action_description_text,
            ),
            global_prompt=global_prompt.format(
                round_number=round_number,
                current_html=current_html_snippet,
                action=action_str,
                action_description=action_description_text,
            ),
        )

    def _prepare_last_action_prompt(
        self,
        webarena_lite_action: WebArenaLiteAction,
        current_plan: str,
    ) -> ExecutorAnnotationPrompt:
        """
        This last action prompt is for the last `exit()` action. It asks the annotator to provide
        reasoning for why the task is done, considering the final dynamic plan.
        """

        prefix = """Let's continue with the next round of execution. This round is the last round of the task. Now, you need to provide the reasoning for why you think the task is done, why you don't need to take another action, and what your final answer/message is. If your task requires you to make an analysis, then you should provide that analysis as well.

## Round {round_number}"""

        postfix = """

Current HTML state:

{html}

Last exit action that was taken:

{action}

Based on the above, generate expert reasoning that shows how the executor agent should think about choosing the `exit` action by pretending to be the executor agent itself. Remember, the agent itself won't know the ground truth action and the ground truth exit message to return - your reasoning should show how it can expertly analyze the current state and provide excellent answers. You MUST be comprehensive and thorough in your reasoning. You are allowed to think as much as you need to before outputting the next action.

After your thinking, output the action enclosed in '[Start of Action]' and '[End of Action]' tags."""

        current_prompt = (
            prefix
            + """

# Final Dynamic Plan
Here is the final dynamic plan the agent had for this round:

{current_plan}"""
            + postfix
        )

        global_prompt = prefix + postfix

        round_number = webarena_lite_action["index_within_task"] + 1

        action_str = get_raw_action_string_from_action(webarena_lite_action)

        current_prompt_html = clean_html(
            get_html_from_action(webarena_lite_action),
            prettify=True,
            strip_annotation=False,
        )

        global_prompt_html = get_previous_round_html_snippet(
            webarena_lite_action, strip_annotation=False
        )

        return ExecutorAnnotationPrompt(
            current_prompt=current_prompt.format(
                round_number=round_number,
                current_plan=current_plan,
                html=current_prompt_html,
                action=action_str,
            ),
            global_prompt=global_prompt.format(
                round_number=round_number,
                html=global_prompt_html,
                action=action_str,
            ),
        )


class DynamicActAnnotatorDataGenerator:
    _dynamic_act_annotator: DynamicActAnnotator

    # Input data
    _annotation_input_data: list[
        tuple[DynamicPlanAnnotatorDataGeneratorData, ProcessedData]
    ]
    _already_processed: set[str]

    # Output paths
    _output_path: str

    # Results aggregator
    _results: list[DynamicActAnnotatorDataGeneratorData]

    def __init__(
        self,
        dynamic_act_annotator: DynamicActAnnotator,
        annotation_input_data: list[
            tuple[DynamicPlanAnnotatorDataGeneratorData, ProcessedData]
        ],
        output_path: str,
    ) -> None:
        self._dynamic_act_annotator = dynamic_act_annotator
        self._annotation_input_data = annotation_input_data
        self._already_processed = (
            DynamicActAnnotatorDataGenerator._get_already_processed_tasks(output_path)
        )
        self._results = []
        self._output_path = output_path

    async def run(self) -> None:
        # 1) Determine which items have not yet been processed. In the meanwhile, also populate the results list.
        unprocessed_data = [
            d
            for d in self._annotation_input_data
            if d[0]["task"] not in self._already_processed
        ]

        # 2) Create our async job engine
        engine = AsyncDataGenerationJobEngine[
            tuple[DynamicPlanAnnotatorDataGeneratorData, ProcessedData],
            DynamicActAnnotatorDataGeneratorData,
            str,
        ](
            data_to_process=unprocessed_data,
            task_fn=self._task_fn,
            save_fn=self.save_results,
            concurrency=4,
            save_interval=60.0,
            progress_interval=5.0,
        )

        # 3) Run the engine
        await engine.run()

    def save_results(self, new_results: list) -> None:
        """
        Appends a list of JSON-serializable objects to a file in JSON Lines format.
        Ensures that only one process writes to the file at a time using file locking.
        """
        with open(self._output_path, "a", encoding="utf-8") as f:
            fcntl.flock(
                f, fcntl.LOCK_EX
            )  # Acquire exclusive lock (blocks other writers)

            for result in new_results:
                json.dump(result, f)
                f.write("\n")

            fcntl.flock(f, fcntl.LOCK_UN)  # Release lock

    async def _task_fn(
        self, data: tuple[DynamicPlanAnnotatorDataGeneratorData, ProcessedData]
    ) -> DynamicActAnnotatorDataGeneratorData:
        return await self._dynamic_act_annotator.annotate(data[0], data[1])

    @staticmethod
    def _get_already_processed_tasks(output_path: str) -> set[str]:
        if not os.path.exists(output_path):
            return set()

        already_processed = set()
        with open(output_path, "r") as f:
            for line in f.readlines():
                already_processed.add(json.loads(line)["task"])

        print(f"Already processed {len(already_processed)} tasks.")
        return already_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--dynamic_plan_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--start_index", type=int, required=False, default=0)
    parser.add_argument("--end_index", type=int, required=False, default=None)
    parser.add_argument("--verbose", type=bool, required=False, default=False)
    parser.add_argument("--base_url", type=str, required=False, default=None)
    return parser.parse_args()


def load_dynamic_plan_data_and_pair_with_processed_data(
    dynamic_plan_data_path: str, input_data_path: str
) -> list[tuple[DynamicPlanAnnotatorDataGeneratorData, ProcessedData]]:
    """
    Loads both the dynamic plan data and the processed data and then pairs the ones that are
    for the same task together into tuples so that we can use them together during
    DynamicActAnnotatorDataGenerator.
    """
    # 1) Load the processed data
    with open(input_data_path, "r") as f:
        unprocessed_data = json.load(f)
    processed_data = sorted(
        preprocess_webarena_data(unprocessed_data),
        key=lambda x: x["task"],
    )

    print(f"Loaded {len(processed_data)} processed data items.")

    # 2) Load the dynamic plan data (.jsonl)
    dynamic_plan_data: list[DynamicPlanAnnotatorDataGeneratorData] = []
    with open(dynamic_plan_data_path, "r") as f:
        for line in f.readlines():
            plan_data = json.loads(line)
            # Ensure the data has the right structure
            if "task" in plan_data and "data" in plan_data:
                dynamic_plan_data.append(plan_data)
            else:
                print(f"Warning: Skipping malformed plan data: {plan_data}")

    dynamic_plan_data = sorted(dynamic_plan_data, key=lambda x: x["task"])

    print(f"Loaded {len(dynamic_plan_data)} dynamic plan data items.")

    # 3) Pair the ones that are for the same task together
    paired_data = []
    for plan_data_item in dynamic_plan_data:
        for processed_data_item in processed_data:
            if plan_data_item["task"] == processed_data_item["task"]:
                paired_data.append((plan_data_item, processed_data_item))
                break

    print(f"Paired {len(paired_data)} data items.")

    # Assert that the tasks are the same for each tuple in the paired data
    assert all(
        plan_data_item["task"] == processed_data_item["task"]
        for plan_data_item, processed_data_item in paired_data
    ), "The tasks are not the same for each tuple in the paired data."

    return paired_data


if __name__ == "__main__":
    """
    Example usage:
    python3 plan_and_act/cot/data_generation/dynamic_act.py --input_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/raw_all_2036.json" \
        --dynamic_plan_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/dynamic_plan_annotation_data_all_2036_0_end_with_initial_html_state_DeepSeek-R1-Distill-Llama-70B.jsonl" \
        --output_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/dynamic_act_annotation_data_all_2036.jsonl" \
        --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
        --start_index 0 \
        --end_index 100 \
        --base_url "http://localhost:8000/v1"
    """
    args = parse_args()

    llm = LLM(
        model_name=args.model_name,
        max_tokens=16384,
        max_length=128000,
        temperature=0.6,  # Recommended for CoT by deepseek
        base_url=args.base_url,
    )

    dynamic_act_annotator = DynamicActAnnotator(llm=llm, verbose=args.verbose)
    paired_data = load_dynamic_plan_data_and_pair_with_processed_data(
        args.dynamic_plan_data_path, args.input_data_path
    )
    annotation_input_data = paired_data[args.start_index : args.end_index]

    # Customize output path with range and model information
    output_path = args.output_path.replace(
        ".jsonl",
        f"_{args.start_index}_{args.end_index if args.end_index else 'end'}_{args.model_name.split('/')[-1]}.jsonl",
    )

    data_generator = DynamicActAnnotatorDataGenerator(
        dynamic_act_annotator=dynamic_act_annotator,
        annotation_input_data=annotation_input_data,
        output_path=output_path,
    )

    asyncio.run(data_generator.run())
