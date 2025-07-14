import json
import random
import re
from typing import Literal

from tqdm import tqdm

from plan_and_act.cot.inference.plan import CoTPlanner
from plan_and_act.cot.models import (
    LLM,
    CotSyntheticPlanDataGeneratorData,
    DynamicPlanAnnotatorDataGeneratorData,
    DynamicPlanInferenceInput,
    LlamaFactoryData,
    Plan,
    PlanInferenceInput,
    ServerMessage,
    TorchTuneData,
)
from plan_and_act.cot.utils import (
    clean_html,
    format_training_data_for_llama_factory,
    format_training_data_for_torchtune,
    get_raw_action_string_from_action,
    prepare_completions_prompt_for_reasoning_models,
)


class DynamicCoTPlanner:
    """
    Dynamic replanner model that generates updated plans during task execution.

    This class:
    1. Takes the current state, previous plan, and previous actions as input
    2. Generates an updated plan based on the current state
    """

    _MAX_RETRY = 3
    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool = False) -> None:
        """
        Initialize the dynamic replanner.

        Args:
            llm: The language model to use for generating replans
            verbose: Whether to print verbose output
        """
        self._llm = llm
        self._verbose = verbose

    async def replan(self, input: DynamicPlanInferenceInput) -> Plan:
        """
        Run the dynamic replanner model during inference.

        This method:
        1. Constructs a prompt based on the current state and previous plan
        2. Generates an updated plan based on the current state

        Args:
            input: The input data for dynamic replanning

        Returns:
            The updated plan
        """
        messages: list[ServerMessage] = [
            {"role": "user", "content": self.construct_prompt(input)},
        ]

        if self._verbose:
            print(f"Prompt for replanning:\n{messages[0]['content']}")

        reasoning = None
        plan_text = None

        for _ in range(self._MAX_RETRY):
            try:
                prompt = prepare_completions_prompt_for_reasoning_models(
                    messages,
                    self._llm._tokenizer,  # type: ignore
                )

                response = await self._llm.acall_completions(prompt)
                if response is None:
                    raise ValueError("Failed to generate updated plan.")

                # Extract the reasoning and plan
                if "</think>" in response:
                    reasoning = response.split("</think>")[0].strip()
                else:
                    reasoning = ""

                # Extract the plan
                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    plan_text = (
                        response.split("[Start of Plan]")[1]
                        .split("[End of Plan]")[0]
                        .strip()
                    )
                    break
                else:
                    # If no plan found, ask for it explicitly
                    messages.extend(
                        [
                            {"role": "assistant", "content": response},
                            {
                                "role": "user",
                                "content": "You need to provide an updated plan enclosed between [Start of Plan] and [End of Plan] tags.",
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

        if plan_text is None:
            raise ValueError("Failed to generate updated plan.")

        # Create the updated plan
        updated_plan = Plan(
            reasoning=reasoning or "",
            plan=plan_text,
        )

        if self._verbose:
            print(f"Reasoning: {reasoning}")
            print(f"Updated plan: {plan_text}")

        return updated_plan

    def construct_prompt(self, input: DynamicPlanInferenceInput) -> str:
        """
        Construct the prompt for the dynamic replanner model.

        Args:
            input: The input data for dynamic replanning

        Returns:
            The prompt for the dynamic replanner model
        """
        task = input["task"]
        previous_plan = input["previous_plan"]
        current_html_state = clean_html(
            input["current_html_state"], prettify=True, strip_annotation=True
        )

        # Format previous actions for display in the prompt
        formatted_previous_steps = ""
        if len(input["previous_actions"]) > 0:
            previous_actions = []
            for action in input["previous_actions"]:
                if "conversations" in action and "index_within_task" in action:
                    previous_actions.append(get_raw_action_string_from_action(action))
                else:
                    previous_actions.append(action["action_str"])

            # Process each action and create formatted strings
            formatted_actions = []

            # Process WebArenaLiteAction objects first (if any)
            for i, action in enumerate(previous_actions):
                formatted_actions.append(f"* Action {i+1}: {action}")

            formatted_previous_steps = "\n\n".join(formatted_actions)

        prompt = f"""## Your Role: The ExecutionReplanner

You are an ExecutionReplanner helping a user complete a web task. Your job is to analyze the current state of execution and update the plan as needed.

## What is Replanning?

Replanning is the process of updating an existing plan based on:
- Actions that have already been taken
- The current state of the webpage
- New information that has been revealed
- Changes in the environment since the original plan was created

## Your Task: Generate an Updated Plan

You need to:
1. Analyze the current state of execution (actions taken so far and current HTML)
2. Compare it with the existing plan
3. Generate a comprehensive updated plan that reflects the current state and guides future actions

## Task Query
Here is the task that the user wants to accomplish: 

{task}

## Previous Plan
```
{previous_plan["plan"]}
```

## Previous Actions Taken
{formatted_previous_steps}

## Current HTML State (what the user sees RIGHT NOW)
{current_html_state}

Based on this information, please:

1. Analyze the current state of execution and how it relates to the previous plan
2. Determine what steps have been completed and what remains to be done
3. Consider how the current HTML state affects the remaining steps
4. Generate a comprehensive updated plan that will guide the user to task completion

-**Important:** Before generating your updated plan, you should write down your detailed chain of thought and reasoning. You MUST be comprehensive and thorough in your reasoning. You are allowed to think as much as you need to before outputting your plan.
- After your thinking process, provide your updated plan between "[Start of Plan]" and "[End of Plan]" tags."""

        return prompt


def create_dynamic_plan_training_data(
    model_name: str,
    dynamic_plan_data_path: str,
    synthetic_plan_data_path: str,
    output_path: str,
    type: Literal["llama_factory", "torchtune"],
    validation_split: float = 0.05,
) -> tuple[
    list[TorchTuneData | LlamaFactoryData], list[TorchTuneData | LlamaFactoryData]
]:
    """
    Create training data for the dynamic replanner model.

    This function processes the dynamic plan annotator data and converts it
    into a format suitable for training, such as TorchTune. It uses:
    - CoTPlanner for generating initial plan prompts
    - DynamicCoTPlanner for generating replan prompts

    Args:
        model_name: The name of the model to use for tokenization
        dynamic_plan_data_path: Path to the dynamic plan annotator data
        synthetic_plan_data_path: Path to the synthetic plan data
        output_path: Path to save the training data
    """
    data: list[DynamicPlanAnnotatorDataGeneratorData] = []
    with open(dynamic_plan_data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    synthetic_plan_data: list[CotSyntheticPlanDataGeneratorData] = []
    with open(synthetic_plan_data_path, "r") as f:
        for line in f.readlines():
            synthetic_plan_data.append(json.loads(line))

    # Initialize both planners
    cot_planner = CoTPlanner(llm=LLM(0, 0, model_name))
    dynamic_planner = DynamicCoTPlanner(llm=LLM(0, 0, model_name))

    def create_data_and_save(
        data: list[DynamicPlanAnnotatorDataGeneratorData],
        synthetic_plan_data: list[CotSyntheticPlanDataGeneratorData],
        output_path: str,
    ) -> list[TorchTuneData | LlamaFactoryData]:

        # Prepare data structures for both types of examples
        initial_plan_pairs: list[tuple[PlanInferenceInput, Plan]] = []
        replan_pairs: list[tuple[DynamicPlanInferenceInput, Plan]] = []

        for item in data:
            task = item["task"]
            plan_history = item["data"]
            initial_plan = plan_history["initial_plan"]

            # Sort replan steps by index within task
            sorted_replan_steps = sorted(
                plan_history["replan_steps"], key=lambda x: x["index_within_task"]
            )

            # Get the initial HTML state from the first replan step
            # Typically, the first replan step will have the HTML state after the first action
            # If available, use an initial_html_state field directly
            initial_html_state = plan_history["initial_html_state"]
            # Create an example for the initial plan
            initial_input = PlanInferenceInput(
                task=task, initial_html_state=initial_html_state
            )
            initial_plan_pairs.append((initial_input, initial_plan))

            # Create a training example for each replan step
            for step in sorted_replan_steps:
                # The plan before this replan step is either:
                # - The initial plan (for the first step)
                # - The plan from the previous step (for subsequent steps)
                previous_plan = initial_plan
                if step["index_within_task"] > 0:
                    # Find the most recent previous step
                    previous_steps = [
                        s
                        for s in sorted_replan_steps
                        if s["index_within_task"] < step["index_within_task"]
                    ]
                    if len(previous_steps) > 0:
                        previous_plan = sorted(
                            previous_steps, key=lambda x: x["index_within_task"]
                        )[-1]["plan"]

                # Create the replan input
                replan_input = DynamicPlanInferenceInput(
                    task=task,
                    previous_plan=previous_plan,
                    current_html_state=step["html_state"],
                    previous_actions=sorted(
                        step["previous_executor_actions"],
                        key=lambda x: x["index_within_task"],
                    ),
                )

                # Use the updated plan from the step as the output
                replan_pairs.append((replan_input, step["plan"]))

        # Process the synthetic plan data and add it to the initial plan pairs
        # You need to fix some things about the synthetic plan data
        fixed = 0
        for item in synthetic_plan_data:
            for plan_training_data in item["datas"]:

                task = plan_training_data["task"]
                if "Initial HTML State from Example Index:" in task:
                    task = re.sub(
                        r"\nInitial HTML State from Example Index: \d+", "", task
                    )
                    fixed += 1

                initial_input = PlanInferenceInput(
                    task=task,
                    initial_html_state=plan_training_data["initial_html_state"],
                )
                initial_plan_pairs.append((initial_input, plan_training_data["plan"]))

        print(f"Fixed: {fixed}")
        print(f"Initial plan pairs: {len(initial_plan_pairs)}")
        print(f"Replan pairs: {len(replan_pairs)}")

        # Generate training data in TorchTune format
        torchtune_data = []
        formatting_function = (
            format_training_data_for_torchtune
            if type == "torchtune"
            else format_training_data_for_llama_factory
        )

        # Process initial plan examples
        for input_tuple, output_plan in tqdm(
            initial_plan_pairs, desc="Creating initial plan training data"
        ):
            prompt = cot_planner.construct_prompt(input_tuple)

            # Special case: Sometimes the reasoning was ending with '</think>' So we need to remove it.
            if "</think>" in output_plan["reasoning"]:
                output_plan["reasoning"] = (
                    output_plan["reasoning"].split("</think>")[0].strip()
                )
            else:
                output_plan["reasoning"] = output_plan["reasoning"].strip()

            # Format the expected output
            output_text = f"{output_plan['reasoning']}\n</think>\n\n[Start of Plan]\n{output_plan['plan']}\n[End of Plan]"

            # Create the formatted training data
            formatted_data = formatting_function(
                messages=[{"role": "user", "content": prompt}],
                tokenizer=cot_planner._llm._tokenizer,  # type: ignore
                output=output_text,
            )
            torchtune_data.append(formatted_data)

        # Process replan examples
        for input_tuple, output_plan in tqdm(
            replan_pairs, desc="Creating replan training data"
        ):
            prompt = dynamic_planner.construct_prompt(input_tuple)

            # Special case: Sometimes the reasoning was ending with '</think>' So we need to remove it.
            if "</think>" in output_plan["reasoning"]:
                output_plan["reasoning"] = (
                    output_plan["reasoning"].split("</think>")[0].strip()
                )
            else:
                output_plan["reasoning"] = output_plan["reasoning"].strip()

            # Format the expected output
            output_text = f"{output_plan['reasoning']}\n</think>\n\n[Start of Plan]\n{output_plan['plan']}\n[End of Plan]"

            # Create the formatted training data
            formatted_data = formatting_function(
                messages=[{"role": "user", "content": prompt}],
                tokenizer=dynamic_planner._llm._tokenizer,  # type: ignore
                output=output_text,
            )
            torchtune_data.append(formatted_data)

        # Save the combined data
        print(
            f"Saving {len(torchtune_data)} examples ({len(initial_plan_pairs)} initial plans, {len(replan_pairs)} replans) to {output_path}"
        )
        with open(output_path, "w") as f:
            json.dump(torchtune_data, f, indent=4, ensure_ascii=False)

        return torchtune_data

    random.seed(42)
    random.shuffle(data)
    random.shuffle(synthetic_plan_data)

    train_data = data[: int(len(data) * (1 - validation_split))]
    train_synthetic_plan_data = synthetic_plan_data[
        : int(len(synthetic_plan_data) * (1 - validation_split))
    ]
    print(f"Train data: {len(train_data)} + {len(train_synthetic_plan_data)}")
    val_data = data[int(len(data) * (1 - validation_split)) :]
    val_synthetic_plan_data = synthetic_plan_data[
        int(len(synthetic_plan_data) * (1 - validation_split)) :
    ]
    print(f"Val data: {len(val_data)} + {len(val_synthetic_plan_data)}")

    train_torchtune_data = create_data_and_save(
        train_data,
        train_synthetic_plan_data,
        output_path.replace(".json", "_train.json"),
    )
    val_torchtune_data = create_data_and_save(
        val_data,
        val_synthetic_plan_data,
        output_path.replace(".json", "_val.json"),
    )

    return train_torchtune_data, val_torchtune_data
