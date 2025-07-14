import json
import random
from typing import Literal, TypedDict

from tqdm import tqdm

from plan_and_act.cot.models import (
    LLM,
    ActInferenceOutput,
    ActInferencePreviousRound,
    DynamicActAnnotatorDataGeneratorData,
    DynamicExecutorInferenceInput,
    ExecutorAction,
    LlamaFactoryData,
    PlanHistory,
    ServerMessage,
    TorchTuneData,
)
from plan_and_act.cot.utils import (
    clean_html,
    format_training_data_for_llama_factory,
    format_training_data_for_torchtune,
    get_action_information,
    get_action_information_from_action_str,
    get_html_from_action,
    get_previous_round_html_snippet,
    get_previous_round_html_snippet_from_html,
    get_raw_action_string_from_action,
    prepare_completions_prompt_for_reasoning_models,
)


class DynamicExecutor:
    """
    This executor is responsible for executing web tasks based on dynamic plans.
    Unlike the static CoTExecutor, this executor handles plans that update with each round.
    It generates actions based on the current state, the most recent plan from the plan history,
    and previous actions.
    """

    _FIRST_ROUND_PROMPT_PREFIX = """# Goal
You are the Dynamic Executor Agent, a powerful assistant that can complete complex web navigation tasks by issuing web actions such as clicking, typing, selecting, and more. You will be provided with:
1. The user's task and CURRENT execution plan (which may be updated in future rounds)
2. The previous rounds of actions that were taken (if any)
3. The current HTML state of the webpage

Your goal is to analyze the current state and determine the next immediate action to take that will progress toward completing the task. You should:

- Carefully analyze the current HTML state to identify the correct elements to interact with
- Consider how your action will change the webpage state
- Ensure your action aligns with the current step in the CURRENT plan
- Keep track of important information using # Note: comments for future steps
- **Important:** Before outputting your action (and before adding any comments), you should write down your detailed chain of thought and reasoning. You MUST be comprehensive and thorough in your reasoning. You are allowed to think as much as you need to before outputting the action.

You must output actions in one of these formats:
1. Click: `do(action="Click", element="<element_id>")`
2. Type: `do(action="Type", element="<element_id>", argument="<text>")`
3. Scroll: `do(action="Scroll Down")` or `do(action="Scroll Up")`
4. Hover: `do(action="Hover", element="<element_id>")`
5. Select: `do(action="Select Dropdown Option", element="<element_id>", argument="<value>")`
6. Search: `do(action="Search", element="<element_id>", argument="<text>")`
7. Back: `go_backward()`
8. Exit: `exit(message="<answer>")`

Before outputting an action, you can add comments:
- Use `# Note:` to store important information needed for future steps
- Use `# Element:` to briefly describe the targeted element

Examples:
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
Here is the task that the user wants to complete: {task}"""

    _FIRST_ROUND_PROMPT_POSTFIX = """
    
---

Let's start with the first round of action.

## Round 1

Current HTML state:

{current_html}

Based on the task, CURRENT plan, and current HTML state, what should be the next immediate action? First, write down your detailed reasoning and then output your action enclosed between the tags "[Start of Action]" and "[End of Action]"."""

    _INTERMEDIATE_ROUND_PROMPT_PREFIX = """Let's continue with the next round of execution.

## Round {round_number}"""

    _INTERMEDIATE_ROUND_PROMPT_POSTFIX = """

Current HTML state:

{current_html}

Based on the task, UPDATED plan, and current HTML state, what should be the next immediate action? First, write down your detailed reasoning and then output your action enclosed between the tags "[Start of Action]" and "[End of Action]"."""

    _MAX_RETRY = 3
    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool = False) -> None:
        self._llm = llm
        self._verbose = verbose

    async def act(
        self,
        input: DynamicExecutorInferenceInput,
    ) -> ActInferenceOutput:
        """
        This function runs the Dynamic Executor model during evaluation.
        It takes into account the dynamic plan history and generates the next action.
        """

        messages = self.construct_prompt(input)

        reasoning = None
        action_str = None
        for _ in range(self._MAX_RETRY):
            try:
                prompt = prepare_completions_prompt_for_reasoning_models(
                    messages,
                    self._llm._tokenizer,  # type: ignore
                )

                print(">> Calling the dynamic executor")
                response = await self._llm.acall_completions(prompt)
                if response is None:
                    raise ValueError("Failed to generate action.")

                if "[Start of Action]" in response and "[End of Action]" in response:
                    action_str = (
                        response.split("[Start of Action]")[1]
                        .split("[End of Action]")[0]
                        .strip()
                    )
                    reasoning = reasoning or response.split("</think>")[0].strip()
                    break

                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": "You need to ensure that you enclose the action between the '[Start of Action]' and '[End of Action]' tags.",
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
                print(f"Error during action generation: {e}")

        if reasoning is None or action_str is None:
            raise ValueError(
                f"Failed to generate action. reasoning: {reasoning}, action_str: {action_str}"
            )

        if self._verbose:
            print(f"Reasoning: {reasoning}")
            print(f"Action: {action_str}")

        return ActInferenceOutput(
            action_str=action_str,
            reasoning=reasoning,
        )

    def construct_prompt(
        self,
        input: DynamicExecutorInferenceInput,
    ) -> list[ServerMessage]:
        """
        Constructs the prompt for the dynamic executor to generate the next action.
        This takes into account the dynamic plan history to provide the current plan.
        """
        task = input["task"]
        plan_history = input["plan_history"]
        previous_rounds = input["previous_rounds"]
        current_round = input["current_round"]

        global_messages: list[ServerMessage] = []

        # Iterate over all past actions to construct the conversation history.
        # Each round will use the appropriate plan from the plan history.
        for i, executor_action in enumerate(previous_rounds):
            round_number = i + 1

            # Get the HTML and action string from the executor_action
            if "act" in executor_action and "uncleaned_html" in executor_action:
                reasoning = executor_action["act"]["reasoning"]
                action_str = executor_action["act"]["action_str"]
                current_html = get_previous_round_html_snippet_from_html(
                    html=executor_action["uncleaned_html"],
                    target_element_id=get_action_information_from_action_str(
                        action_str
                    )["action"]["element_id"],
                ).strip()
            else:
                assert "action" in executor_action
                reasoning = executor_action["reasoning"]
                action_str = get_raw_action_string_from_action(
                    executor_action["action"]
                )
                current_html = get_previous_round_html_snippet(
                    executor_action["action"], strip_annotation=False
                ).strip()

            # Get the appropriate plan for this round
            if i == 0:
                # First round uses the initial plan
                current_plan = plan_history["initial_plan"]["plan"]
                prompt = DynamicExecutor._FIRST_ROUND_PROMPT_PREFIX.format(
                    task=task,
                ) + DynamicExecutor._FIRST_ROUND_PROMPT_POSTFIX.format(
                    current_html=current_html,
                )
            else:
                # Subsequent rounds use the replanned plans if available
                if i - 1 < len(plan_history["replan_steps"]):
                    current_plan = plan_history["replan_steps"][i - 1]["plan"]["plan"]
                else:
                    # If no replan is available for this round, use the last available plan
                    if len(plan_history["replan_steps"]) > 0:
                        current_plan = plan_history["replan_steps"][-1]["plan"]["plan"]
                    else:
                        current_plan = plan_history["initial_plan"]["plan"]

                prompt = DynamicExecutor._INTERMEDIATE_ROUND_PROMPT_PREFIX.format(
                    round_number=round_number,
                ) + DynamicExecutor._INTERMEDIATE_ROUND_PROMPT_POSTFIX.format(
                    current_html=current_html,
                )

            # Add the user+assistant conversation history
            global_messages.extend(
                [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>\n{reasoning}\n</think>\n\n[Start of Action]\n{action_str}\n[End of Action]",
                    },
                ]
            )

        # Now add the HTML state for the current round of action.
        round_number = len(previous_rounds) + 1

        # Get the HTML for the current round
        if "act" in current_round and "uncleaned_html" in current_round:
            current_html = clean_html(
                html=current_round["uncleaned_html"],
                prettify=True,
                strip_annotation=False,
            ).strip()
        else:
            assert "action" in current_round
            try:
                dont_remove_ids = {
                    get_action_information(current_round["action"])["action"][
                        "element_id"
                    ]
                }
            except Exception:
                dont_remove_ids = set()

            current_html = clean_html(
                html=get_html_from_action(current_round["action"]),
                prettify=True,
                strip_annotation=False,
                dont_remove_ids=dont_remove_ids,
            ).strip()

        # Get the current plan for this round
        if round_number == 1:
            # First round uses the initial plan
            current_plan = plan_history["initial_plan"]["plan"]
            prompt = (
                DynamicExecutor._FIRST_ROUND_PROMPT_PREFIX.format(
                    task=task,
                )
                + f"""

# Current Plan
Here is the CURRENT plan that you need to follow for this round:

{current_plan}"""
                + DynamicExecutor._FIRST_ROUND_PROMPT_POSTFIX.format(
                    current_html=current_html,
                )
            )
        else:
            # Subsequent rounds use the replanned plans if available
            if round_number - 2 < len(plan_history["replan_steps"]):
                current_plan = plan_history["replan_steps"][round_number - 2]["plan"][
                    "plan"
                ]
            else:
                # If no replan is available for this round, use the last available plan
                if len(plan_history["replan_steps"]) > 0:
                    current_plan = plan_history["replan_steps"][-1]["plan"]["plan"]
                else:
                    current_plan = plan_history["initial_plan"]["plan"]

            prompt = (
                DynamicExecutor._INTERMEDIATE_ROUND_PROMPT_PREFIX.format(
                    round_number=round_number,
                )
                + f"""

# Updated Plan
Here is the UPDATED plan that you need to follow for this round:

{current_plan}"""
                + DynamicExecutor._INTERMEDIATE_ROUND_PROMPT_POSTFIX.format(
                    current_html=current_html,
                )
            )

        # Add the current round's user prompt
        global_messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        if self._verbose:
            for message in global_messages:
                print(f"{message['role']}:\n{message['content']}")

        return global_messages


def create_dynamic_act_training_data(
    model_name: str,
    dynamic_act_data_path: str,
    output_path: str,
    type: Literal["llama_factory", "torchtune"],
    validation_split: float = 0.1,
) -> tuple[
    list[TorchTuneData | LlamaFactoryData], list[TorchTuneData | LlamaFactoryData]
]:
    """
    Creates training data for the dynamic act model using the dynamic act data.
    Similar to create_act_training_data but handles the dynamic plan history.
    """
    data: list[DynamicActAnnotatorDataGeneratorData] = []
    with open(dynamic_act_data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    def create_input_output_pairs_and_save(
        data: list[DynamicActAnnotatorDataGeneratorData],
        output_path: str,
    ) -> list[TorchTuneData | LlamaFactoryData]:
        input_output_pairs: list[
            tuple[DynamicExecutorInferenceInput, ActInferenceOutput]
        ] = []
        for item in data:
            sorted_actions = sorted(
                item["data"]["executor_actions"],
                key=lambda x: x["action"]["index_within_task"],
            )
            for i, action in enumerate(sorted_actions):
                input_output_pairs.append(
                    (
                        DynamicExecutorInferenceInput(
                            task=item["task"],
                            plan_history=item["data"]["plan_history"],
                            previous_rounds=sorted_actions[:i],
                            current_round=action,
                        ),
                        ActInferenceOutput(
                            action_str=get_raw_action_string_from_action(
                                action["action"]
                            ),
                            reasoning=action["reasoning"],
                        ),
                    )
                )

        # Create a dummy LLM and executor object so that we can use the `construct_prompt` function.
        dynamic_executor = DynamicExecutor(llm=LLM(0, 0, model_name))

        torchtune_data = []
        formatting_function = (
            format_training_data_for_torchtune
            if type == "torchtune"
            else format_training_data_for_llama_factory
        )

        for input_output in tqdm(
            input_output_pairs, desc="Creating dynamic act training data"
        ):
            input, output = input_output
            prompt = dynamic_executor.construct_prompt(input)

            # Create the formatted training data. This function will construct the input message with the think token already appended so we don't need it in the output.
            formatted_data = formatting_function(
                messages=prompt,
                tokenizer=dynamic_executor._llm._tokenizer,  # type: ignore
                output=f"{output['reasoning']}\n</think>\n\n[Start of Action]\n{output['action_str']}\n[End of Action]",
            )
            torchtune_data.append(formatted_data)

        with open(output_path, "w") as f:
            json.dump(torchtune_data, f, indent=4, ensure_ascii=False)

        return torchtune_data

    random.seed(42)
    random.shuffle(data)

    num_validation_tasks = int(len(data) * validation_split)
    train_data = data[:-num_validation_tasks]
    print(f"Training data: {len(train_data)}")
    val_data = data[-num_validation_tasks:]
    print(f"Validation data: {len(val_data)}")

    train_torchtune_data = create_input_output_pairs_and_save(
        train_data, output_path.replace(".json", "_train.json")
    )
    val_torchtune_data = create_input_output_pairs_and_save(
        val_data, output_path.replace(".json", "_val.json")
    )

    return train_torchtune_data, val_torchtune_data
