import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TypedDict

from tqdm import tqdm

from plan_and_act.cot.models import (
    LLM,
    ActInferenceInput,
    ActInferenceOutput,
    ActInferencePreviousRound,
    CotActAnnotatorDataGeneratorData,
    ExecutorAction,
    Plan,
    ServerMessage,
    TorchTuneData,
)
from plan_and_act.cot.utils import (
    clean_html,
    format_training_data_for_torchtune,
    get_action_information,
    get_action_information_from_action_str,
    get_html_from_action,
    get_previous_round_html_snippet,
    get_previous_round_html_snippet_from_html,
    get_raw_action_string_from_action,
    prepare_completions_prompt_for_reasoning_models,
)


class CoTExecutor:
    """
    This executor is responsible for executing web tasks based on a plan.
    Unlike the CoTActAnnotator which generates training data by providing reasoning for ground truth actions, this executor needs to generate the actual actions based on the current state, plan, and previous actions.
    """

    _FIRST_ROUND_PROMPT = """# Goal
You are the Executor Agent, a powerful assistant that can complete complex web navigation tasks by issuing web actions such as clicking, typing, selecting, and more. You will be provided with:
1. The user's task and execution plan
2. The previous rounds of actions that were taken (if any)
3. The current HTML state of the webpage

Your goal is to analyze the current state and determine the next immediate action to take that will progress toward completing the task. You should:

- Carefully analyze the current HTML state to identify the correct elements to interact with
- Consider how your action will change the webpage state
- Ensure your action aligns with the current step in the plan
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
Here is the task that the user wants to complete: {task}

# Plan
Here is the plan that you need to follow:

{plan}

---

Let's start with the first round of action.

## Round 1

Current HTML state:

{current_html}

Based on the task, plan, and current HTML state, what should be the next immediate action? First, write down your detailed reasoning and then output your action enclosed between the tags "[Start of Action]" and "[End of Action]".
"""

    _INTERMEDIATE_ROUND_PROMPT = """Let's continue with the next round of execution. 

## Round {round_number}

Current HTML state:

{current_html}

Based on the task, plan, and current HTML state, what should be the next immediate action? First, write down your detailed reasoning and then output your action enclosed between the tags "[Start of Action]" and "[End of Action]"."""

    _MAX_RETRY = 3
    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool = False) -> None:
        self._llm = llm
        self._verbose = verbose

    async def act(
        self,
        input: ActInferenceInput,
    ) -> ActInferenceOutput:
        """
        This function runs the Executor model during evaluation.
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
        input: ActInferenceInput,
    ) -> list[ServerMessage]:
        """
        Constructs the prompt for the executor to generate the next action.
        This is similar to the annotator's prompt but without ground truth actions.
        """
        task = input["task"]
        plan = input["plan"]
        previous_rounds = input["previous_rounds"]
        current_round = input["current_round"]

        global_messages: list[ServerMessage] = []

        # Iterate over all past actions to construct the conversation history first. These guys will have HTMLs that are represented only as snippets while the last action will have the full HTML. The assistant message will be constructed to have thinking tokens as well.
        for i, executor_action in enumerate(previous_rounds):
            round_number = i + 1

            if "act" in executor_action and "uncleaned_html" in executor_action:
                reasoning = executor_action["act"]["reasoning"]
                action_str = executor_action["act"]["action_str"]
                current_html = get_previous_round_html_snippet_from_html(
                    html=executor_action["uncleaned_html"],
                    target_element_id=get_action_information_from_action_str(
                        action_str
                    )["action"]["element_id"],
                    strip_annotation=False,
                )
            else:
                assert "action" in executor_action
                reasoning = executor_action["reasoning"]
                action_str = get_raw_action_string_from_action(
                    executor_action["action"]
                )
                current_html = get_previous_round_html_snippet(
                    executor_action["action"], strip_annotation=False
                )

            if i == 0:
                prompt = CoTExecutor._FIRST_ROUND_PROMPT.format(
                    task=task,
                    plan=plan,
                    current_html=current_html,
                )
            else:
                prompt = CoTExecutor._INTERMEDIATE_ROUND_PROMPT.format(
                    round_number=round_number,
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
        if "act" in current_round and "uncleaned_html" in current_round:
            current_html = clean_html(
                html=current_round["uncleaned_html"],
                prettify=True,
                strip_annotation=False,
            )
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
            )

        if round_number == 1:
            prompt = CoTExecutor._FIRST_ROUND_PROMPT.format(
                task=task,
                plan=plan,
                current_html=current_html,
            )
        else:
            prompt = CoTExecutor._INTERMEDIATE_ROUND_PROMPT.format(
                round_number=round_number,
                current_html=current_html,
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


def create_act_training_data(
    model_name: str,
    act_data_path: str,
    output_path: str,
) -> None:
    data: list[CotActAnnotatorDataGeneratorData] = []
    with open(act_data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    input_output_pairs: list[tuple[ActInferenceInput, ActInferenceOutput]] = []
    for item in data:
        sorted_actions = sorted(
            item["data"]["executor_actions"],
            key=lambda x: x["action"]["index_within_task"],
        )
        for i, action in enumerate(sorted_actions):
            input_output_pairs.append(
                (
                    ActInferenceInput(
                        task=item["task"],
                        plan=item["data"]["plan"],
                        previous_rounds=sorted_actions[:i],
                        current_round=action,
                    ),
                    ActInferenceOutput(
                        action_str=get_raw_action_string_from_action(action["action"]),
                        reasoning=action["reasoning"],
                    ),
                )
            )

    # Create a dummy LLM and executor object so that we can use the `construct_prompt` function.
    cot_executor = CoTExecutor(llm=LLM(0, 0, model_name))

    torchtune_data: list[TorchTuneData] = []
    for input_output in tqdm(input_output_pairs, desc="Creating act training data"):
        input, output = input_output
        prompt = cot_executor.construct_prompt(input)

        # Create the formatted training data. This function will construct the input message with the think token already appended so we don't need it in the output.
        formatted_data = format_training_data_for_torchtune(
            messages=prompt,
            tokenizer=cot_executor._llm._tokenizer,  # type: ignore
            output=f"{output['reasoning']}\n</think>\n\n[Start of Action]\n{output['action_str']}\n[End of Action]",
        )
        torchtune_data.append(formatted_data)

    with open(output_path, "w") as f:
        json.dump(torchtune_data, f, indent=4, ensure_ascii=False)
