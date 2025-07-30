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
    CotActAnnotatorDataGeneratorData,
    CotPlanAnnotatorDataGeneratorData,
    ExecutorAction,
    ExecutorTrainingData,
    ProcessedData,
    ServerMessage,
    WebArenaLiteAction,
)
from plan_and_act.cot.utils import (
    clean_html,
    get_action_information,
    get_action_information_from_action_str,
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


class CoTActAnnotator:

    _MAX_RETRY = 3
    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool) -> None:
        self._llm = llm
        self._verbose = verbose

    async def annotate(
        self,
        plan_data: CotPlanAnnotatorDataGeneratorData,
        processed_data: ProcessedData,
    ) -> CotActAnnotatorDataGeneratorData:
        """
        This annotator is a simple CoT annotator for the actual WebArena Lite actions for our executor model.

        This executor model takes in the plan and the current state of the task (which includes the previous actions, the HTML state and the task query) and then generates a reasoning for the next immediate action as well as the action itself as a function call. Since each action should see the reasoning of the next action, this function cannot be parallelized.

        For this, this `annotate` function should take:
            - plan: The plan object that was generated from the plan annotator in `plan.py`.
            - processed_data: The processed data object which includes the all the actions belonging to this task as well as the task itself.

        This function then returns a list of `ExecutorData` objects which contain the reasoning and the action for each turn.
        """

        actions = sorted(
            processed_data["actions"], key=lambda x: x["index_within_task"]
        )

        if len(actions) == 0:
            raise ValueError("No actions found in the processed data.")

        # Initialize the output list.
        output_executor_actions: list[ExecutorAction] = []

        # Initialize the global messages list.
        global_messages: list[ServerMessage] = []

        if self._verbose:
            print(
                f"=========\n\n### Plan:\n\n{plan_data['data'][2]['plan']['plan']}\n\n========="
            )

        # Iterate over all the actions.
        for i, webarena_lite_action in enumerate(actions):
            # If this is the first action, then there will be no previous plans. This is the round where we put the plan and the system instructions into the context.
            is_action_exit = (
                get_action_information(webarena_lite_action)["action"]["action_type"]
                == ActionTypes.STOP
            )

            print(
                f">>> Action {i}: {get_raw_action_string_from_action(webarena_lite_action)}"
            )

            if i == 0 and not is_action_exit:
                prompts = self._prepare_first_action_prompt(
                    plan_data,
                    processed_data,
                    webarena_lite_action,
                )
            # If however this is the last action, then we will need to make a special kind of prompt which asks the model why the task is done and what the answer is. This is similar to an intermediate action prompt but with a different action goal of exit() function.
            elif i == len(actions) - 1:
                prompts = self._prepare_last_action_prompt(webarena_lite_action)
            # Else, this guy will have the previous actions. The plan will already be in the context, so we will just give the current HTML and ask for the next immediate action.
            else:
                prompts = self._prepare_intermediate_action_prompt(
                    plan_data,
                    webarena_lite_action,
                )

            print(f"Current html: {get_html_from_action(webarena_lite_action)}")

            # Now call the model to get the reasoning and the action and then put the entire think into the context again.
            reasoning = None
            action = None

            # Keep track of an internal messages list in case there were any errors.
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

                    # Check if the action is there (enclosed in "[Start of Action]" and "[End of Action]" tags)
                    if (
                        "[Start of Action]" in response
                        and "[End of Action]" in response
                    ):
                        # Check if the output action is the ground truth action
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

            # Then append the new rounds messages to the global messages list
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

        return CotActAnnotatorDataGeneratorData(
            task=processed_data["task"],
            data=ExecutorTrainingData(
                task=processed_data["task"],
                initial_html_state=processed_data["initial_html_state"],
                plan=plan_data["data"][2]["plan"],
                executor_actions=output_executor_actions,
            ),
        )

    def _same_action_str(self, action_1: str, action_2: str) -> bool:
        """
        This function checks if two action strings are the same.
        """
        action_1_info = get_action_information_from_action_str(action_1)
        action_2_info = get_action_information_from_action_str(action_2)
        for key in action_1_info["action"].keys():
            if key in ("action_type", "element_id", "text", "direction"):
                if action_1_info["action"][key] != action_2_info["action"][key]:
                    return False
        return True

    def _prepare_first_action_prompt(
        self,
        plan_data: CotPlanAnnotatorDataGeneratorData,
        processed_data: ProcessedData,
        webarena_lite_action: WebArenaLiteAction,
    ) -> ExecutorAnnotationPrompt:
        """
        This first action prompt is the first user message which will include:
            - System instructions explaining the training data generation task
            - The user task and plan
            - The current HTML state
            - The ground truth action and resulting HTML state (for training data generation)
        """

        prompt_template = """# Goal
You are an annotator LLM helping to generate training data for an executor agent model. The executor agent is a powerful model that can complete complex web navigation tasks by issuing web actions such as clicking, typing, selecting, and more. The agent takes the current HTML state, user's task, and a plan, then decides what immediate action to take.

For this training data generation task, for each round of execution, you will be given:
1. The user's task and execution plan
2. The previous rounds of actions that were taken (as previous conversations)
3. The current HTML state
4. The ground truth action that the executor agent needs to take at this round
5. The description of the ground truth action

Your job is to generate expert-level reasoning that the executor agent should use when deciding its next action (which is the ground truth action). The executor agent itself will only have access to the task, plan, and current HTML state - it won't know the ground truth action or future HTML state. Your reasoning should demonstrate the analytical process of discovering and choosing this action through careful examination of the page and understanding of the task.

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

{task}

# Plan
Here is the plan the agent should follow:

{plan}

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
- Explain why this action aligns with the current step in the plan
- Act as a "world model" by anticipating how the webpage will respond to the action
- Pretend as if you are the executor agent itself. Hence, you MUST always talk in "I" or "we" instead. Avoid mentioning the "executor agent" or "agent" in your reasoning.

After your thinking, output the action enclosed in '[Start of Action]' and '[End of Action]' tags."""

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

        return ExecutorAnnotationPrompt(
            current_prompt=prompt_template.format(
                task=processed_data["task"],
                plan=plan_data["data"][2]["plan"]["plan"],
                current_html=current_html,
                action=action_str,
                action_description=plan_data["data"][0][0]["action_description"],
            ),
            global_prompt=prompt_template.format(
                task=processed_data["task"],
                plan=plan_data["data"][2]["plan"]["plan"],
                current_html=current_html_snippet,
                action=action_str,
                action_description=plan_data["data"][0][0]["action_description"],
            ),
        )

    def _prepare_intermediate_action_prompt(
        self,
        plan_data: CotPlanAnnotatorDataGeneratorData,
        webarena_lite_action: WebArenaLiteAction,
    ) -> ExecutorAnnotationPrompt:
        """
        This intermediate action prompt is used when there are previous actions that the executor has already taken
        and this is not the last action. The prompt will include:
            - The current round number
            - The current HTML state
            - The ground truth action that was taken
            - The HTML state after taking that action
        """

        prompt = """Let's continue with the next round of execution. You will be given the current HTML of the webpage, the ground truth action that was taken, and the description of the action.

## Round {round_number}

Current HTML state:

{current_html}

Action to be taken by the executor agent in this round (you also need to generate this exact action - including the '# Note:' comment - in your response) for which you are going to generate the reasoning:

{action}

Description of the action:

{action_description}

Based on the above, generate expert reasoning that shows how the executor agent would analyze the current situation and arrive at this action. The executor agent itself won't know the ground truth action or the description of the action - your reasoning should demonstrate the analytical process of discovering and choosing this action through careful examination of the page and understanding of the task. Hence, you should:

- Reflect on the previous rounds of actions and the changes they made to the HTML state
- Demonstrate expert-level understanding of web navigation and DOM manipulation
- Show how to analyze the current HTML state to confidently identify the correct element to interact with by including any relevant element attributes, text content, or positioning that will help locate the target
- Predict what changes the action will cause to the webpage state (how does the action change the current HTML state to the next HTML state)
- Explain why this action aligns with the current step in the plan
- Act as a "world model" by anticipating how the webpage will respond to the action
- Pretend as if you are the executor agent itself. Hence, you MUST always talk in "I" or "we" instead. Avoid mentioning the "executor agent" or "agent" in your reasoning.

After your thinking, output the action enclosed in '[Start of Action]' and '[End of Action]' tags."""

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

        return ExecutorAnnotationPrompt(
            current_prompt=prompt.format(
                round_number=round_number,
                current_html=current_html,
                action=action_str,
                action_description=plan_data["data"][0][round_number - 1][
                    "action_description"
                ],
            ),
            global_prompt=prompt.format(
                round_number=round_number,
                current_html=current_html_snippet,
                action=action_str,
                action_description=plan_data["data"][0][round_number - 1][
                    "action_description"
                ],
            ),
        )

    def _prepare_last_action_prompt(
        self, webarena_lite_action: WebArenaLiteAction
    ) -> ExecutorAnnotationPrompt:
        """
        This last action prompt is the prompt we use for the last `exit()` action. It will ask the annotator to provide the reasoning for why the task is done, why it doesn't need to take another action, and what its last message/answer is.
        """

        prompt = """Let's continue with the next round of execution. This round is the last round of the task. Now, you need to provide the reasoning for why you think the task is done, why you don't need to take another action, and what your final answer/message is. If your task requires you to make an analysis, then you should provide that analysis as well.

## Round {round_number}

Current HTML state:

{html}

Last exit action that was taken:

{action}

Based on the above, generate expert reasoning that shows how the executor agent should think about choosing the `exit` action by pretending to be the executor agent itself. Remember, the agent itself won't know the ground truth action and the ground truth exit message to return - your reasoning should show how it can expertly analyze the current state and provide excellent answers. You MUST be comprehensive and thorough in your reasoning. You are allowed to think as much as you need to before outputting the next action.

After your thinking, output the action enclosed in '[Start of Action]' and '[End of Action]' tags."""

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
            current_prompt=prompt.format(
                round_number=round_number, html=current_prompt_html, action=action_str
            ),
            global_prompt=prompt.format(
                round_number=round_number, html=global_prompt_html, action=action_str
            ),
        )


class CoTActAnnotatorDataGeneratorData(TypedDict):
    task: str
    data: ExecutorTrainingData


class CoTActAnnotatorDataGenerator:
    _cot_annotator: CoTActAnnotator

    # Input data
    _annotation_input_data: list[
        tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]
    ]
    _already_processed: set[str]

    # Output paths
    _output_path: str

    # Results aggreagator
    _results: list[CotActAnnotatorDataGeneratorData]

    def __init__(
        self,
        cot_annotator: CoTActAnnotator,
        annotation_input_data: list[
            tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]
        ],
        output_path: str,
    ) -> None:
        self._cot_annotator = cot_annotator
        self._annotation_input_data = annotation_input_data
        self._already_processed = (
            CoTActAnnotatorDataGenerator._get_already_processed_tasks(output_path)
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
            tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData],
            CotActAnnotatorDataGeneratorData,
            str,
        ](
            data_to_process=unprocessed_data,
            task_fn=self._task_fn,
            save_fn=self.save_results,
            concurrency=16,
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
        self, data: tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]
    ) -> CotActAnnotatorDataGeneratorData:
        return await self._cot_annotator.annotate(data[0], data[1])

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
    parser.add_argument("--plan_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--start_index", type=int, required=False, default=0)
    parser.add_argument("--end_index", type=int, required=False, default=None)
    parser.add_argument("--verbose", type=bool, required=False, default=False)
    parser.add_argument("--base_url", type=str, required=False, default=None)
    return parser.parse_args()


def load_plan_data_and_pair_with_processed_data(
    plan_data_path: str, input_data_path: str
) -> list[tuple[CotPlanAnnotatorDataGeneratorData, ProcessedData]]:
    """
    Loads both the plan data and the processed data and then pairs the ones that are for the same task together into tuples so that we can use them together during CoTActAnnotatorDataGenerator.
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
    python3 plan_and_act/cot/data_generation/act.py --input_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/raw_all_2036.json" \
        --plan_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/plan_annotation_data_all_2036_DeepSeek-R1-Distill-Llama-70B.jsonl" \
        --output_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/act_annotation_data_all_2036.jsonl" \
        --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
        --start_index 0 \
        --end_index 688 \
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

    cot_annotator = CoTActAnnotator(llm=llm, verbose=args.verbose)
    paired_data = load_plan_data_and_pair_with_processed_data(
        args.plan_data_path, args.input_data_path
    )
    annotation_input_data = paired_data[args.start_index : args.end_index]

    data_generator = CoTActAnnotatorDataGenerator(
        cot_annotator=cot_annotator,
        annotation_input_data=annotation_input_data,
        output_path=args.output_path,
    )

    asyncio.run(data_generator.run())
