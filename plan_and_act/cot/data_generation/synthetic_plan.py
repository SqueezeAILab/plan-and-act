import argparse
import asyncio
import functools
import json
import os
import random
from collections import defaultdict
from typing import TypedDict

import aiometer

from plan_and_act.cot.data_generation.plan import CoTPlanAnnotator
from plan_and_act.cot.models import (
    LLM,
    AsyncDataGenerationJobEngine,
    CotPlanAnnotatorDataGeneratorData,
    CotSyntheticPlanDataGeneratorData,
    Plan,
    PlanTrainingData,
    ProcessedData,
    ServerMessage,
    WebArenaLiteWebsite,
)
from plan_and_act.cot.utils import (
    classify_website,
    format_action_string,
    get_action_information,
    get_html_snippet_str,
    prepare_completions_prompt_for_reasoning_models,
    preprocess_webarena_data,
)


class TaskGenerationResult(TypedDict):
    task: str
    initial_html_example_index: int
    initial_html_state: str


class CoTActInContextExampleRepository:
    """
    This repository stores in-context examples for data generation.
    It classifies each example by website and allows for round-robin sampling,
    BUT each call to get_round_robin_in_context_examples(n) will sample *n*
    examples from exactly one website at a time, then advances to the next website.
    """

    _in_context_examples_by_website: dict[
        WebArenaLiteWebsite, list[CotPlanAnnotatorDataGeneratorData]
    ]
    _remaining_examples_by_website: dict[
        WebArenaLiteWebsite, list[CotPlanAnnotatorDataGeneratorData]
    ]
    _round_robin_website_index: int

    def __init__(
        self,
        path: str,
        processed_data: list[ProcessedData],
        already_processed: set[str],
    ) -> None:
        """
        Load the in-context examples from the given path and classify each example by website.
        """
        # Get the initial html state of each task from the processed data.
        task_to_initial_html_state_dict = {
            processed_data["task"]: processed_data["initial_html_state"]
            for processed_data in processed_data
        }

        data: list[CotPlanAnnotatorDataGeneratorData] = []
        with open(path, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))

        print(f"Loaded {len(data)} in-context examples.")

        # Initialize a dictionary to store examples by website
        self._in_context_examples_by_website = defaultdict(list)

        # Classify each data entry and place it in the corresponding website list
        for item in data:
            website = classify_website(task_to_initial_html_state_dict[item["task"]])
            if website is not None:
                self._in_context_examples_by_website[website].append(item)
            else:
                # If desired, you can handle "unknown" items here
                pass

        # Create a "remaining" dict that will be shuffled independently of the master copy
        already_processed_set = set(already_processed)
        self._remaining_examples_by_website = {}
        for website, examples in self._in_context_examples_by_website.items():
            # Only for the very first time, remove the already processed examples since they were already used as in-context examples.
            if len(already_processed_set) > 0:
                examples = [
                    example
                    for example in examples
                    if example["task"] not in already_processed_set
                ]

            self._remaining_examples_by_website[website] = examples[:]
            random.shuffle(self._remaining_examples_by_website[website])

        # Start round-robin at index 0
        self._round_robin_website_index = 0

    def _reshuffle_examples_by_website(self, website: WebArenaLiteWebsite) -> None:
        """
        Reshuffle the examples for a given website, resetting its 'remaining' pool.
        """
        self._remaining_examples_by_website[website] = (
            self._in_context_examples_by_website[website][:]
        )
        random.shuffle(self._remaining_examples_by_website[website])

    def get_round_robin_in_context_examples(
        self, n: int
    ) -> list[CotPlanAnnotatorDataGeneratorData]:
        """
        Return exactly n in-context examples from *one* website, advancing the round-robin index.
        """
        # If no websites are found, return empty
        websites = list(self._in_context_examples_by_website.keys())
        if not websites:
            return []

        # Select the current website by index
        current_website = websites[self._round_robin_website_index]

        # Prepare for the next call by moving index forward
        self._round_robin_website_index = (self._round_robin_website_index + 1) % len(
            websites
        )

        # Ensure enough examples remain for this website; if not, reshuffle
        if len(self._remaining_examples_by_website[current_website]) < n:
            self._reshuffle_examples_by_website(current_website)

        # Collect n examples from this website
        results = []
        for _ in range(n):
            # If in the extremely rare case we still don't have enough,
            # do a safety check or handle as needed:
            if not self._remaining_examples_by_website[current_website]:
                self._reshuffle_examples_by_website(current_website)
            results.append(self._remaining_examples_by_website[current_website].pop())

        return results


class CoTSyntheticPlanDataGenerator:
    """
    This class generates synthetic plan data as well as the CoT reasoning for them. At each iteration, we randomly sample a website and from that website, we sample random in-context examples. These in-context examples include:
        - Action trajectory
        - Reasoning for the plan
        - Final plan

    Then the synthetic plan data generator will generate new plans and reasonings for them by trying to increase diversity in the generated data while also keeping the generated daata grounded on the in-context examples since we don't want to generate hallucinated plans that don't actually map to the target environment.

    Before using this class, make sure that the in-context examples are real trajectories and that you have already run `CotActAnnotatorDataGenerator` to generate the in-context examples first.
    """

    _MAX_RETRY = 3
    _in_context_example_repository: CoTActInContextExampleRepository
    _how_many_to_generate_at_once: int
    _n_context_examples: int
    _llm: LLM
    _verbose: bool

    def __init__(
        self,
        llm: LLM,
        how_many_to_generate_at_once: int,
        in_context_example_path: str,
        processed_data: list[ProcessedData],
        already_processed: set[str],
        n_context_examples: int = 3,
        verbose: bool = False,
    ) -> None:
        self._llm = llm
        self._in_context_example_repository = CoTActInContextExampleRepository(
            path=in_context_example_path,
            processed_data=processed_data,
            already_processed=already_processed,
        )
        self._how_many_to_generate_at_once = how_many_to_generate_at_once
        self._n_context_examples = n_context_examples
        self._verbose = verbose

    async def generate(self) -> CotSyntheticPlanDataGeneratorData:
        """
        Generate synthetic tasks and their corresponding plans using in-context examples as grounding.
        """
        # 1) Get random in-context examples and format them
        in_context_examples = (
            self._in_context_example_repository.get_round_robin_in_context_examples(
                self._n_context_examples
            )
        )

        # 2) Generate tasks and select initial HTML states
        tasks = await self._generate_tasks(in_context_examples)

        if self._verbose:
            print(f">>> Got {len(tasks)} tasks")

        # 3) Generate reasoning and plans in parallel for each task
        plan_tasks = []
        for task in tasks:
            plan_tasks.append(
                functools.partial(
                    self._generate_plan_for_task,
                    task,
                    in_context_examples,
                )
            )

        plans = await aiometer.run_all(
            plan_tasks,
            max_at_once=10,
            max_per_second=10,
        )

        if self._verbose:
            print(f">>> Got {len(plans)} plans")

        return CotSyntheticPlanDataGeneratorData(
            in_context_examples=[example["task"] for example in in_context_examples],
            datas=[plan for plan in plans if plan is not None],
        )

    async def _generate_tasks(
        self, in_context_examples: list[CotPlanAnnotatorDataGeneratorData]
    ) -> list[TaskGenerationResult]:
        """Generate diverse tasks and select appropriate initial HTML states."""

        formatted_examples = self._format_examples_for_task_generation(
            in_context_examples
        )

        prompt = f"""# Task Generation
You are a synthetic task generator creating novel but grounded tasks for web-based interactions. Your goal is to generate {self._how_many_to_generate_at_once} diverse tasks that:

1. Explore new use cases on the website while remaining feasible
2. Are grounded in the website's actual capabilities (based on the examples)
3. Start from an appropriate initial HTML state from the provided examples

# In-context Examples
Here are {self._n_context_examples} examples from the website to help you understand its capabilities:

{formatted_examples}

# Output Format
For each task, provide:
[Start Task N]
Task: ... task user query ...
Initial HTML State from Example Index: M # Specify which example's HTML state to use (M = 1 to {self._n_context_examples}, don't put 'M' in the output - just an integer)
[End Task N]

# Important Guidelines
- Tasks should be diverse and explore different aspects of the website
- Each task should be clearly achievable given the website's capabilities
- Choose initial HTML states that make sense for each task
- Tasks should be specific and well-defined
- Try to use different initial HTML states across tasks to increase diversity

Generate {self._how_many_to_generate_at_once} tasks now:"""

        messages = [ServerMessage(role="user", content=prompt)]

        response = None
        tasks = []

        for retry_count in range(self._MAX_RETRY):
            try:
                response = await self._llm.acall_completions(
                    prepare_completions_prompt_for_reasoning_models(
                        messages, self._llm._tokenizer  # type: ignore
                    )
                )

                if response is None:
                    raise ValueError("Failed to generate tasks")

                if self._verbose:
                    print(f">>> Task Stage Response (Attempt {retry_count + 1}):")
                    print(response)

                # Check if we have the expected format for all tasks
                all_tasks_found = True
                tasks = []
                errors = []

                for i in range(self._how_many_to_generate_at_once):
                    try:
                        if (
                            f"[Start Task {i+1}]" not in response
                            or f"[End Task {i+1}]" not in response
                        ):
                            all_tasks_found = False
                            errors.append(f"Missing task {i+1} markers in the response")
                            continue

                        task = (
                            response.split(f"[Start Task {i+1}]")[1]
                            .split(f"[End Task {i+1}]")[0]
                            .strip()
                        )

                        if "Task:" not in task:
                            all_tasks_found = False
                            errors.append(f"Missing 'Task:' in task {i+1}")
                            continue

                        if "Initial HTML State from Example Index:" not in task:
                            all_tasks_found = False
                            errors.append(
                                f"Missing 'Initial HTML State from Example Index:' in task {i+1}"
                            )
                            continue

                        task_description = task.split("Task:")[1].strip()
                        if not task_description:
                            all_tasks_found = False
                            errors.append(f"Empty task description in task {i+1}")
                            continue

                        initial_html_example_index_str = task.split(
                            "Initial HTML State from Example Index:"
                        )[1].strip()
                        try:
                            initial_html_example_index = int(
                                initial_html_example_index_str
                            )
                            if (
                                initial_html_example_index < 1
                                or initial_html_example_index > self._n_context_examples
                            ):
                                all_tasks_found = False
                                errors.append(
                                    f"Invalid example index in task {i+1}: {initial_html_example_index}"
                                )
                                continue
                        except ValueError:
                            all_tasks_found = False
                            errors.append(
                                f"Could not parse example index in task {i+1}: '{initial_html_example_index_str}'"
                            )
                            continue

                        initial_html_state = in_context_examples[
                            initial_html_example_index - 1
                        ]["data"][2]["initial_html_state"]

                        tasks.append(
                            TaskGenerationResult(
                                task=task_description,
                                initial_html_example_index=initial_html_example_index,
                                initial_html_state=initial_html_state,
                            )
                        )
                    except Exception as e:
                        all_tasks_found = False
                        print(f"Failed to parse task {i+1}: {e}")
                        errors.append(f"Failed to parse task {i+1}: {e}")

                # If all tasks were found and parsed successfully, break out of the retry loop
                if all_tasks_found:
                    break

                # If we're here, we need to retry
                error_message = "\n".join(errors)
                messages.extend(
                    [
                        {"role": "assistant", "content": response},
                        {
                            "role": "user",
                            "content": f"There were issues with your response:\n{error_message}\n\nPlease ensure you follow the output format exactly:\n\n[Start Task N]\nTask: ... task user query ...\nInitial HTML State from Example Index: M\n[End Task N]\n\nGenerate exactly {self._how_many_to_generate_at_once} tasks with this format.",
                        },
                    ]
                )

            except Exception as e:
                print(f"Error during task generation (attempt {retry_count + 1}): {e}")
                if retry_count == self._MAX_RETRY - 1:
                    raise ValueError(
                        f"Failed to generate tasks after {self._MAX_RETRY} attempts: {e}"
                    )

                # Add error message to conversation for retry
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}. Please try again and ensure you follow the output format exactly.",
                    }
                )

        if not tasks:
            raise ValueError(
                f"Failed to generate any valid tasks after {self._MAX_RETRY} attempts"
            )

        return tasks

    async def _generate_plan_for_task(
        self,
        task: TaskGenerationResult,
        in_context_examples: list[CotPlanAnnotatorDataGeneratorData],
    ) -> PlanTrainingData | None:
        """Generate detailed reasoning and plan for a single task."""

        formatted_examples = self._format_examples_for_plan_generation(
            in_context_examples,
            task["initial_html_example_index"] - 1,  # Convert to 0-based index
        )

        prompt = f"""# Goal
You are an expert web navigation planner with deep understanding of web interfaces and user interactions. Your task is to create a detailed execution plan for accomplishing a specific web-based task. You have access to:

1. The user's task query
2. The initial HTML state of the webpage
3. A relevant example of a similar task (for reference)

As an expert planner, you should:
- Demonstrate expert-level understanding of web navigation and DOM manipulation
- Think like a "world model" by predicting how the webpage will respond to different actions
- Handle dynamic content carefully without assuming specific values
- Create detailed, actionable plans that can adapt to varying webpage states

# Task Query
{task["task"]}

# Initial HTML State
{task["initial_html_state"]}

# Reference Example
Here is a relevant example to help ground your planning:

{formatted_examples}

**Important Note:** The example is only for context so that you understand the website's capabilities. You should not follow the example exactly.

Based on the above, analyze the task and generate a comprehensive execution plan. You should:

1. Think through the task requirements carefully
2. Consider the current webpage state and available interactions
3. Predict how the webpage will respond to different actions
4. Break down complex goals into clear, actionable steps
5. Plan for handling dynamic content and unknown values
6. Ensure each step has enough context to identify correct elements

Remember:
- Think as if you are actively planning and reasoning through the task
- Use "I" or "we" language to express your thought process
- Don't mention the "reference example" in your plan since during the execution, you will not have access to it
- Be thorough in analyzing potential challenges and solutions
- Don't assume specific values for dynamic content
- Explain how to handle varying webpage states

After your detailed thinking process, output the final plan enclosed in '[Start of Plan]' and '[End of Plan]' tags.

Begin your analysis now:"""

        messages = [ServerMessage(role="user", content=prompt)]

        reasoning = None
        plan = None

        for retry_count in range(self._MAX_RETRY):
            try:
                response = await self._llm.acall_completions(
                    prepare_completions_prompt_for_reasoning_models(
                        messages, self._llm._tokenizer  # type: ignore
                    )
                )

                if response is None:
                    raise ValueError("Failed to generate plan")

                if self._verbose:
                    print(
                        f">>> Plan Stage Response for Task '{task['task']}' (Attempt {retry_count + 1}):"
                    )
                    print(response)

                # Check if the response contains the plan tags
                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    # The reasoning is everything before the plan tags
                    reasoning = response.split("</think>")[0].strip()
                    plan = (
                        response.split("[Start of Plan]")[1]
                        .split("[End of Plan]")[0]
                        .strip()
                    )

                    # Validate that we have both reasoning and plan
                    if reasoning and plan:
                        break
                    else:
                        # If either is empty, we need to retry
                        if not reasoning:
                            raise ValueError("Reasoning section is empty")
                        if not plan:
                            raise ValueError("Plan section is empty")
                else:
                    # If the response doesn't contain the plan tags, we need to retry
                    messages.extend(
                        [
                            {"role": "assistant", "content": response},
                            {
                                "role": "user",
                                "content": "You need to ensure that you enclose the final plan between the '[Start of Plan]' and '[End of Plan]' tags. Please provide your reasoning followed by the plan with these tags.",
                            },
                        ]
                    )
                    continue

            except Exception as e:
                print(f"Error during plan generation (attempt {retry_count + 1}): {e}")
                if retry_count == self._MAX_RETRY - 1:
                    print(
                        f"Failed to generate plan after {self._MAX_RETRY} attempts: {e}"
                    )
                    return None

                # Add error message to conversation for retry
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}. Please try again and ensure you follow the output format exactly. Your reasoning should be followed by the plan enclosed in '[Start of Plan]' and '[End of Plan]' tags.",
                    }
                )

        if reasoning is None or plan is None:
            print(f"Failed to generate valid plan after {self._MAX_RETRY} attempts")
            return None

        return PlanTrainingData(
            task=task["task"],
            initial_html_state=task["initial_html_state"],
            plan=Plan(reasoning=reasoning, plan=plan),
        )

    def _format_examples_for_task_generation(
        self, examples: list[CotPlanAnnotatorDataGeneratorData]
    ) -> str:
        """Format examples for the task generation prompt."""
        formatted = []
        for i, example in enumerate(examples):
            formatted.append(f"## Example {i+1}:")
            formatted.append(f"Task: {example['task']}")
            formatted.append(f"\nInitial HTML State:")
            formatted.append(example["data"][2]["initial_html_state"])
            formatted.append("\nPlan Overview:")
            formatted.append(example["data"][2]["plan"]["plan"])
        return "\n".join(formatted)

    def _format_examples_for_plan_generation(
        self, examples: list[CotPlanAnnotatorDataGeneratorData], focus_example_idx: int
    ) -> str:
        """Format the most relevant example for plan generation."""
        example = examples[focus_example_idx]
        formatted = [
            f"## Reference Example",
            f"Task: {example['task']}",
            f"\nReasoning:",
            example["data"][2]["plan"]["reasoning"],
            f"\nPlan:",
            example["data"][2]["plan"]["plan"],
        ]
        return "\n".join(formatted)

    @staticmethod
    def _format_in_context_example(
        in_context_example: CotPlanAnnotatorDataGeneratorData,
    ) -> str:
        """
        For each in-context example, we need to list out the action trajectory (similar to the one in `plan.py`), the reasoning for the plan, and the final plan itself. Then we will combine these by a separator like `Example {i}:` where `i` is the index of the in-context example.
        """
        action_step_descriptions = in_context_example["data"][0]
        action_steps_formatted = []
        for i, action_step in enumerate(action_step_descriptions):
            action_str = [f"### Action {i+1}"]

            # If this action has an element_id, then can include a contextual html snippet.
            try:
                element_id = get_action_information(action_step["action"])["action"][
                    "element_id"
                ]
                if len(element_id) < 0:
                    raise ValueError("Element id is not present in the action.")

                html_snippet = get_html_snippet_str(
                    action_step["action"],
                    num_siblings=3,
                )
                action_str.append(
                    f"HTML Snippet before taking the action on the target element:\n{html_snippet}"
                )
            except Exception:
                pass

            # Add the formatted action string.
            action_str.append("\nUser Event:")
            action_str.append(
                format_action_string(action_step["action"], use_simple_html=True)
            )

            # Add the action description.
            action_str.append("\nDescription:")
            action_str.append(action_step["action_description"])

            action_steps_formatted.append("\n".join(action_str))

        return "\n\n".join(action_steps_formatted)


class CotSyntheticPlanDataGeneratorEngine:
    _cot_synthetic_plan_data_generator: CoTSyntheticPlanDataGenerator
    _max_number_of_plans: int
    _how_many_to_generate_at_once: int

    # Input data
    _already_processed: set[str]

    # Output patgs
    _output_path: str

    # Results aggregator
    _results: list[PlanTrainingData]

    def __init__(
        self,
        llm: LLM,
        how_many_to_generate_at_once: int,
        in_context_example_path: str,
        n_context_examples: int,
        max_number_of_plans: int,
        processed_data: list[ProcessedData],
        output_path: str,
        verbose: bool = False,
    ) -> None:
        self._already_processed = (
            CotSyntheticPlanDataGeneratorEngine._get_already_processed_tasks(
                output_path
            )
        )
        self._cot_synthetic_plan_data_generator = CoTSyntheticPlanDataGenerator(
            llm=llm,
            how_many_to_generate_at_once=how_many_to_generate_at_once,
            in_context_example_path=in_context_example_path,
            processed_data=processed_data,
            already_processed=self._already_processed,
            n_context_examples=n_context_examples,
            verbose=verbose,
        )
        self._results = []
        self._output_path = output_path
        self._max_number_of_plans = max_number_of_plans
        self._how_many_to_generate_at_once = how_many_to_generate_at_once
        self._verbose = verbose

    async def run(self) -> None:
        """
        Run the synthetic plan data generator engine.
        """
        # Create a list of 1 to process as much as max_number_of_plans // how_many_to_generate_at_once times.
        data_to_process = [1] * (
            self._max_number_of_plans // self._how_many_to_generate_at_once
        )

        engine = AsyncDataGenerationJobEngine[
            int, CotSyntheticPlanDataGeneratorData, str
        ](
            data_to_process=data_to_process,
            task_fn=self._task_fn,
            save_fn=self.save_results,
            concurrency=2,
            save_interval=60.0,
            progress_interval=5.0,
        )
        await engine.run()

    async def _task_fn(self, _) -> CotSyntheticPlanDataGeneratorData:
        return await self._cot_synthetic_plan_data_generator.generate()

    def save_results(
        self, new_results: list[CotSyntheticPlanDataGeneratorData]
    ) -> None:
        """
        Appends a list of JSON-serializable objects to a file in JSON Lines format.

        Each object in new_results is serialized to a JSON string and written as a new line.
        """
        with open(self._output_path, "a", encoding="utf-8") as f:
            for result in new_results:
                json_line = json.dumps(result)
                f.write(json_line + "\n")

    @staticmethod
    def _get_already_processed_tasks(output_path: str) -> set[str]:
        """
        Get the already processed tasks from the output path.
        """
        if not os.path.exists(output_path):
            return set()

        # Get all the tasks that were already used as in-context examples from the output path.
        already_processed = set()
        with open(output_path, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                in_context_examples = data["in_context_examples"]
                for example in in_context_examples:
                    already_processed.add(example)

        return already_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--how_many_to_generate_at_once", type=int, required=True)
    parser.add_argument("--max_number_of_plans", type=int, required=True)
    parser.add_argument("--plan_data_path", type=str, required=True)
    parser.add_argument("--n_context_examples", type=int, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_processed_data(input_data_path: str) -> list[ProcessedData]:
    with open(input_data_path, "r") as f:
        unprocessed_data = json.load(f)

    return sorted(
        preprocess_webarena_data(unprocessed_data),
        key=lambda x: x["task"],
    )


if __name__ == "__main__":
    """
    This script is used to generate synthetic plan data for the given model.

    Example usage:
    python3 plan_and_act/cot/data_generation/synthetic_plan.py --input_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/raw_all_2036.json" \
        --plan_data_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/plan_annotation_data_all_2036_DeepSeek-R1-Distill-Llama-70B.jsonl" \
        --output_path "/home/lerdogan/VisualAgentBench/VAB-WebArena-Lite/plan_and_act/cot/data/synthetic_plan_data_10000.jsonl" \
        --model_name "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" \
        --how_many_to_generate_at_once 10 \
        --max_number_of_plans 10000 \
        --n_context_examples 5
        --verbose
    """
    args = parse_args()

    llm = LLM(
        model_name=args.model_name,
        max_tokens=16384,
        max_length=128000,
        temperature=0.6,  # Recommended for CoT by deepseek
    )

    processed_data = load_processed_data(args.input_data_path)

    cot_synthetic_plan_data_generator_engine = CotSyntheticPlanDataGeneratorEngine(
        llm=llm,
        how_many_to_generate_at_once=args.how_many_to_generate_at_once,
        in_context_example_path=args.plan_data_path,
        n_context_examples=args.n_context_examples,
        max_number_of_plans=args.max_number_of_plans,
        processed_data=processed_data,
        output_path=args.output_path,
        verbose=args.verbose,
    )

    asyncio.run(cot_synthetic_plan_data_generator_engine.run())
