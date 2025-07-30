from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os

import aiometer

from plan_and_act.cot.models import (
    LLM,
    ActionStep,
    AsyncDataGenerationJobEngine,
    CotPlanAnnotatorDataGeneratorData,
    Plan,
    PlanAnnotatorOutput,
    PlanTrainingData,
    ProcessedData,
    ServerMessage,
    WebArenaLiteAction,
    WebArenaLiteWebsite,
)
from plan_and_act.cot.utils import (
    classify_website,
    clean_html,
    format_action_steps,
    format_action_string,
    format_last_exit_action,
    get_action_information,
    get_html_from_action,
    prepare_completions_prompt_for_reasoning_models,
    preprocess_webarena_data,
)


class CoTPlanAnnotator:
    _MAX_RETRY = 3

    _llm: LLM
    _verbose: bool

    def __init__(self, llm: LLM, verbose: bool = False) -> None:
        self._llm = llm
        self._verbose = verbose

    async def annotate(
        self, processed_data: ProcessedData
    ) -> CotPlanAnnotatorDataGeneratorData:
        """
        Plan annotator generates the following data:
            - action_step_descriptions: A list of action step descriptions. Used for logging purposes.
            - plan: The final plan. This is also for logging purposes.
            - plan_training_data: The training data for the plan. This is used to construct the training data for the planner model.

        The plan annotator is a two step process:
            1. Generate the action step descriptions for each action in the task.
            2. Using the action descriptions, generate the plan. First the LLM is asked to generate the plan (with some internal reasoning) and then it is again asked to generate the reasoning for the plan pretending to only have seen the user query and the initial HTML state.
        """

        # Sort the actions by the index within task.
        actions = sorted(
            processed_data["actions"], key=lambda x: x["index_within_task"]
        )

        # 1. Get all the action step descriptions except for the last exit action.
        action_step_descriptions = await self.generate_action_step_descriptions(actions)

        # 2. Using the action descriptions, generate the plan.
        plan = await self.generate_plan(
            action_step_descriptions,
            processed_data["task"],
            processed_data["initial_html_state"],
            actions[-1],
        )

        return CotPlanAnnotatorDataGeneratorData(
            task=processed_data["task"],
            data=(
                action_step_descriptions,
                plan,
                PlanTrainingData(
                    task=processed_data["task"],
                    initial_html_state=processed_data["initial_html_state"],
                    plan=Plan(
                        reasoning=plan["reasoning"],
                        plan=plan["plan"],
                    ),
                ),
            ),
        )

    async def generate_action_step_descriptions(
        self,
        actions: list[WebArenaLiteAction],
    ) -> list[ActionStep]:
        """
        This function generates the action step descriptions for all the actions in the task except for the last exit action.
        """
        action_step_description_tasks = []

        if len(actions) == 0:
            raise ValueError("No actions found in the processed data.")

        # 1. Get all the action step descriptions except for the last exit action.
        for start_action, end_action in zip(actions, actions[1:]):
            action_step_description_tasks.append(
                functools.partial(
                    self._generate_action_step_description,
                    start_action,
                    end_action,
                )
            )

        # NOTE: I am assuming this section will not throw any errors. But need to look for errors in the future.
        action_step_descriptions: list[ActionStep] = await aiometer.run_all(
            action_step_description_tasks,
            max_at_once=8,
            max_per_second=8,
        )

        return action_step_descriptions

    async def _generate_action_step_description(
        self, start_action: WebArenaLiteAction, end_action: WebArenaLiteAction
    ) -> ActionStep:
        """
        This function generates the action step description for a given start and end action.

        The action step description is a detailed description of the action that the user took. It is used for representing the action in natural language and provide context for the planner model while it is generating the plan for ALL the actions in the task.
        """
        try:
            start_dont_remove_ids = {
                get_action_information(start_action)["action"]["element_id"]
            }
        except Exception as e:
            start_dont_remove_ids = set()

        cleaned_start_action_html = clean_html(
            get_html_from_action(start_action),
            prettify=True,
            strip_annotation=True,
            dont_remove_ids=start_dont_remove_ids,
        )

        try:
            end_dont_remove_ids = {
                get_action_information(end_action)["action"]["element_id"]
            }
        except Exception as e:
            end_dont_remove_ids = set()

        cleaned_end_action_html = clean_html(
            get_html_from_action(end_action),
            prettify=True,
            strip_annotation=True,
            dont_remove_ids=end_dont_remove_ids,
        )

        prompt = f"""# Task
You are the ActionStepDescriber, an AI assistant that generates a comprehensive description for a single web action. You will be provided with the HTML state of the page before and after the action as well as the details about the specific action the user took (target element, event type, and event-specific data), and you are tasked to create a comprehensive description that will allow someone to understand the action in detail. Be thorough and include all necessary details to:

- Understand the intention of the action
- Understand the expected outcome of the action by including what change the action is expected to cause on the page (how does the action change HTML 1 to become HTML 2)
- Confidently identify the correct element to interact with by including any relevant element attributes, text content, or positioning that will help locate the target

Output Instructions:
- Before outputting the action description, write down your detailed chain of thought and reasoning steps. Be comprehensive and thorough in your reasoning.
- After your thinking process, output the action description enclosed between the "[Start of Action Description]" and "[End of Action Description]" tags.

## HTML 1
Here is the HTML of the page before the action:

{cleaned_start_action_html}

## HTML 2
Here is the HTML of the page after the action:

{cleaned_end_action_html}

## Action
Here is the action that the user took:

{format_action_string(start_action, use_simple_html=False)}
"""
        messages: list[ServerMessage] = [
            {"role": "user", "content": prompt},
        ]

        action_description = None
        reasoning = None
        for _ in range(CoTPlanAnnotator._MAX_RETRY):
            try:
                prompt = prepare_completions_prompt_for_reasoning_models(
                    messages,
                    self._llm._tokenizer,  # type: ignore
                )
                response = await self._llm.acall_completions(prompt)
                if response is None:
                    raise ValueError("Failed to generate action step description.")

                # Check if the response contains the action description
                if (
                    "[Start of Action Description]" in response
                    and "[End of Action Description]" in response
                ):
                    # Extract the reasoning from the response (any text before the </think> tag)
                    reasoning = reasoning or response.split("</think>")[0].strip()
                    action_description = (
                        response.split("[Start of Action Description]")[1]
                        .split("[End of Action Description]")[0]
                        .strip()
                    )
                    break

                # If the response does not contain the action description, retry.
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": response,
                        },
                        {
                            "role": "user",
                            "content": "You need to ensure that you enclose the final action description between the '[Start of Action Description]' and '[End of Action Description]' tags.",
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

        if action_description is None or reasoning is None:
            raise ValueError("Failed to generate action step description.")

        return ActionStep(
            task=start_action["task"],
            action=start_action,
            reasoning=reasoning,
            action_description=action_description,
        )

    async def generate_plan(
        self,
        action_step_descriptions: list[ActionStep],
        task: str,
        initial_html_state: str,
        last_exit_action: WebArenaLiteAction,
    ) -> PlanAnnotatorOutput:
        """
        This function generates the plan for the given action step descriptions.

        The plan is a detailed plan that describes the sequence of actions that will be taken to complete the task. In order to generate the plan, we first give the LLM the list of actions (user events) and their descriptions and then ask it to generate the plan. The LLM is instructed to:
          - Analyze the provided action steps (which include an HTML snippet, user event, and a description).
          - Build an execution plan that reduces these steps into high-level goals.
          - First, document internal reasoning for the plan.
          - Then output the final plan enclosed between the tags "[Start of Plan]" and "[End of Plan]".

        After the plan is generated, we again ask the LLM to pretend to be a user that has only access to the user query and the initial HTML state and then generate the reasoning for the plan. This LLM needs to generate the reasoning as if its predicting the expected outcomes of each action it needs to take (as well as the expected outcome of the individual high level goals and the final plan) and predict what kind of page/HTML state it will encounter as if it is a world model of this particular website. This reasoning will be used to form the training data for a planner that will be trained to only take the user query and the initial HTML state and then generate the perfect plan that was generated by the first LLM.
        """

        first_stage_response = await self._generate_plan_stage_1(
            action_step_descriptions,
            task,
            last_exit_action,
        )
        messages, _, plan_generation_reasoning = first_stage_response

        second_stage_response = await self._generate_plan_stage_2(
            task,
            initial_html_state,
            messages,
        )
        return PlanAnnotatorOutput(
            task=task,
            plan_generation_reasoning=plan_generation_reasoning,
            plan=second_stage_response["plan"],
            reasoning=second_stage_response["reasoning"],
        )

    async def _generate_plan_stage_1(
        self,
        action_step_descriptions: list[ActionStep],
        task: str,
        last_exit_action: WebArenaLiteAction,
    ) -> tuple[list[ServerMessage], str, str]:
        # 1) Format the recorded action step descriptions using the utility function
        actions_steps_formatted_str = format_action_steps(
            action_step_descriptions,
            "Action",
            include_html_snippet=True,
            use_simple_html=True,
        )

        # 2) Format the last exit action using the utility function
        last_exit_section = format_last_exit_action(
            last_exit_action, include_html_snippet=True
        )

        # 3) Construct the prompt for the plan generation.
        prompt = f"""## Goal
You are the ExecutionPlanReducer, an AI assistant that reduces a sequence of web action steps into a coherent, high-level execution plan for accomplishing a task. Your objective is to analyze the provided action steps—which include details about the web page state before and after each action—and generate an execution plan outlining high-level goals along with the sub-goals needed to achieve each goal.

Guidelines:
- Analyze the provided action steps (which include an HTML snippet, user event, and a description).
- Build an execution plan that reduces these steps into high-level goals.
- Each high-level goal should be detailed and include any information and context needed to fully understand and execute it succesfully, such as a clear description of the goal, the sub-goals needed to achieve it, specific actions that might need to be taken, expected outcomes, enough context to identify the correct elements to interact with, etc.
- **Important:** Before presenting the final plan, you should write down your detailed chain of thought and reasoning steps. You MUST be comprehensive and thorough in your reasoning. You are allowed to think as much as you need to before outputting the final plan.
- After your thinking process, output the final plan enclosed between the tags "[Start of Plan]" and "[End of Plan]".

## Task
Here is the user query that these actions are meant to accomplish:

{task}

## Actions
Here is the list of actions (user events) and their descriptions:

{actions_steps_formatted_str}

{last_exit_section}

Based on the above, please generate the execution plan.
"""

        # Add a small prompts for the postmill website saying that this website is a reddit website and that the model shouldn't be confused or navigate to the reddit website.
        if (
            classify_website(get_html_from_action(last_exit_action))
            == WebArenaLiteWebsite.REDDIT
        ):
            prompt += "\nNote: Even though the websites show that we are in the 'Postmill' website, this is actually a reddit website. You shouldn't be confused by this and navigate to the reddit website."

        if self._verbose:
            print(f"@@@@ Stage 1 prompt:\n\n{prompt}\n\n@@@@")

        messages: list[ServerMessage] = [
            {"role": "user", "content": prompt},
        ]

        plan_generation_reasoning = None
        plan = None
        for _ in range(CoTPlanAnnotator._MAX_RETRY):
            try:
                prompt = prepare_completions_prompt_for_reasoning_models(
                    messages,
                    self._llm._tokenizer,  # type: ignore
                )
                response = await self._llm.acall_completions(prompt)
                if response is None:
                    raise ValueError("Failed to generate plan.")

                # Check if the response contains the plan.
                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    # Extract the reasoning from the response (any text before the </think> tag)
                    plan_generation_reasoning = (
                        plan_generation_reasoning
                        or response.split("</think>")[0].strip()
                    )
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
            raise ValueError("Failed to generate plan.")

        if self._verbose:
            print(f"@@@@ Stage 1 response:\n\n{response}\n\n@@@@")

        return messages, plan, plan_generation_reasoning

    async def _generate_plan_stage_2(
        self,
        task: str,
        initial_html_state: str,
        prev_messages: list[ServerMessage],
    ) -> Plan:
        """
        Second stage of plan generation that continues the conversation with the LLM
        to generate reasoning as if only having access to initial state.
        """

        prompt = f"""Now, I want you to be an annotator LLM helping to generate training data for an execution planner model. You have already generated a perfect plan for the given action steps in the previous conversation; however, the execution planner model only has access to the task query and the initial HTML state (not the action steps).

Your task is to generate reasoning and a plan by pretending that you are the execution planner that has only access to the task query and the initial HTML state. This execution planner must be an expert in web-based task planning, creating execution plans that are as comprehensive as the one you previously generated (as if it can "see the future", as if it is a "world model" of this particular website). Its reasoning should anticipate outcomes, predict webpage state changes, and dynamically adapt to unknown content. This reasoning will serve as training data to teach a model to generate optimal plans using only the task query and initial HTML state. Hence, this reasoning should not directly mention the list of user events you were given previously but actually be "intelligent" enough to predict those user events/actions and their effects on the webpage.
         
Here is the task query and the initial HTML state that the execution planner will use to generate the plan:
        
Task Query:

"{task}"

Initial HTML State:

{clean_html(initial_html_state, prettify=True, strip_annotation=True)}

Some special notes:
- Since the execution planner only has access to the initial HTML state, it must handle dynamic content carefully. It should not assume specific search results or values.
  * Example: When searching for gas stations, explain how to analyze results rather than assuming specific stations
  * Example: When finding order #178, explain how to locate and verify the order rather than assuming its position

- Similarly, if the task requires the analysis of some page and then give the final answer, the planner should 
not assume the final answer but instead should explain how to analyze the page and extract an answer from it. 
The agent that will execute the plan of the execution planner will be the one that actually gives the final
answer.

- Guidelines for handling dynamic/unknown content:
  * Describe how to analyze search results
  * Explain methods to extract information dynamically
  * Show how to make decisions based on available content
  * Avoid hardcoding any specific values or answers

Remember:
- You are generating training data to teach another model how to reason about generating plans
- The reasoning should be very detailed, comprehensive, and thorough about the expected outcomes, webpage state changes, and how to achieve the high level goals. It should at least be as detailed as the reasoning you generated in the previous conversation and ideally include more analysis and world/task awareness since it only has access to the initial HTML state.
- Focus on the high-level goals and explain how the webpage will look like after each step like a "world model". 
- Explain how to handle dynamic content without assuming specific values
- Output your final plan between [Start of Plan] and [End of Plan] tags.
- Avoid mentioning the "execution planner" in your reasoning. Always talk in "I" or "we" instead.
"""

        # Append our new prompt to the existing conversation
        prev_messages.append({"role": "user", "content": prompt})

        if self._verbose:
            print(f"@@@@ Stage 2 prompt:\n\n{prompt}\n\n@@@@")

        reasoning = None
        plan = None
        for _ in range(self._MAX_RETRY):
            try:
                prompt = prepare_completions_prompt_for_reasoning_models(
                    prev_messages,
                    self._llm._tokenizer,  # type: ignore
                )
                response = await self._llm.acall_completions(prompt)
                if response is None:
                    raise ValueError("Failed to generate stage 2 plan reasoning.")

                # Check if reasoning and plan are present in the response.
                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    reasoning = response.split("</think>")[0].strip()
                    plan = (
                        response.split("[Start of Plan]")[1]
                        .split("[End of Plan]")[0]
                        .strip()
                    )
                    break

                prev_messages.extend(
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
                prev_messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {e}",
                    }
                )

        if reasoning is None or plan is None:
            raise ValueError("Failed to generate stage 2 plan reasoning.")

        if self._verbose:
            print(f"@@@@ Stage 2 response:\n\n{response}\n\n@@@@")

        return Plan(
            reasoning=reasoning,
            plan=plan,
        )


class CotPlanAnnotatorDataGenerator:
    _cot_planner: CoTPlanAnnotator

    # Input data
    _processed_data: list[ProcessedData]
    _already_processed: set[str]

    # Output paths
    _output_path: str

    # Results aggreagator
    _results: list[CotPlanAnnotatorDataGeneratorData]

    def __init__(
        self,
        cot_planner: CoTPlanAnnotator,
        processed_data: list[ProcessedData],
        output_path: str,
    ) -> None:
        self._cot_planner = cot_planner
        self._processed_data = processed_data
        self._already_processed = (
            CotPlanAnnotatorDataGenerator._get_already_processed_tasks(output_path)
        )
        self._results = []
        self._output_path = output_path

    async def run(self) -> None:
        # 1) Determine which items have not yet been processed. In the meanwhile, also populate the results list.
        unprocessed_data = [
            d for d in self._processed_data if d["task"] not in self._already_processed
        ]

        # 2) Create our async job engine
        engine = AsyncDataGenerationJobEngine[
            ProcessedData, CotPlanAnnotatorDataGeneratorData, str
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

    def save_results(
        self, new_results: list[CotPlanAnnotatorDataGeneratorData]
    ) -> None:
        """
        Appends a list of JSON-serializable objects to a file in JSON Lines format.

        Each object in new_results is serialized to a JSON string and written as a new line.
        """
        with open(self._output_path, "a", encoding="utf-8") as f:
            for result in new_results:
                json_line = json.dumps(result)
                f.write(json_line + "\n")

    async def _task_fn(self, data: ProcessedData) -> CotPlanAnnotatorDataGeneratorData:
        return await self._cot_planner.annotate(data)

    @staticmethod
    def _get_already_processed_tasks(output_path: str) -> set[str]:
        if not os.path.exists(output_path):
            return set()

        already_processed = set()
        with open(output_path, "r") as f:
            for line in f:
                data = json.loads(line)
                already_processed.add(data["task"])

        print(f"Already processed {len(already_processed)} tasks.")
        return already_processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--start_index", type=int, required=False, default=0)
    parser.add_argument("--end_index", type=int, required=False, default=None)
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
    args = parse_args()

    llm = LLM(
        model_name=args.model_name,
        max_tokens=16384,
        max_length=128000,
        temperature=0.6,  # Recommended for CoT by deepseek
    )

    cot_planner = CoTPlanAnnotator(llm=llm, verbose=args.verbose)
    processed_data = load_processed_data(args.input_data_path)
    processed_data = processed_data[args.start_index : args.end_index]

    print(
        f"Processing {len(processed_data)} tasks. From {args.start_index} to {args.end_index}"
    )

    output_path = args.output_path.replace(
        ".jsonl",
        f"_{args.start_index}_{args.end_index}_{args.model_name.split('/')[-1]}.jsonl",
    )

    data_generator = CotPlanAnnotatorDataGenerator(
        cot_planner=cot_planner,
        processed_data=processed_data,
        output_path=output_path,
    )
    asyncio.run(data_generator.run())
