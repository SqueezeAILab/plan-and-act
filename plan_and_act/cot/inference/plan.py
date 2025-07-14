import json

from tqdm import tqdm

from plan_and_act.cot.models import (
    LLM,
    CotPlanAnnotatorDataGeneratorData,
    CotSyntheticPlanDataGeneratorData,
    Plan,
    PlanInferenceInput,
    ServerMessage,
    TorchTuneData,
)
from plan_and_act.cot.utils import (
    clean_html,
    format_training_data_for_torchtune,
    prepare_completions_prompt_for_reasoning_models,
)


class CoTPlanner:
    """
    This class defines the prompts that the actual Planner model in Plan-and-Act project will use during inference. These prompts should be used to both:
        - Generate the training data for the Planner
        - Generate the prompt for the Planner during evaluation
    It is key that both cases the prompts are the same.
    """

    _MAX_RETRY = 3

    _llm: LLM

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    async def plan(self, input: PlanInferenceInput) -> Plan:
        """
        This function runs the Planner model during evaluation.
        """
        messages: list[ServerMessage] = [
            {"role": "user", "content": self.construct_prompt(input)},
        ]

        reasoning = None
        plan = None
        for _ in range(self._MAX_RETRY):
            try:
                prompt = prepare_completions_prompt_for_reasoning_models(
                    messages,
                    self._llm._tokenizer,  # type: ignore
                )

                print(f"Calling the COT planner")

                response = await self._llm.acall_completions(prompt)
                if response is None:
                    raise ValueError("Failed to generate plan.")

                if "[Start of Plan]" in response and "[End of Plan]" in response:
                    plan = (
                        response.split("[Start of Plan]")[1]
                        .split("[End of Plan]")[0]
                        .strip()
                    )
                    reasoning = reasoning or response.split("</think>")[0].strip()
                    break

                messages.extend(
                    [
                        {"role": "assistant", "content": response},
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

        if plan is None or reasoning is None:
            raise ValueError("Failed to generate plan.")

        return Plan(reasoning=reasoning, plan=plan)

    def construct_prompt(self, input: PlanInferenceInput) -> str:
        """
        This function constructs the prompt for the Planner during evaluation. The planner only needs to have:
            - System instructions telling it what it needs to do
            - The user task
            - The initial HTML state
        """

        task = input["task"]
        initial_html_state = clean_html(
            input["initial_html_state"], prettify=True, strip_annotation=True
        )

        prompt = f"""## Goal
You are the WebPlanner, an expert plan generator for web navigation tasks. Your role is to create detailed, comprehensive execution plans for accomplishing web-based tasks. You will be provided with a task query and the initial HTML state of the webpage, and you must generate a plan that outlines the high-level goals and steps needed to complete the task.

## Task
Here is the task that the user wants to accomplish: 

{task}

## Initial HTML State
Here is the initial HTML state of the webpage:

{initial_html_state}

## Guidelines
Guidelines for Plan Generation:
- Create a detailed plan that breaks down the task into clear, high-level goals
- For each goal, provide comprehensive context including:
  * Clear description of what needs to be accomplished and the sub-goals needed to achieve it
  * Specific actions that might need to be taken
  * Expected outcomes and webpage state changes
  * How to identify and interact with relevant elements
  * How to handle dynamic or unknown content
- When dealing with dynamic content:
  * Describe how to analyze search results or lists
  * Explain methods to extract information dynamically
  * Show how to make decisions based on available content
  * Avoid assuming specific values or positions
- If the task requires analysis and a final answer:
  * Explain the analysis process
  * Describe how to extract and verify the required information
  * Don't assume specific values - focus on the method

- **Important:** Before presenting the final plan, you should write down your detailed chain of thought and reasoning. You MUST be comprehensive and thorough in your reasoning. You are allowed to think as much as you need to before outputting the final plan.
- After your thinking process, output the final plan enclosed between the tags "[Start of Plan]" and "[End of Plan]".

Based on the above, please generate a detailed execution plan."""

        return prompt


def create_plan_training_data(
    model_name: str,
    plan_data_path: str,
    output_path: str,
) -> None:
    """
    This function creates the training data for the Planner model using the generated plan data from the `CotPlanAnnotatorDataGenerator`.
    """
    data: list[CotPlanAnnotatorDataGeneratorData] = []
    with open(plan_data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    input_output_pairs: list[tuple[PlanInferenceInput, Plan]] = []
    for item in data:
        plan_training_data = item["data"][2]
        input_output_pairs.append(
            (
                PlanInferenceInput(
                    task=plan_training_data["task"],
                    initial_html_state=plan_training_data["initial_html_state"],
                ),
                plan_training_data["plan"],
            )
        )

    # Create a dummy LLM and planner object so that we can use the `construct_prompt` function.
    cot_planner = CoTPlanner(llm=LLM(0, 0, model_name))

    torchtune_data: list[TorchTuneData] = []
    for input, output in tqdm(input_output_pairs, desc="Creating plan training data"):
        prompt = cot_planner.construct_prompt(input)

        # Create the formatted training data. This function will construct the input message with the think token already appended so we don't need it in the output.
        formatted_data = format_training_data_for_torchtune(
            messages=[{"role": "user", "content": prompt}],
            tokenizer=cot_planner._llm._tokenizer,  # type: ignore
            output=f"{output['reasoning']}\n</think>\n\n[Start of Plan]\n\n{output['plan']}\n\n[End of Plan]",
        )
        torchtune_data.append(formatted_data)

    with open(output_path, "w") as f:
        json.dump(torchtune_data, f, indent=4, ensure_ascii=False)


def create_synthetic_plan_training_data(
    model_name: str,
    plan_data_path: str,
    output_path: str,
) -> None:
    data: list[CotSyntheticPlanDataGeneratorData] = []
    with open(plan_data_path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    input_output_pairs: list[tuple[PlanInferenceInput, Plan]] = []
    for item in data:
        for plan_training_data in item["datas"]:
            input_output_pairs.append(
                (
                    PlanInferenceInput(
                        task=plan_training_data["task"],
                        initial_html_state=plan_training_data["initial_html_state"],
                    ),
                    plan_training_data["plan"],
                )
            )

    # Create a dummy LLM and planner object so that we can use the `construct_prompt` function.
    cot_planner = CoTPlanner(llm=LLM(0, 0, model_name))

    torchtune_data: list[TorchTuneData] = []
    for input, output in tqdm(input_output_pairs, desc="Creating plan training data"):
        prompt = cot_planner.construct_prompt(input)

        # Create the formatted training data. This function will construct the input message with the think token already appended so we don't need it in the output.
        formatted_data = format_training_data_for_torchtune(
            messages=[{"role": "user", "content": prompt}],
            tokenizer=cot_planner._llm._tokenizer,  # type: ignore
            output=f"{output['reasoning']}\n</think>\n\n[Start of Plan]\n\n{output['plan']}\n\n[End of Plan]",
        )
        torchtune_data.append(formatted_data)

    with open(output_path, "w") as f:
        json.dump(torchtune_data, f, indent=4, ensure_ascii=False)
