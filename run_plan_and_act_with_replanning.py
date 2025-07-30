"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import asyncio
import glob
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import nest_asyncio
import openai
import requests
import torch
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import RenderHelper
from PIL import Image

from evaluation_harness import image_utils
from evaluation_harness.previous_rounds_evaluators import (
    previous_rounds_evaluator_router,
)
from plan_and_act.cot.inference.act import CoTExecutor
from plan_and_act.cot.inference.dynamic_act import DynamicExecutor
from plan_and_act.cot.inference.dynamic_plan import DynamicCoTPlanner
from plan_and_act.cot.inference.plan import CoTPlanner
from plan_and_act.cot.models import (
    LLM,
    ActInferencePreviousRound,
    DynamicExecutorInferenceInput,
    DynamicPlanInferenceInput,
    PlanHistory,
    PlanInferenceInput,
    ReplanDecision,
    ReplanStep,
)
from plan_and_act.cot.utils import get_action_information_from_action_str

nest_asyncio.apply()

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def text_wrap(text, font, max_width):
    lines = []
    paragraphs = text.split("\n")  # 按照 \n 分割文本为段落
    for paragraph in paragraphs:
        words = paragraph.split(" ")
        line = ""
        for word in words:
            # 临时行
            test_line = f"{line} {word}".strip()
            # 获取临时行的宽度
            test_line_bbox = font.getbbox(test_line)
            test_line_width = test_line_bbox[2] - test_line_bbox[0]
            if test_line_width <= max_width:
                # 如果临时行的宽度不超过图片宽度，继续添加单词
                line = test_line
            else:
                # 如果超过了最大宽度，保存当前行，开始新的一行
                lines.append(line)
                line = word
        # 添加每段的最后一行
        if line:
            lines.append(line)
        # 每个段落后添加一个空行，以保留段落的换行
        lines.append("")
    # 移除最后一个空行（不需要额外的空行）
    if lines[-1] == "":
        lines.pop()
    return lines


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument("--render", action="store_true", help="Render the browser")

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
            "webrl",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")

    # planner ip
    parser.add_argument("--actor_ip", type=str, required=True)
    parser.add_argument("--planner_ip", type=str, required=True)

    # Plan and Act related arguments
    parser.add_argument(
        "--cot_actor_model",
        type=str,
        required=True,
        help="Model name of the CoT actor",
    )
    parser.add_argument(
        "--cot_planner_model",
        type=str,
        required=True,
        help="Model name of the CoT planner",
    )

    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int], actions=None
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    if actions is None:
        last_action: Action = action_seq[-1]
        if last_action["action_type"] != ActionTypes.TYPE:
            if len(last_k_actions) >= k:
                if all(
                    [is_equivalent(action, last_action) for action in last_k_actions]
                ):
                    return True, f"Same action for {k} times"
        else:
            # check the action sequence
            if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
                return True, f"Same typing action for {k} times"
        return False, ""

    else:
        last_k_actions = actions[-k:]
        last_action = actions[-1]
        if len(last_k_actions) >= k:
            if all([action == last_action for action in last_k_actions]):
                return True, f"Same action for {k} times"
        return False, ""


def early_stop_with_previous_rounds(
    previous_rounds: list[
        tuple[DynamicExecutorInferenceInput, ActInferencePreviousRound]
    ],
    max_steps: int,
    thresholds: dict[str, int],
) -> tuple[bool, str]:
    """Check whether need to stop early using the previous_rounds list"""

    previous_actions = [output[1]["act"]["action_str"] for output in previous_rounds]

    # If there are no previous rounds, we can't stop early
    if not previous_actions:
        return False, ""

    # reach the max step
    if len(previous_actions) >= max_steps:
        return True, f"Reach max steps {max_steps}"

    # Case: parsing failure for k times - check for empty action_str
    k = thresholds["parsing_failure"]
    if len(previous_actions) >= k:
        last_k_rounds = previous_actions[-k:]
        if all([not round for round in last_k_rounds]):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    if len(previous_rounds) >= k:
        last_k_rounds = previous_rounds[-k:]
        last_action_str = last_k_rounds[1][-1]["act"].get("action_str", "")

        # Skip this check for empty actions
        if not last_action_str:
            return False, ""

        # Use the get_action_information_from_action_str function to extract action information
        try:
            last_action_info = get_action_information_from_action_str(last_action_str)
            last_action = last_action_info["action"]

            # For typing actions, we check if the same typing action appears k times
            if last_action["action_type"] == ActionTypes.TYPE:
                typing_count = 0
                for round in previous_rounds:
                    round_action_str = round[1]["act"].get("action_str", "")
                    if not round_action_str:
                        continue

                    try:
                        round_action_info = get_action_information_from_action_str(
                            round_action_str
                        )
                        round_action = round_action_info["action"]

                        if (
                            round_action["action_type"] == ActionTypes.TYPE
                            and round_action["text"] == last_action["text"]
                        ):
                            typing_count += 1
                    except Exception:
                        # If we can't parse the action, just continue
                        continue

                if typing_count >= k:
                    return True, f"Same typing action for {k} times"
            # For other actions, we check if the last k actions are the same
            else:
                same_action_count = 0
                for round in last_k_rounds:
                    round_action_str = round[1]["act"].get("action_str", "")
                    if not round_action_str:
                        continue

                    try:
                        round_action_info = get_action_information_from_action_str(
                            round_action_str
                        )
                        round_action = round_action_info["action"]

                        if is_equivalent(round_action, last_action):
                            same_action_count += 1
                    except Exception:
                        # If we can't parse the action, just continue
                        continue

                if same_action_count >= k:
                    return True, f"Same action for {k} times"
        except Exception:
            # If we can't parse the action, we can't check for repeating actions
            pass

    return False, ""


def update_action_history(
    path: str, task_id: int, actions: List[str], score: float = -0.1
):
    obj = {"task_id": task_id, "score": score, "actions": actions}
    json.dump(obj, open(path, "w"), indent=4)


def update_action_history_from_previous_rounds(
    path: str,
    task_id: int,
    previous_rounds: list[
        tuple[DynamicExecutorInferenceInput, ActInferencePreviousRound]
    ],
    score: float = -0.1,
):
    """Update action history using the previous_rounds list"""
    # Extract action strings from previous_rounds
    actions = []
    for input, output in previous_rounds:
        # From the input, get the current plan
        replan_steps_in_plan_history = input["plan_history"]["replan_steps"]
        if len(replan_steps_in_plan_history) == 0:
            plan = input["plan_history"]["initial_plan"]
        else:
            plan = replan_steps_in_plan_history[-1]["plan"]

        # From the output, get the action string and the reasoning.
        reasoning = output["act"]["reasoning"]
        action_str = output["act"]["action_str"]

        # Save all of these honestly.
        actions.append(
            {
                "plan": plan,
                "reasoning": reasoning,
                "action_str": action_str,
            }
        )

    obj = {"task_id": task_id, "score": score, "actions": actions}
    json.dump(obj, open(path, "w"), indent=4)


def save_trajectory_from_previous_rounds(
    path: str,
    task_id: int,
    intent: str,
    previous_rounds: list[
        tuple[DynamicExecutorInferenceInput, ActInferencePreviousRound]
    ],
):
    """Save trajectory to a jsonl file using the previous_rounds list"""
    traces = []
    for i, (input, output) in enumerate(previous_rounds):
        html = output["uncleaned_html"]
        reasoning = output["act"].get("reasoning", "")
        action_str = output["act"].get("action_str", "")
        item = {
            "trace_id": task_id,
            "index": i,
            "plan": (
                input["plan_history"]["initial_plan"]
                if i == 0
                else (
                    input["plan_history"]["replan_steps"][i - 1]["plan"]
                    if i - 1 < len(input["plan_history"]["replan_steps"])
                    else "Don't know sorry"
                )
            ),
            "prompt": intent if i == 0 else "** Simplified html **",
            "html": html,
            "response": f"<think>\n{reasoning}\n</think>\n[Start of Action]\n{action_str}\n[End of Action]\n",
            "target": intent,
        }
        traces.append(item)

    with open(path, "w") as f:
        for item in traces:
            f.write(json.dumps(item) + "\n")


def test(args: argparse.Namespace, config_file_list: list[str]) -> None:
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        # "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == "visualwebarena":
        if caption_image_fn and args.eval_captioning_model == args.captioning_model:
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                (
                    torch.float16
                    if (
                        torch.cuda.is_available()
                        and args.eval_captioning_model_device == "cuda"
                    )
                    else torch.float32
                ),
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []

                sites = _c["sites"]

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(
                        _c["storage_state"]
                    ), f"Cookie file not found: {_c['storage_state']}"
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            input_image = Image.open(
                                requests.get(image_path, stream=True).raw
                            )
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            actor_llm = LLM(
                max_length=128000,
                max_tokens=1024,
                model_name=args.cot_actor_model,
                base_url=args.actor_ip,
            )
            planner_llm = LLM(
                max_length=128000,
                max_tokens=4096,
                model_name=args.cot_planner_model,
                base_url=args.planner_ip,
            )
            cot_planner = CoTPlanner(llm=planner_llm)
            dynamic_planner = DynamicCoTPlanner(llm=planner_llm)
            actor = DynamicExecutor(llm=actor_llm)

            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)
            meta_data = {"action_history": ["None"]}
            out_path = os.path.join(args.result_dir, "actions", f"{task_id}.json")

            # Initialize a list to store previous rounds for the CoT executor
            previous_rounds: list[
                tuple[DynamicExecutorInferenceInput, ActInferencePreviousRound]
            ] = []

            os.makedirs(os.path.join(args.result_dir, "screehshots"), exist_ok=True)
            if os.path.exists(
                os.path.join(args.result_dir, "screehshots", f"{task_id}")
            ):
                shutil.rmtree(
                    os.path.join(args.result_dir, "screehshots", f"{task_id}")
                )
            os.makedirs(os.path.join(args.result_dir, "screehshots", f"{task_id}"))

            while True:
                update_action_history_from_previous_rounds(
                    out_path, task_id, previous_rounds, score=-0.1
                )
                # Use our new early stop function with previous_rounds
                early_stop_flag, stop_info = early_stop_with_previous_rounds(
                    previous_rounds, max_steps, early_stop_thresholds
                )

                # Get the current HTML from the observation
                current_html = str(state_info["observation"]["text"])

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        # First create the plan/replan and use the generated plan to form the new plan history.
                        if len(previous_rounds) == 0:
                            print(">> Using the COT planner")
                            # If this is the first rounds, use the cot aplnner
                            plan = asyncio.run(
                                cot_planner.plan(
                                    PlanInferenceInput(
                                        task=intent,
                                        initial_html_state=current_html,
                                    )
                                )
                            )
                            print(">> Plan generated by the COT planner")
                            plan_history = PlanHistory(
                                initial_plan=plan,
                                initial_html_state=current_html,
                                replan_steps=[
                                    ReplanStep(
                                        index_within_task=0,
                                        plan=plan,
                                        html_state=current_html,
                                        previous_executor_actions=[],  # Don't matter
                                        replan_decision=ReplanDecision(
                                            needs_replan=False,
                                            reasoning=plan["reasoning"],  # Don't matter
                                        ),
                                    )
                                ],
                            )
                        else:
                            print(">> Using the dynamic planner")
                            # If this is not the first rounds, use the dynamic planner
                            # Get the previous plan and previous actions
                            plan_history = previous_rounds[-1][0]["plan_history"]
                            initial_plan = plan_history["initial_plan"]
                            initial_html_state = plan_history["initial_html_state"]
                            previous_plan = plan_history["replan_steps"][-1]["plan"]
                            previous_actions = [
                                output[1]["act"] for output in previous_rounds
                            ]

                            plan = asyncio.run(
                                dynamic_planner.replan(
                                    DynamicPlanInferenceInput(
                                        task=intent,
                                        previous_plan=previous_plan,
                                        current_html_state=current_html,
                                        previous_actions=previous_actions,
                                    )
                                )
                            )
                            print(">> Plan generated by the dynamic planner")

                            previous_replan_steps = plan_history["replan_steps"]
                            previous_replan_steps.append(
                                ReplanStep(
                                    index_within_task=len(previous_replan_steps),
                                    plan=plan,
                                    html_state=current_html,
                                    previous_executor_actions=[],  # Don't matter
                                    replan_decision=ReplanDecision(
                                        needs_replan=False,
                                        reasoning=plan["reasoning"],  # Don't matter
                                    ),
                                )
                            )
                            plan_history = PlanHistory(
                                initial_plan=initial_plan,
                                initial_html_state=initial_html_state,
                                replan_steps=previous_replan_steps,
                            )

                        # Then call the executor with this updated plan/replan
                        action_input = DynamicExecutorInferenceInput(
                            task=intent,
                            plan_history=plan_history,
                            previous_rounds=[
                                previous_round[1] for previous_round in previous_rounds
                            ],
                            current_round=ActInferencePreviousRound(
                                act={"action_str": "", "reasoning": ""},
                                uncleaned_html=current_html,
                            ),
                        )
                        action = asyncio.run(actor.act(action_input))
                        print(">> Action generated by the dynamic executor")
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")

                if "action_str" in action:
                    action_str = f"<think>\n{action['reasoning']}\n</think>\n[Start of Action]\n{action['action_str']}\n[End of Action]\n"
                    action_info = get_action_information_from_action_str(
                        action["action_str"]
                    )
                    print("Action String: ", action["action_str"])

                    # Store this round for future reference in the ExecutorAction format
                    previous_rounds.append(
                        (
                            action_input,
                            ActInferencePreviousRound(
                                act=action, uncleaned_html=current_html
                            ),
                        )
                    )

                    action = action_info["action"]
                else:
                    # This is the stop action if there was an early stop.
                    action_str = "stop"

                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )

                current_screenshot = os.path.join(
                    args.result_dir,
                    "screehshots",
                    f"{task_id}",
                    f"{len(previous_rounds)}.png",
                )
                _ = env.page.viewport_size
                env.page.screenshot(path="/dev/null")
                env.page.screenshot(path=current_screenshot)
                element_id = action["element_id"]
                if element_id != "":
                    element = env.page.query_selector(f"[data-label-id='{element_id}']")
                    if element:
                        bbox = element.bounding_box()
                        bbox = [
                            int(bbox["x"]),  # type: ignore
                            int(bbox["y"]),  # type: ignore
                            int(bbox["width"]),  # type: ignore
                            int(bbox["height"]),  # type: ignore
                        ]
                        image = cv2.imread(current_screenshot)
                        cv2.rectangle(
                            image,
                            (bbox[0], bbox[1]),
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                            (0, 255, 0),
                            2,
                        )
                        cv2.circle(
                            image,
                            (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)),
                            radius=0,
                            color=(0, 255, 0),
                            thickness=2,
                        )
                        cv2.imwrite(current_screenshot, image)

                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            # save trajectory
            if args.observation_type == "webrl":
                current_path = os.path.join(
                    args.result_dir, "traces", f"{task_id}.jsonl"
                )
                save_trajectory_from_previous_rounds(
                    current_path, task_id, intent, previous_rounds
                )

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            # Use our new previous_rounds_evaluator_router instead of the original evaluator_router
            evaluator = previous_rounds_evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                previous_rounds=[
                    previous_round[1] for previous_round in previous_rounds
                ],
                config_file=config_file,
                page=env.page,
            )

            update_action_history_from_previous_rounds(
                out_path, task_id, previous_rounds, score=score
            )
            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")
        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    os.makedirs(os.path.join(result_dir, "actions"), exist_ok=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [os.path.basename(f).split(".")[0].split("_")[1] for f in result_files]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        try:
            with open(f"{result_dir}/actions/{task_id}.json", "r") as f:
                jd = json.load(f)
        except:
            jd = {}
        if task_id not in task_ids or jd.get("score", -1) < 0:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 3.0
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    print(test_config_base_dir)

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, test_file_list)
