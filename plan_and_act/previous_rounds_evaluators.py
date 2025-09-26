"""base class for evaluation using previous_rounds"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union
from urllib.parse import urljoin, urlparse

import evaluate
import requests
from beartype import beartype
from beartype.typing import Dict, List
from browser_env.env_config import ACCOUNTS, SHOPPING
from evaluation_harness import image_utils
from llms.providers.openai_utils import generate_from_openai_chat_completion
from nltk.tokenize import word_tokenize
from PIL import Image
from playwright.sync_api import Page

from plan_and_act.cot.models import ActInferenceOutput, ActInferencePreviousRound
from plan_and_act.cot.utils import (
    exact_match,
    get_action_information_from_action_str,
    must_include,
)

logger = logging.getLogger("logger")


@beartype
class PreviousRoundsEvaluator(object):
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(
        previous_rounds: List[ActInferencePreviousRound],
    ) -> ActInferenceOutput:
        """Get the last action from previous_rounds"""
        if not previous_rounds:
            raise ValueError("previous_rounds is empty")

        last_round = previous_rounds[-1]
        return last_round["act"]

    @staticmethod
    def get_last_html(previous_rounds: List[ActInferencePreviousRound]) -> str:
        """Get the last HTML from previous_rounds"""
        if not previous_rounds:
            raise ValueError("previous_rounds is empty")

        last_round = previous_rounds[-1]
        return last_round["uncleaned_html"]


@beartype
class NumericPreviousRoundsEvaluator(PreviousRoundsEvaluator):
    """Check if the numerical relationship is correct"""

    @staticmethod
    @beartype
    def str_2_int(s: str) -> Optional[int]:
        try:
            s = s.strip()
            if "," in s:
                s = s.replace(",", "")

            return int(s)
        except ValueError:
            # Return None if the string cannot be converted to int
            print(f"[NumericEvaluator error]: Cannot convert {s} to int")
            return None

    @staticmethod
    @beartype
    def compare_inequality(
        value: Union[int, float], inequality: str, tol: float = 1e-8
    ) -> bool:
        """
        Compare a value (int or float) against an inequality string.

        Args:
        - value (int/float): The value to be compared.
        - inequality (str): Inequality in the form of "< 700", ">= 300", etc.
        - tol (float): Tolerance for floating point comparisons.

        Returns:
        - bool: True if the value satisfies the inequality, False otherwise.
        """
        # Extract the operator and the number from the inequality string
        ops = {
            "<=": lambda x, y: x <= y + tol,
            ">=": lambda x, y: x >= y - tol,
            "==": lambda x, y: abs(x - y) <= tol,
            "<": lambda x, y: x < y + tol,
            ">": lambda x, y: x > y - tol,
        }

        for op, func in ops.items():
            if op in inequality:
                _, num = inequality.split(op)
                return func(value, float(num.strip()))

        raise ValueError(f"Invalid inequality string: {inequality}")


@beartype
class StringPreviousRoundsEvaluator(PreviousRoundsEvaluator):
    """Check whether the answer is correct with:
    exact match: the answer is exactly the same as the reference answer
    must include: each phrase in the reference answer must be included in the answer
    fuzzy match: the answer is similar to the reference answer, using LLM judge
    """

    @staticmethod
    @beartype
    def clean_answer(answer: str) -> str:
        if answer.startswith("'") and answer.endswith("'"):
            answer = answer[1:-1]
        elif answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        return answer.lower()

    @staticmethod
    @beartype
    def exact_match(ref: str, pred: Union[str, int]) -> float:
        return exact_match(ref, pred)

    @staticmethod
    @beartype
    def must_include(ref: str, pred: str) -> float:
        return must_include(ref, pred, token_subset_any_len=True)

    @staticmethod
    @beartype
    def must_exclude(ref: str, pred: str) -> float:
        """Returns 1 if pred is not in ref, and 0 otherwise"""
        clean_ref = StringPreviousRoundsEvaluator.clean_answer(ref)
        clean_pred = StringPreviousRoundsEvaluator.clean_answer(pred)
        # tokenize the answer if the ref is a single word
        # prevent false positive (e.g, 0)
        if len(word_tokenize(clean_ref)) == 1:
            tok_pred = word_tokenize(clean_pred)
            return float(clean_ref not in tok_pred)
        else:
            return float(clean_ref not in clean_pred)

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    @staticmethod
    @beartype
    def ua_match(ref: str, pred: str, intent: str) -> float:
        return llm_ua_match(pred, ref, intent)

    def _evaluate_normal_mode(self, pred: str, configs: dict) -> float:
        """Normal evaluation mode using multiplicative scoring (strict)"""
        score = 1.0

        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)

                case "required_values":
                    required_values = value
                    assert isinstance(required_values, list)
                    pred_int = NumericPreviousRoundsEvaluator.str_2_int(pred)
                    if pred_int is None:
                        score = 0.0
                    else:
                        for v in required_values:
                            value_or = v.split(" |OR| ")
                            score *= any(
                                [
                                    NumericPreviousRoundsEvaluator.compare_inequality(
                                        pred_int, value
                                    )
                                    for value in value_or
                                ]
                            )

                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        value_or = must_value.split(" |OR| ")
                        score *= any(
                            [self.must_include(ref=v, pred=pred) for v in value_or]
                        )

                case "must_exclude":
                    assert isinstance(value, list)
                    for must_excl_value in value:
                        score *= self.must_exclude(ref=must_excl_value, pred=pred)

                case "one_of":
                    assert isinstance(value, list)
                    found = False
                    for one_of_value in value:
                        one_of_value = self.clean_answer(one_of_value)
                        if one_of_value in pred:
                            found = True
                            break
                    score = score * found

                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        # Try exact match first
                        exact_score = self.exact_match(ref=value, pred=pred)
                        if exact_score == 1.0:
                            score *= exact_score
                        else:
                            # Try unachievable reason match
                            ua_score = self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                            score *= ua_score
                    else:
                        assert isinstance(value, list)
                        reference = ", ".join(value)
                        fuzzy_score = self.fuzzy_match(
                            ref=reference, pred=pred, intent=intent
                        )
                        score *= fuzzy_score

        return score

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page | None = None,
        evaluation_mode: str = "normal",
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(previous_rounds)

        def evaluate_action(action_str: str) -> float:
            # Try multiple answer sources for maximum coverage
            answer_sources = []

            action_information = get_action_information_from_action_str(action_str)
            action_obj = action_information["action"]
            answer_sources.append(action_obj["answer"])
            answer_sources.append(action_information["comment"])
            answer_sources.append(action_str)

            # Try each answer source and take the best score
            best_score = 0.0

            for pred_raw in answer_sources:
                pred = self.clean_answer(pred_raw)

                if len(pred) == 0 or pred is None:
                    continue

                score = self._evaluate_normal_mode(pred, configs)

                best_score = max(best_score, score)

            return best_score

        # Try it for the chosen exit action as well as all the other exit actions in the vote metadata
        last_action_strs_to_evaluate = [last_action.get("action_str", "")]
        any_of_them_got_a_non_zero_score = any(
            evaluate_action(action_str) > 0.0
            for action_str in last_action_strs_to_evaluate
        )
        if any_of_them_got_a_non_zero_score:
            return 1.0
        else:
            return 0.0


@beartype
class StringSoftPreviousRoundsEvaluator(PreviousRoundsEvaluator):
    """Use text generation metrics such as BLEU, ROUGE, etc. to evaluate the answer"""

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(previous_rounds)

        def evaluate_action(action_str: str) -> float:
            action_obj = get_action_information_from_action_str(action_str)["action"]
            pred = action_obj["answer"]

            ref = configs["eval"]["reference_answers"]
            # rouge
            m = evaluate.load("rouge")
            rouge = m.compute(predictions=[pred], references=[ref])
            if not rouge or "rouge1" not in rouge:
                return 0.0
            return float(rouge.get("rouge1", 0.0))

        # Try it for the chosen exit action as well as all the other exit actions in the vote metadata
        last_action_strs_to_evaluate = [last_action.get("action_str", "")]

        # If any of them score > 0.0, then return 1.0 - if all failed then return 0.0
        overall_score = 0.0
        for action_str in last_action_strs_to_evaluate:
            curr_score = evaluate_action(action_str)
            if curr_score > 0.0:
                overall_score = 1.0
                break
        return overall_score


@beartype
class URLExactPreviousRoundsEvaluator(PreviousRoundsEvaluator):
    """Check whether the URL is exactly the same as of the reference URLs"""

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        def clean_url(url: str) -> str:
            url = str(url)
            # Replace http://localhost with http://127.0.0.1 to keep things consistent across evals.
            url = url.replace("localhost", "127.0.0.1")
            if url.endswith("/"):
                url = url[:-1]
            return url

        # Check for all previous rounds
        def check_url(url: str) -> float:
            pred = clean_url(url)
            ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
            ref_urls = [clean_url(url) for url in ref_urls]
            matching_rule = configs["eval"].get("url_note", "EXACT")
            if matching_rule == "EXACT":
                if pred in ref_urls:
                    return 1.0
                else:
                    return 0.0
            elif matching_rule == "GOLD in PRED":
                if any([ref in pred for ref in ref_urls]):
                    return 1.0
                else:
                    return 0.0
            else:
                raise ValueError(f"Unknown matching rule: {matching_rule}")

        urls_to_check = [page.url]

        # If any of them score 1.0, then return 1.0 - if all failed then return 0.0
        return 1.0 if any(check_url(url) == 1.0 for url in urls_to_check) else 0.0


@beartype
class HTMLContentExactPreviousRoundsEvaluator(PreviousRoundsEvaluator):
    """Check whether the contents appear in the page"""

    @staticmethod
    @beartype
    def fuzzy_match(ref: str, pred: str, intent: str) -> float:
        return llm_fuzzy_match(pred, ref, intent)

    def _evaluate_required_contents_normal_mode(
        self, selected: Any, target: dict, configs: dict
    ) -> bool:
        """Normal evaluation mode using multiplicative scoring (strict)"""
        if selected is None:
            return False

        # Normalize to string where appropriate for string-based checks
        selected_str = selected if isinstance(selected, str) else str(selected)

        req = target["required_contents"]
        score = 1.0

        if "exact_match" in req:
            required_contents = req["exact_match"]
            score *= StringPreviousRoundsEvaluator.exact_match(
                ref=required_contents, pred=selected_str
            )
        elif "must_include" in req:
            required_contents = req["must_include"]
            assert isinstance(required_contents, list)
            for content in required_contents:
                content_or = content.split(" |OR| ")
                score *= any(
                    [
                        StringPreviousRoundsEvaluator.must_include(
                            ref=content_candidate, pred=selected_str
                        )
                        for content_candidate in content_or
                    ]
                )
        elif "must_exclude" in req:
            required_contents = req["must_exclude"]
            assert isinstance(required_contents, list)
            for content in required_contents:
                assert " |OR| " not in content
                score *= StringPreviousRoundsEvaluator.must_exclude(
                    content, pred=selected_str
                )
        elif "required_values" in req:
            required_values = req["required_values"]
            assert isinstance(required_values, list)
            value_to_check: Optional[int]
            if isinstance(selected, str):
                value_to_check = NumericPreviousRoundsEvaluator.str_2_int(selected)
            else:
                value_to_check = selected  # type: ignore[assignment]
            if value_to_check is None:
                score = 0.0
            else:
                for value in required_values:
                    value_or = value.split(" |OR| ")
                    score *= any(
                        [
                            NumericPreviousRoundsEvaluator.compare_inequality(
                                value_to_check, value_candidate
                            )
                            for value_candidate in value_or
                        ]
                    )
        elif "fuzzy_match" in req:
            required_contents = req["fuzzy_match"]
            intent = configs["intent"]
            assert isinstance(required_contents, list)
            reference = ", ".join(required_contents)
            score *= self.fuzzy_match(ref=reference, pred=selected_str, intent=intent)
        elif "one_of" in req:
            required_contents = req["one_of"]
            assert isinstance(required_contents, list)
            found = False
            for content in required_contents:
                if StringPreviousRoundsEvaluator.exact_match(
                    ref=content, pred=selected_str
                ):
                    found = True
                    break
            score *= float(found)
        else:
            raise ValueError(f"Unknown required_contents: {req.keys()}")

        return bool(score)

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page,
        evaluation_mode: str = "normal",
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        targets = configs["eval"]["program_html"]

        overall_pass = True
        for target in targets:
            # Resolve candidate URLs to try: configured target, all trajectory URLs, and the current page URL
            target_url: str = target["url"]
            candidate_urls: list[str] = []
            try:
                if target_url.startswith("func"):
                    func = target_url.split("func:")[1]
                    func = func.replace("__last_url__", page.url)
                    resolved = eval(func)
                    if isinstance(resolved, str) and resolved:
                        candidate_urls.append(resolved)
                elif target_url == "last":
                    candidate_urls.append(page.url)
                else:
                    candidate_urls.append(target_url)
            except Exception:
                # If resolving fails, skip adding the configured url
                pass

            # Deduplicate while preserving order
            candidate_urls = list(
                dict.fromkeys([u for u in candidate_urls if isinstance(u, str) and u])
            )

            locator: str = target["locator"]

            def evaluate_required_contents(selected: Any) -> bool:
                # Use the appropriate evaluation mode
                if evaluation_mode == "normal":
                    return self._evaluate_required_contents_normal_mode(
                        selected, target, configs
                    )
                else:  # default to relaxed mode
                    return self._evaluate_required_contents_relaxed_mode(
                        selected, target, configs
                    )

            # Try combinations of candidate URLs and two locator strategies: configured locator and full document outerText
            target_passed = False
            for url in candidate_urls:
                try:
                    page.goto(url)
                    time.sleep(3)  # TODO: remove hard-coded sleep
                except Exception:
                    # If navigation fails, try next URL
                    continue

                selected_element: Any = None
                # First, try the configured locator
                try:
                    if not locator.strip():
                        selected_element = page.content()
                    elif locator.startswith("document.") or locator.startswith(
                        "[...document."
                    ):
                        if "prep_actions" in target:
                            try:
                                for prep_action in target["prep_actions"]:
                                    page.evaluate(f"() => {prep_action}")
                            except Exception:
                                pass
                        try:
                            selected_element = (
                                str(page.evaluate(f"() => {locator}")) or ""
                            )
                        except Exception:
                            selected_element = self._get_fallback_element(page)
                    elif locator.startswith("lambda:"):
                        try:
                            locator_fn = locator.lstrip("lambda:")
                            selected_element = page.evaluate(locator_fn)
                            if not selected_element:
                                selected_element = self._get_fallback_element(page)
                        except Exception:
                            selected_element = self._get_fallback_element(page)
                    elif locator.startswith("func:"):
                        func = locator.split("func:")[1]
                        func = func.replace("__page__", "page")
                        selected_element = eval(func)
                    else:
                        raise ValueError(f"Unknown locator: {locator}")
                except Exception:
                    selected_element = self._get_fallback_element(page)

                # If either strategy satisfies the requirement, the target passes
                selected_element_passed = evaluate_required_contents(selected_element)
                if selected_element_passed:
                    target_passed = True
                    break

            if not target_passed:
                overall_pass = False
                break

        return 1.0 if overall_pass else 0.0

    def _get_fallback_element(self, page: Page) -> str:
        return ""


@beartype
class PageImagePreviousRoundsEvaluator(PreviousRoundsEvaluator):
    """Check whether the answer is correct by querying a vision model."""

    def __init__(self, captioning_fn):
        self.captioning_fn = captioning_fn
        # Default to 0.8 as the threshold for similarity to account for compression, resizing, etc
        # This might be too generous but we bias towards minimizing false negatives.
        self.ssim_threshold = 0.8

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page | None = None,
    ) -> float:
        if page is None:
            return 0.0
        with open(config_file, "r") as f:
            configs = json.load(f)

        for query in configs["eval"]["page_image_query"]:
            locator: str = query["eval_image_class"]
            target_url: str = query["eval_image_url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", getattr(page, "url", ""))
                target_url = eval(func)

            # navigate to that url
            if target_url != "last" and isinstance(target_url, str) and target_url:
                page.goto(target_url)
                time.sleep(3)  # TODO(jykoh): fix this hard-coded sleep

            # empty, use the full page
            if not locator.strip():
                images = page.get_by_role("img").all()
            # use JS to select the element
            elif locator.startswith("."):
                # Get all img children under the locator
                elements = page.query_selector_all(locator)
                images = []
                for element in elements:
                    is_img = element.evaluate('element => element.tagName === "IMG"')
                    if is_img:
                        images.append(element)
                    else:
                        images.extend(element.query_selector_all("img"))
            else:
                raise ValueError(f"Unknown locator: {locator}")

            if images == []:
                return 0.0

            all_image_pixels = []
            for image in images:
                try:
                    # Get image from URL.
                    image_url = image.get_attribute("src")
                    if not isinstance(image_url, str) or not image_url:
                        continue
                    if not image_url.startswith(("http://", "https://", "www.")):
                        image_url = urljoin(getattr(page, "url", ""), image_url)
                    resp = requests.get(image_url, stream=True)
                    img = Image.open(resp.raw)
                    all_image_pixels.append(img)
                except Exception as e:
                    print("[WARNING]: ", e)

            score = 1.0
            if all_image_pixels == []:
                return 0.0
            else:
                # Run the VQA eval on the image elements.
                eval_vqas = query.get("eval_vqa", [])
                assert (
                    len(eval_vqas) > 0 or "eval_fuzzy_image_match" in query
                ), "eval_vqa must have at least 2 questions or eval_fuzzy_image_match must be True"
                for qa in eval_vqas:
                    question, answer = qa["question"], qa["answer"]
                    prompt = f"Q: {question} A:"
                    pred_ans = self.captioning_fn(
                        all_image_pixels, [prompt] * len(all_image_pixels)
                    )
                    score *= float(
                        any([answer.lower() in ans.lower() for ans in pred_ans])
                    )

                if "eval_fuzzy_image_match" in query:
                    ssim_threshold = query.get("ssim_threshold", self.ssim_threshold)
                    exact_match_imgs = query["eval_fuzzy_image_match"].split(" |OR| ")
                    all_exact_match_pixels = []

                    for exact_match_img in exact_match_imgs:
                        if exact_match_img.startswith("http"):
                            exact_match_pixels = Image.open(
                                requests.get(exact_match_img, stream=True).raw
                            )
                        else:
                            exact_match_pixels = Image.open(exact_match_img)
                        all_exact_match_pixels.append(exact_match_pixels)

                    # Check if any of the images on the page match
                    found_exact_match = False
                    for exact_match_pixels in all_exact_match_pixels:
                        for image_pixels in all_image_pixels:
                            ssim = image_utils.get_image_ssim(
                                image_pixels, exact_match_pixels
                            )
                            if ssim > ssim_threshold:
                                found_exact_match = True
                                break
                    score *= float(found_exact_match)

        return score


class PreviousRoundsEvaluatorComb:
    def __init__(self, evaluators: list[PreviousRoundsEvaluator]) -> None:
        self.evaluators = evaluators

    def __call__(
        self,
        previous_rounds: List[ActInferencePreviousRound],
        config_file: Path | str,
        page: Page,
    ) -> float:
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator(previous_rounds, config_file, page)
            score *= cur_score

        return score


@beartype
def previous_rounds_evaluator_router(
    config_file: Path | str, captioning_fn=None
) -> PreviousRoundsEvaluatorComb:
    """Router to get the evaluator class that uses previous_rounds"""
    with open(config_file, "r") as f:
        configs = json.load(f)

    eval_types = configs["eval"]["eval_types"]
    evaluators: list[PreviousRoundsEvaluator] = []
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringPreviousRoundsEvaluator())
            case "url_match":
                evaluators.append(URLExactPreviousRoundsEvaluator())
            case "program_html":
                evaluators.append(HTMLContentExactPreviousRoundsEvaluator())
            case "page_image_query":
                evaluators.append(PageImagePreviousRoundsEvaluator(captioning_fn))
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return PreviousRoundsEvaluatorComb(evaluators)


@beartype
def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


@beartype
def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(f"{SHOPPING}/rest/V1/orders", params=params, headers=header)
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


@beartype
def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


@beartype
def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


@beartype
def shopping_get_sku_latest_review_text(sku: str) -> str:
    """Get the latest review text for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    text: str = response_obj[-1]["detail"]
    return text


@beartype
def shopping_get_sku_latest_review_title(sku: str) -> str:
    """Get the latest review title for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    title: str = response_obj[-1]["title"]
    return title


@beartype
def shopping_get_sku_product_page_url(sku: str) -> str:
    """Get product page url from sku"""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{SHOPPING}/rest/V1/products/{sku}", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    for custom_attributes in response_obj["custom_attributes"]:
        if custom_attributes["attribute_code"] == "url_key":
            return f"{SHOPPING}/{custom_attributes['value']}.html"
    return ""


@beartype
def shopping_get_all_product_order(
    page: Page,
) -> List[Dict[str, str]]:
    """
    Get info of all product in a given order page.

    Example output:
    [
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Strawberry, 19oz Box (12 Bars)\nSize\n12 Count (Pack of 1)",
            "options": {
                "Size": "12 Count (Pack of 1)"
            },
            "sku": "B00MXUFL0E",
            "price": "$24.50",
            "qty": "Ordered2",
            "subtotal": "$49.00"
        },
        {
            "name": "Kellogg's Special K Protein Bars, Meal Replacement, Protein Snacks, Value Size, Chocolatey Chip Cookie Dough, 19oz Box (12 Bars)",
            "sku": "B07ZD2PB9F",
            "price": "$42.30",
            "qty": "Ordered2",
            "subtotal": "$84.60"
        }
    ]
    """
    try:
        result = page.evaluate(
            f"""
(() => {{
    try {{
        const products = [...document.querySelector("#my-orders-table").getElementsByTagName('tbody')].map(
            (x) => {{
                return [...x.getElementsByTagName('td')].reduce(function(obj, y) {{
                    const key = y.className.split(' ')[1];
                    obj[key] = y.outerText;
                    // check if options exist
                    if (key === 'name' && y.querySelector('dl')) {{
                        var option_dict = {{}}
                        const options = [...y.querySelector('dl').children];
                        for (let i = 0; i < options.length; i += 2) {{
                            option_dict[options[i].outerText] = options[i+1].outerText;
                        }}
                        obj['options'] = option_dict;
                    }}
                    return obj;
                }}, {{}})
            }}
        );
        return products;
    }} catch (e) {{
        // If any errors are caught, return an empty string
        return e;
        return [];
    }}
}})();
            """
        )
        return result
    except Exception as e:
        result = []

    return result


@beartype
def shopping_get_order_product_name_list(page: Page) -> str:
    try:
        products = shopping_get_all_product_order(page)

        return " |OR| ".join([p["name"] for p in products])
    except Exception:
        return ""


@beartype
def shopping_get_order_product_quantity(page: Page, sku: str) -> int:
    try:
        if "|OR|" in sku:
            skus = sku.split(" |OR| ")
        else:
            skus = [sku]

        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() in skus:
                # Ordered{qty}
                return int(product["qty"][7:])
        return 0
    except Exception:
        return 0


@beartype
def shopping_get_order_product_option(page: Page, sku: str, option_name: str) -> str:
    try:
        products = shopping_get_all_product_order(page)
        for product in products:
            if product["sku"].strip() == sku:
                # Ordered{qty}
                options = product.get("options", {})
                if isinstance(options, dict):
                    return str(options.get(option_name, ""))
                return ""
        return ""
    except Exception as e:
        return ""


@beartype
def shopping_get_product_attributes(page: Page, attribute: str) -> str:
    # Get the values of all cells in the table for the given attribute
    try:
        result = page.evaluate(
            f"""
                (() => {{
                try {{
                    // Create an array of search terms, splitting the string by ' |OR| '
                    const searchTerms = '{attribute}'.toLowerCase().split(' |or| ');
                    // Convert the children of the tbody inside the element with the given ID into an array
                    return Array.from(
                    document.querySelector('#productDetails_detailBullets_sections1 > tbody').children
                    )
                    // Filter the array to only include elements where the first child's text includes any of the search terms
                    .filter(x =>
                    searchTerms.some(term => x.children[0].outerText.toLowerCase().includes(term))
                    )
                    // Map over the filtered elements to get the outerText of their second child
                    .map(x => x.children[1].outerText)
                    // Join all the resulting strings with a comma and a space
                    .join(', ')
                }} catch (e) {{
                    // If any errors are caught, return an empty string
                    return ''
                }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


@beartype
def shopping_get_product_price(page: Page) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseFloat(document.querySelector(\"#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span\")
                    .outerText.substr(1));
                    return res ? res : 0;
                }})();
            """
        )
    except Exception:
        result = 0

    return result


@beartype
def shopping_get_num_reviews(page: Page) -> int:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {{
                    res = parseInt(document.querySelector(\"#tab-label-reviews-title\")
                    .outerText.split(' ')[1]);
                    return res ? res : 0; }}
                )();
            """
        )
    except Exception:
        result = 0

    return result


@beartype
def shopping_get_rating_as_percentage(page: Page) -> int:
    """Get the rating of the product on the shopping website as a percentage out of 100."""
    try:
        rating = page.evaluate(
            """
                (() => {{
                    ratingPercentage = parseFloat(document.querySelector('.rating-result').title.replace('%', ''));
                    return ratingPercentage ? ratingPercentage : 0;
                }})();
            """
        )
    except Exception:
        rating = 0

    return rating


@beartype
def get_query_text(page: Page, selector: str) -> str:
    """Get the text content of the element matching the given selector.

    Note that this function DOES NOT perform downcasing.
    """
    try:
        result = page.evaluate(
            f"""
                (() => {{
                    try {{
                        return document.querySelector('{selector}').textContent;
                    }} catch (e) {{
                        return '';
                    }}
                }})();
            """
        )
    except Exception:
        result = ""

    return result


@beartype
def get_query_text_lowercase(page: Page, selector: str) -> str:
    """Get the lowercase text content of the element matching the given selector."""
    return get_query_text(page, selector).lower()


@beartype
def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


@beartype
def reddit_get_post_comment_tree(page: Page) -> Dict[str, Any]:
    try:
        comment_tree = page.evaluate(
            f"""(function buildCommentTree(node, data_level) {{
    let tree = {{
        "username": node.querySelector(".fg-inherit").outerText,
        "net_score": parseInt(node.querySelector(".vote__net-score").outerText),
        "content": node.querySelector(".comment__content").outerText,
        "time": new Date(node.querySelector('.comment__main > header > h1 > span > time').dateTime),
        "children": []
    }};
    node.querySelectorAll(".comment").forEach((child) => {{
        if (parseInt(child.getAttribute('data-level')) === data_level+1) {{
            tree['children'].push(buildCommentTree(child, data_level+1));
        }}
    }})

    return tree;
}})(document.querySelector("#main"), 0)"""
        )
    except Exception:
        comment_tree = {}

    return comment_tree


@beartype
def reddit_get_latest_comment_obj_by_username(
    page: Page, username: str
) -> Dict[str, Any]:
    try:
        comment_tree = reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            if node["username"] == username:
                if node["time"] > latest_time:
                    comment = {
                        "username": node["username"],
                        "net_score": node["net_score"],
                        "content": node["content"],
                        "time": node["time"],
                    }
                    latest_time = node["time"]

            for child in node["children"]:
                dfs(child)

        dfs(comment_tree)

    except Exception as e:
        comment = {}
    return comment


@beartype
def reddit_get_latest_comment_content_by_username(page: Page, username: str) -> str:
    try:
        comment = reddit_get_latest_comment_obj_by_username(page, username)
        content = comment["content"]

    except Exception:
        content = ""

    return content


@beartype
def reddit_get_parent_comment_obj_of_latest_comment_by_username(
    page: Page, username: str
) -> Dict[str, Any]:
    try:
        comment_tree = reddit_get_post_comment_tree(page)
        latest_time = datetime.min.replace(tzinfo=timezone.utc)
        comment = {}

        def dfs(node):
            nonlocal latest_time
            nonlocal comment
            for child in node["children"]:
                if child["username"] == username:
                    if child["time"] > latest_time:
                        comment = {
                            "username": node["username"],
                            "net_score": node["net_score"],
                            "content": node["content"],
                            "time": node["time"],
                        }
                        latest_time = child["time"]
                else:
                    dfs(child)

        dfs(comment_tree)

    except Exception:
        comment = {}
    return comment


@beartype
def reddit_get_parent_comment_username_of_latest_comment_by_username(
    page: Page, username: str
) -> str:
    try:
        comment = reddit_get_parent_comment_obj_of_latest_comment_by_username(
            page, username
        )
        username = comment["username"]

    except Exception:
        username = ""

    return username


@beartype
def gitlab_get_project_memeber_role(page: Page, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role


@beartype
def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = "Help a teacher to grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by 'correct', 'incorrect', or 'partially correct'. Only output one of these options, and nothing else."
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    logger.info(f"[R] {reference}")
    logger.info(f"[P] {pred}")

    response = generate_from_openai_chat_completion(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=1.0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()

    logger.info(f"[LLM Response] {response}")

    if "incorrect" in response:
        return 0.0
    else:
        # Be lenient: treat 'correct' or legacy 'partially correct' as pass
        assert ("correct" in response) or ("partially correct" in response), response
        return 1.0


def llm_ua_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-4-turbo"""
    messages: list[dict[str, Any]] = []
    # construct the question to ask
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under 'actual unachievable reason'. "
        "An individual previously attempted this task and was unable to complete it. They provided a reason for their failure, "
        "which is listed under 'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. Otherwise, respond with 'different'."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]

    response = generate_from_openai_chat_completion(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=1.0,
        max_tokens=768,
        top_p=1.0,
        context_length=0,
    ).lower()
    if "different" in response:
        return 0.0
    else:
        assert "same" in response
        return 1.0
