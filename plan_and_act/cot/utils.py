import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import AbstractSet, Any, Coroutine, Iterable, TypeVar, cast

from browser_env.actions import ActionTypes, create_webrl_id_based_action
from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from plan_and_act.cot.models import (
    ActionStep,
    LlamaFactoryData,
    PlanAndActAction,
    ProcessedData,
    ServerMessage,
    TorchTuneData,
    WebArenaLiteAction,
    WebArenaLiteWebsite,
)
from plan_and_act.html_cleaner import HTMLCleaner


def preprocess_webarena_data(data: list[WebArenaLiteAction]) -> list[ProcessedData]:
    """
    This function preprocesses the WebArena Lite data to groups actions by task.
    It also extracts the task instructions and the initial html state from the data for convenient access. Run this function once and store the result for efficient access.
    """
    already_processed_tasks = set()

    # 1. Extract the task instructions and also label each action with its position within the task
    for d in cast(list[WebArenaLiteAction], data):
        conversations = d["conversations"]
        human_prompt = conversations[0]
        task_instruction_match = re.search(
            r"Task Instruction: (.*)\s*Round 0", human_prompt["value"], re.DOTALL
        )
        if task_instruction_match is None:
            raise ValueError(
                f"Could not find task instruction in {human_prompt['value']}"
            )

        task_instruction = task_instruction_match.group(1)
        d["task"] = task_instruction.strip()

        if d["task"] not in already_processed_tasks:
            already_processed_tasks.add(d["task"])

        round_matches = re.findall(r"Round (\d+)", human_prompt["value"])
        if len(round_matches) == 0:
            raise ValueError(f"Could not find round in {human_prompt['value']}")

        last_round_number = int(round_matches[-1])
        d["index_within_task"] = last_round_number

    # 2. Group the actions by task
    task_data = {}
    for d in data:
        task_id = d["task"]
        if task_id not in task_data:
            task_data[task_id] = {
                "task": d["task"],
                "actions": [],
            }
        task_data[task_id]["actions"].append(d)

        # Also add the initial html state by getting the html at teh "index_within_task" = 0
        if d["index_within_task"] == 0:
            html = get_html_from_action(d)
            task_data[task_id]["initial_html_state"] = html

    return list(task_data.values())


def get_html_from_action(action: WebArenaLiteAction) -> str:
    index = action["index_within_task"]
    while index >= 0:  # Ensure we don't go into an infinite loop
        try:
            html = (
                action["conversations"][0]["value"]
                .split(
                    f"Round {index}\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>"
                )[1]
                .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0]
                .strip()
            )
            return html
        except IndexError:
            print(
                f"IndexError: No 'Round {index}' found. Decrementing index and retrying..."
            )
            index -= 1  # Decrement the index and try again

    raise ValueError(f"Unable to find a valid 'Round' for any index within the task.")


def get_raw_action_string_from_action(action: WebArenaLiteAction) -> str:
    """
    Returns the raw action string from the action.
    """
    return action["conversations"][1]["value"]


def get_action_information(action: WebArenaLiteAction) -> PlanAndActAction:
    """
    A helper function that gives you the action object from the original WebArenaLite codebase as well as some useful information for Plan-and-Act project.
    """
    return get_action_information_from_action_str(
        get_raw_action_string_from_action(action)
    )


def get_action_information_from_action_str(action_str: str) -> PlanAndActAction:
    """
    A helper function that gives you the action object from the original WebArenaLite codebase as well as some useful information for Plan-and-Act project.
    """
    # 1) Get the action object
    action_obj = create_webrl_id_based_action(action_str)

    # 2) Get the any kind of comment from the action
    comments = []
    lines = action_str.split("\n")

    for line in lines:
        if line.strip().startswith("#"):
            comments.append(line)  # Collect comment lines

    return PlanAndActAction(
        action=action_obj,
        comment="\n".join(comments),
    )


def clean_html(
    html: str,
    prettify: bool = False,
    add_the_destructive_visitor: bool = False,
    strip_annotation: bool = False,
    dont_remove_ids: AbstractSet[str] = set(),
) -> str:
    """
    Utility function to clean the html.

    Arguments:
        - prettify: Whether to prettify the html at the end.
        - add_the_destructive_visitor: Whether to add the destructive visitor to the HTMLCleaner. This basically shrinks the HTML consdierably by removing a lot of the non-interactive elements. Since this type of cleaning removes a lot of the structure of the HTML, it is recommended to only use this when the HTML is too big it doesn't fit into the model's context length.
        - strip_annotation: Whether to strip the annotation from the html.
        - dont_remove_ids: A set of ids that should not be removed from the html.
    """
    return HTMLCleaner(
        html=html, id_name="id", dont_remove_ids=dont_remove_ids
    ).clean_html(
        prettify=prettify,
        add_the_destructive_visitor=add_the_destructive_visitor,
        strip_annotation=strip_annotation,
    )


def prepare_completions_prompt_for_reasoning_models(
    messages: list[ServerMessage],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
) -> str:
    """
    The official DeepSeek Usage Recommendations guidelines recommend that:
        - We don't use any system messages.
        - We use a complations endpoint instead of a chat complations endpoints and also start the new assistant message with the `<think>\n` tag in order force the model to generate CoT first.
    """
    # Check whether there any system messages
    assert not any(
        message["role"] == "system" for message in messages
    ), "System messages are not allowed."

    prompt = tokenizer.apply_chat_template(
        messages,  # type: ignore
        tokenize=False,
        add_generation_prompt=True,
    )

    assert isinstance(prompt, str), f"The prompt is not a string: {prompt}"

    # If the prompt is a string and does not end with the `<think>\n` tag, add it.
    if not prompt.endswith("<think>\n"):
        prompt = f"{prompt}<think>\n"

    return prompt


# Constants: maximum allowed descendant nodes and maximum recursion depth.
MAX_TOTAL_CHILDREN = (
    5  # Total descendant nodes allowed (not counting the sibling itself)
)
MAX_DEPTH = 3  # Maximum depth to traverse


def clone_children_with_limit(
    children: Iterable[PageElement],
    max_total: int,
    max_depth: int,
    current_depth: int,
    count: int,
) -> tuple[list[PageElement], int, bool]:
    """
    Recursively clone a sequence of nodes (children) while ensuring that:
      - The total number of nodes (across all levels) does not exceed max_total.
      - The recursion does not go deeper than max_depth.

    Args:
        children (iterable): The children nodes to clone.
        max_total (int): Maximum total number of nodes allowed.
        max_depth (int): Maximum recursion depth.
        current_depth (int): Current recursion depth.
        count (int): Number of nodes cloned so far.

    Returns:
        A tuple (cloned_children, updated_count, truncated) where:
          - cloned_children is a list of cloned nodes.
          - updated_count is the new count after cloning.
          - truncated is True if the cloning was cut off due to limits.
    """
    cloned_children = []
    truncated = False

    for child in children:
        # If we've reached our limit, break and mark truncation.
        if count >= max_total:
            truncated = True
            break

        if isinstance(child, NavigableString):
            # For a text node, you don't count it as a node.
            cloned_children.append(child)

        elif isinstance(child, Tag):
            # Create a new tag with the same name and attributes.
            new_child = BeautifulSoup("", "html.parser").new_tag(
                child.name, **child.attrs
            )
            count += 1  # Count the tag node itself

            if current_depth < max_depth:
                # Recurse into children if we haven't reached the depth limit.
                child_clones, count, child_truncated = clone_children_with_limit(
                    child.children, max_total, max_depth, current_depth + 1, count
                )
                for clone in child_clones:
                    new_child.append(clone)
                if child_truncated:
                    # Indicate that further content was truncated.
                    new_child.append(NavigableString("... truncated ..."))
            else:
                # At max depth, do not include further descendants.
                if list(child.children):
                    new_child.append(
                        NavigableString("... truncated due to depth limit ...")
                    )

            cloned_children.append(new_child)

    return cloned_children, count, truncated


def truncate_element(
    element: Tag, max_total: int = MAX_TOTAL_CHILDREN, max_depth: int = MAX_DEPTH
) -> Tag:
    """
    Clones the provided element tag while limiting the total number of descendant nodes
    (at any depth) and the recursion depth. When the limits are reached, a truncated
    comment is inserted.

    Args:
        element (Tag): The element to clone and truncate.
        max_total (int): The maximum number of descendant nodes to include.
        max_depth (int): The maximum recursion depth.

    Returns:
        Tag: A clone of the element with its children truncated according to the limits.
    """
    # Clone the element tag (without its children yet)
    new_element = BeautifulSoup("", "html.parser").new_tag(
        element.name, **element.attrs
    )
    count = (
        0  # Start counting from 0 for the descendants (not counting the element itself)
    )

    cloned_children, _, truncated = clone_children_with_limit(
        element.children, max_total, max_depth, current_depth=0, count=count
    )
    for child in cloned_children:
        new_element.append(child)

    if truncated:
        # Append a truncated indicator if we did not clone all content.
        new_element.append(NavigableString("... truncated ..."))

    return new_element


# Alias for backward compatibility
truncate_sibling = truncate_element


def get_html_snippet_str_from_element(
    html: str,
    target_element_id: str,
    num_siblings: int = 3,
    strip_annotation: bool = True,
) -> str:
    """
    Base function that handles the HTML cleaning logic given raw HTML and a target element ID.

    Args:
        html (str): Raw HTML string
        target_element_id (str): ID of the target element
        num_siblings (int): Number of siblings to ensure are captured
        strip_annotation (bool): Whether to strip annotations

    Returns:
        str: Cleaned HTML
    """
    # 1) Build the target_element CSS selector
    target_element_selector = f"[id='{target_element_id}']"

    # 2) Parse HTML and find target element
    soup = BeautifulSoup(html, "html.parser")
    target_element = soup.select_one(target_element_selector)

    # If we cannot find the target element, raise error
    if target_element is None:
        raise ValueError("Cannot find the target element in the HTML.")

    # 3-9) Shared logic for processing the HTML
    cleaned_html = _process_html_with_target(
        target_element=target_element,
        target_element_id=target_element_id,
        num_siblings=num_siblings,
        strip_annotation=strip_annotation,
    )

    # Then beautify it again
    return BeautifulSoup(cleaned_html, "html.parser").prettify()


def get_html_snippet_str(
    action: WebArenaLiteAction,
    num_siblings: int = 3,
    strip_annotation: bool = True,
) -> str:
    """Original function that works with WebArenaLiteAction"""
    html = get_html_from_action(action)
    target_element_id = get_action_information(action)["action"]["element_id"]

    if len(target_element_id) < 0:
        raise ValueError(
            "Cannot find the target element id in the action. This function can only be used on actions that actually interact with an element."
        )

    return get_html_snippet_str_from_element(
        html=html,
        target_element_id=target_element_id,
        num_siblings=num_siblings,
        strip_annotation=strip_annotation,
    )


def _process_html_with_target(
    target_element: Tag,
    target_element_id: str,
    num_siblings: int,
    strip_annotation: bool,
) -> str:
    """
    Shared logic for processing HTML with a target element.
    """
    # 3) Collect real parents (stop before <body>)
    current = target_element.parent
    parent_chain = []
    while current and current.name != "body":
        parent_chain.append(current)
        current = current.parent
    parent_chain.reverse()

    # 4) Collect siblings
    sibling_parent = target_element.parent
    siblings_before = []
    siblings_after = []
    remaining_siblings_needed = num_siblings

    while sibling_parent and remaining_siblings_needed > 0:
        # Get all valid siblings
        all_siblings = [
            sibling
            for sibling in sibling_parent.children
            if (
                isinstance(sibling, Tag)
                and sibling.name
                and sibling is not target_element
            )
        ]

        # Split siblings into before and after based on target element position
        try:
            target_idx = list(sibling_parent.children).index(target_element)
            before = [
                s
                for s in all_siblings
                if list(sibling_parent.children).index(s) < target_idx
            ]
            after = [
                s
                for s in all_siblings
                if list(sibling_parent.children).index(s) > target_idx
            ]

            # Take up to half remaining siblings from each side
            siblings_per_side = remaining_siblings_needed // 2
            if before:
                siblings_before.extend(before[-siblings_per_side:])
            if after:
                siblings_after.extend(after[:siblings_per_side])
            remaining_siblings_needed -= len(before[-siblings_per_side:]) + len(
                after[:siblings_per_side]
            )
        except ValueError:
            # Target element not found at this level, just add all siblings
            siblings_to_add = all_siblings[:remaining_siblings_needed]
            siblings_after.extend(siblings_to_add)
            remaining_siblings_needed -= len(siblings_to_add)

        sibling_parent = sibling_parent.parent

    # 5) Build new soup
    cleaned_soup = BeautifulSoup("<body data-nrd='body-id'></body>", "html.parser")
    current_container = cleaned_soup.body
    assert current_container is not None

    # 6) Rebuild parent chain
    for parent in parent_chain:
        new_parent = cleaned_soup.new_tag(parent.name)
        for attr, value in parent.attrs.items():
            new_parent[attr] = value
        current_container.append(new_parent)
        current_container = new_parent

    # 7) Add siblings in order - before target element position
    for sibling in siblings_before:
        current_container.append(truncate_element(sibling))

    # 8) Add target element
    current_container.append(truncate_element(target_element))

    # 9) Add siblings after target element position
    for sibling in siblings_after:
        current_container.append(truncate_element(sibling))

    # Clean it after
    cleaned_html = clean_html(
        html=str(cleaned_soup),
        prettify=True,
        strip_annotation=strip_annotation,
        dont_remove_ids={target_element_id},
    )

    # Handle empty result
    if len(cleaned_html.strip()) == 0:
        cleaned_html = get_minimum_html_snippet_str(
            target_element,
            target_element_id,
        )

    return cleaned_html


def get_minimum_html_snippet_str(
    target_element: Tag, element_id: str, strip_annotation: bool = True
) -> str:
    """
    If for some reason, we cannot find the target element from the HTML ids, then we can default
    to this function which just returns the minimum HTML snippet that contains just the target element.

    For the executor, specifically, this is used to provide some context to the previous round of execution.
    """
    soup = BeautifulSoup("<body id='body-id'></body>", "html.parser")
    new_tag = soup.new_tag(target_element.name)

    for attr_name, attr_value in target_element.attrs.items():
        new_tag[attr_name] = attr_value

    # Add the innerText of the target element
    new_tag.string = target_element.get_text()

    # **Ensure the new_tag is added to soup**
    assert soup.body is not None
    soup.body.append(new_tag)

    # Clean and return the HTML
    html = str(soup)
    return clean_html(
        html=html,
        prettify=True,
        strip_annotation=strip_annotation,
        dont_remove_ids={element_id},
    )


def get_previous_round_html_snippet_from_html(
    html: str,
    target_element_id: str,
    strip_annotation: bool = False,
) -> str:
    """
    Version of get_previous_round_html_snippet that works with raw HTML input.
    """
    try:
        html_snippet = get_html_snippet_str_from_element(
            html=html,
            target_element_id=target_element_id,
            num_siblings=5,
            strip_annotation=strip_annotation,
        )
    except Exception:
        # If we can't find the target element, return simplified HTML
        html_snippet = clean_html(
            html=html,
            prettify=True,
            strip_annotation=strip_annotation,
            add_the_destructive_visitor=True,
        )

    return html_snippet


def get_previous_round_html_snippet(
    webarena_lite_action: WebArenaLiteAction,
    strip_annotation: bool = False,
) -> str:
    """Original function that works with WebArenaLiteAction"""
    html = get_html_from_action(webarena_lite_action)
    try:
        target_element_id = get_action_information(webarena_lite_action)["action"][
            "element_id"
        ]
    except Exception:
        target_element_id = ""

    return get_previous_round_html_snippet_from_html(
        html=html,
        target_element_id=target_element_id,
        strip_annotation=strip_annotation,
    )


def classify_website(html_str: str) -> WebArenaLiteWebsite | None:
    """
    Classifies the given HTML string into one of:
    - 'shopping_admin'
    - 'map'
    - 'shopping'
    - 'reddit'
    - 'gitlab'
    If none of the rules match, returns None.
    """
    # --- MAP ---
    if re.search(r"OpenStreetMap", html_str, re.IGNORECASE):
        return WebArenaLiteWebsite.MAP

    # --- SHOPPING_ADMIN ---
    if (
        re.search(r"Magento Admin Panel", html_str, re.IGNORECASE)
        or re.search(r"Braintree Virtual Terminal", html_str, re.IGNORECASE)
        or re.search(r"Manage Encryption Key", html_str, re.IGNORECASE)
    ):
        return WebArenaLiteWebsite.SHOPPING_ADMIN

    # --- SHOPPING ---
    if (
        re.search(r"One Stop Market", html_str, re.IGNORECASE)
        or re.search(r"My Wish List", html_str, re.IGNORECASE)
        or re.search(r"My Cart", html_str, re.IGNORECASE)
    ):
        return WebArenaLiteWebsite.SHOPPING

    # --- GITLAB ---
    if re.search(r"GitLab", html_str, re.IGNORECASE):
        return WebArenaLiteWebsite.GITLAB

    # --- REDDIT ---
    if re.search(r"Postmill", html_str, re.IGNORECASE):
        return WebArenaLiteWebsite.REDDIT

    # If no match, return None
    return None


def format_training_data_for_torchtune(
    messages: list[ServerMessage],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    output: str,
) -> TorchTuneData:
    """
    This function formats the training data for the torchtune format.
    """
    prompt = tokenizer.apply_chat_template(
        messages,  # type: ignore
        tokenize=False,
        add_generation_prompt=True,
    )

    assert isinstance(prompt, str), f"The prompt is not a string: {prompt}"

    return TorchTuneData(input=prompt, output=output)


def format_training_data_for_llama_factory(
    messages: list[ServerMessage],
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    output: str,
) -> LlamaFactoryData:
    prompt = tokenizer.apply_chat_template(
        messages,  # type: ignore
        tokenize=False,
        add_generation_prompt=True,
    )

    assert isinstance(prompt, str), f"The prompt is not a string: {prompt}"

    return LlamaFactoryData(instruction=prompt, input="", output=output)


T = TypeVar("T")


def run_coroutine_in_a_separate_thread_with_a_new_event_loop(
    coroutine: Coroutine[Any, Any, T],
) -> T:
    """
    Run a coroutine in a separate thread with a new event loop.
    """

    def f():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coroutine)
            return result
        finally:
            # Cancel leftover tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            # Shut down async generators (important for aiohttp finalizers)
            loop.run_until_complete(loop.shutdown_asyncgens())

            # Now close the loop
            loop.close()
            asyncio.set_event_loop(None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(f).result()


def format_action_string(
    action: WebArenaLiteAction, use_simple_html: bool = False
) -> str:
    """
    Format an action into a human-readable string with detailed information.

    Args:
        action: The action to format
        use_simple_html: Whether to use a simplified HTML representation

    Returns:
        A formatted action string
    """
    # Convert the raw action string to an action object.
    try:
        action_information = get_action_information(action)
    except Exception:
        return get_raw_action_string_from_action(action)

    action_obj, comments = (
        action_information["action"],
        action_information["comment"],
    )

    # Check whether this has returned a none action. If that is so, then you can just return the action string itself.
    if action_obj.get("action_type") == ActionTypes.NONE:
        return action_obj.get("raw_prediction", "")

    # Map the numeric action type (from ActionTypes enum) to a human-readable string.
    action_type_map = {
        ActionTypes.KEY_PRESS: "Press Enter",
        ActionTypes.SCROLL: "Scroll",
        ActionTypes.CLICK: "Click",
        ActionTypes.TYPE: "Type",
        ActionTypes.HOVER: "Hover",
        ActionTypes.GO_BACK: "Go Backward",
        ActionTypes.GO_FORWARD: "Go Forward",
        ActionTypes.SEARCH: "Search",
        ActionTypes.SELECT_DROPDOWN_OPTION: "Select Dropdown Option",
        ActionTypes.STOP: "Exit",
    }
    action_type = action_obj.get("action_type")
    action_type_str = action_type_map.get(ActionTypes(action_type), "Unknown")

    # For scroll actions, append the direction if available.
    if action_type == ActionTypes.SCROLL:
        direction = action_obj.get("direction", "").capitalize()
        if direction:
            action_type_str = f"Scroll {direction}"

    # Begin formatting the output.
    formatted_lines = []

    # Add the comment.
    if len(comments) > 0:
        formatted_lines.append(
            f'**Comment**:\nThe user has taken the following notes before taking the action:\n"{comments}"\n'
        )

    # Add the action type.
    formatted_lines.append(f"**Action Type**: '{action_type_str}'\n")

    # If this action interacts with an element (i.e. an element_id exists),
    # attempt to get and include the target HTML snippet.
    if action_obj.get("element_id") is not None:
        try:
            soup = BeautifulSoup(get_html_from_action(action), "html.parser")
            target_element = soup.select_one(f"[id='{action_obj.get('element_id')}']")
            if target_element is not None:
                html_snippet = (
                    get_minimum_html_snippet_str(
                        target_element, action_obj.get("element_id")
                    )
                    if use_simple_html
                    else get_html_snippet_str(action, num_siblings=3)
                )
                formatted_lines.append(
                    f"**Target Element HTML Snippet:**\n{html_snippet.strip()}\n"
                )
        except Exception:
            # If we cannot get the HTML snippet, then we just skip it.
            pass

    # Add action-specific details.
    if action_type == ActionTypes.TYPE:
        # For a Type action, include the text that was provided.
        text_value = action_obj.get("text", "N/A")
        formatted_lines.append("**Action Specific Information:**")
        formatted_lines.append(f"- Text: {text_value}")
    elif action_type == ActionTypes.SELECT_DROPDOWN_OPTION:
        # For dropdown selection, include the option value.
        option_value = action_obj.get("argument", "N/A")
        formatted_lines.append("**Action Specific Information:**")
        formatted_lines.append(f"- Option Value: {option_value}")
    elif action_type == ActionTypes.SEARCH:
        # For search, include the search text.
        search_text = action_obj.get("text", "N/A")
        formatted_lines.append("**Action Specific Information:**")
        formatted_lines.append(f"- Search Text: {search_text}")
    elif action_type == ActionTypes.STOP:
        # For the stop action, provide the answer.
        answer = action_obj.get("answer", "N/A")
        formatted_lines.append(f"- Answer: {answer}")

    return "\n".join(formatted_lines)


def format_action_steps(
    action_step_descriptions: list[ActionStep],
    formatted_step_prefix: str = "Step",
    use_simple_html: bool = False,
    include_html_snippet: bool = True,
) -> str:
    """
    Format a list of action steps into a detailed, structured format.

    Args:
        action_step_descriptions: List of action step descriptions
        formatted_step_prefix: Prefix to use for step numbering (e.g., "Step" or "Previous Step")

    Returns:
        Formatted string of action steps
    """
    action_steps_formatted = []
    for i, action_step in enumerate(action_step_descriptions):
        action_str = [f"### {formatted_step_prefix} {i+1}"]

        # If this action has an element_id, then can include a contextual html snippet.
        if include_html_snippet:
            try:
                element_id = get_action_information(action_step["action"])["action"][
                    "element_id"
                ]
                if len(element_id) <= 0:
                    raise ValueError("Element id is not present in the action.")

                html_snippet = get_html_snippet_str(
                    action_step["action"],
                    num_siblings=5,
                )
                action_str.append(
                    f"HTML Snippet before taking the action on the target element:\n{html_snippet}"
                )
            except Exception:
                pass

        # Add the formatted action string.
        action_str.append("\nUser Event:")
        action_str.append(
            format_action_string(action_step["action"], use_simple_html=use_simple_html)
        )

        # Add the action description.
        action_str.append("\nDescription:")
        action_str.append(action_step["action_description"])

        action_steps_formatted.append("\n".join(action_str))

    return "\n\n".join(action_steps_formatted)


def format_last_exit_action(
    last_exit_action: WebArenaLiteAction,
    last_exit_action_prefix: str = "Last Action",
    include_html_snippet: bool = True,
) -> str:
    """
    Format the last exit action into a structured format.

    Args:
        last_exit_action: The last exit action

    Returns:
        Formatted string for the last exit action
    """
    if last_exit_action is None:
        return ""

    exit_action_str = [
        f"""
### {last_exit_action_prefix}
After taking all the previous actions, the user exits the task with the following exit action.
""",
    ]

    if include_html_snippet:
        html_snippet = clean_html(
            get_html_from_action(last_exit_action),
            prettify=True,
            strip_annotation=True,
            add_the_destructive_visitor=True,
        )
        exit_action_str.append(
            f"HTML Snippet during the last exit action:\n{html_snippet}"
        )

    exit_action_str.append(format_action_string(last_exit_action, use_simple_html=True))

    return "\n".join(exit_action_str)
