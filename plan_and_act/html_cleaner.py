import re
from abc import ABC
from dataclasses import dataclass
from typing import AbstractSet, TypedDict

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from typing_extensions import NotRequired, Unpack, override

##################################################
#               Useful Models                    #
##################################################


@dataclass
class CleanedHTML:
    cleaned_html: str
    id_mapping: dict[str, int]
    choices: list[list[str]]


class HTMLCleanerArgs(TypedDict):
    html: str
    id_name: str
    hidden_element_ids: NotRequired[AbstractSet[str]]
    self_hidden_element_ids: NotRequired[AbstractSet[str]]
    dont_remove_ids: NotRequired[AbstractSet[str]]


class NodeVisitor(ABC):
    _id_name: str
    _dont_remove_ids: AbstractSet[str]

    def __init__(self, id_name: str, dont_remove_ids: AbstractSet[str]) -> None:
        self._id_name = id_name
        self._dont_remove_ids = dont_remove_ids

    def visit_tag_before_children(self, node: Tag, *, hidden: bool) -> bool:
        return True

    def visit_tag_after_children(self, node: Tag, *, self_hidden: bool) -> None:
        pass

    def visit_string(self, node: NavigableString, *, index: int) -> None:
        pass


#################################################################
#               Types of Node-Specific Visitors                 #
#################################################################


class DefaultVisitor(NodeVisitor):
    # These tags are always removed, without any further checks.
    _TAGS_BLOCKLIST: AbstractSet[str] = {
        "noscript",
        "script",
        "style",
        "footer",
    }

    # Only the following attributes are allowed on tags. There might be more checks
    # later on, but this is a good start.
    _ATTRS_ALLOWLIST: AbstractSet[str] = {
        "alt",
        "contenteditable",
        "placeholder",
        "role",
        "title",
        "type",
        "name",
        "value",
        "selected",
        "expanded",
        "aria_description",
        "aria_label",
        "aria_role",
        "input_checked",
        "input_value",
        "label",
        "option_selected",
        "text_value",
        "data-value",
        "data-text",
        "data-testid",
        "data-label",
        # "data-bbox", # TODO: This makes the HTMLs toooooo messy.
        "data-status",
    }

    # Common values for the "role" attribute that we can safely ignore.
    _COMMON_IGNORED_ROLE_VALUES: AbstractSet[str] = {
        "main",
        "none",
        "presentation",
        "text",
    }

    # These tags are allowed to be collapsed in certain cases, removing this level
    # from the DOM tree.
    _COLLAPSIBLE_TAGS: AbstractSet[str] = {
        "abbr",
        "b",
        "dfn",
        "div",
        "em",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "i",
        "mark",
        "p",
        "small",
        "span",
        "strong",
        "u",
        "text",
        "icon",
        "div",
    }

    # These are tags are allowed to have their text content consolidated.
    _SMOOTHABLE_TAGS: AbstractSet[str] = {
        "abbr",
        "b",
        "dfn",
        "em",
        "i",
        "mark",
        "small",
        "span",
        "strong",
        "u",
        "text",
    }

    # Only these element will have a id attribute in the cleaned HTML.
    _INTERACTABLE_TAGS: AbstractSet[str] = {
        "input",
        "textarea",
        "select",
        "option",
        "button",
        "a",
        "nav",
        "label",
        "radio",
        "combobox",
        "checkbox",
        "tr",
        "link",
        "span",
        "div",
        "li",
        "strong",
        "use",
        "path",
        "svg",
        "img",
        "p",
        "b",
    }

    # These tags should never be decomposed since they are used for user input.
    _NEVER_DECOMPOSE_TAGS: AbstractSet[str] = {
        "textarea",
        "input",
        "select",
        "option",
        "button",
        "a",
    }

    @override
    def visit_tag_before_children(self, node: Tag, *, hidden: bool) -> bool:
        if (
            hidden or node.name in DefaultVisitor._TAGS_BLOCKLIST
        ) and node.name not in DefaultVisitor._NEVER_DECOMPOSE_TAGS:
            if node.attrs.get(self._id_name) not in self._dont_remove_ids:
                node.decompose()
                return False

        self._prune_attrs(node)
        DefaultVisitor._prune_children(node)

        return True

    @override
    def visit_tag_after_children(self, node: Tag, *, self_hidden: bool) -> None:
        if (
            len(node) == 0
            and (self_hidden or self._has_no_attrs(node))
            and node.name not in DefaultVisitor._NEVER_DECOMPOSE_TAGS
            and node.attrs.get(self._id_name) not in self._dont_remove_ids
        ):
            node.decompose()
            return

        DefaultVisitor._consolidate_text(node)
        self._collapse_node(node)

    @override
    def visit_string(self, node: NavigableString, index: int) -> None:
        # Remove comments.
        if isinstance(node, Comment):
            node.extract(index)
            return
        s = node.strip()
        # Remove empty strings.
        if len(s) == 0:
            node.extract(index)
            return
        # Strip leading and trailing whitespace.
        if len(s) < len(node):
            node.replace_with(s)

    def _prune_attrs(self, node: Tag) -> None:
        # Make a copy of the keys, because we're modifying the dict while iterating.
        attr_names = list(node.attrs.keys())
        for attr_name in attr_names:
            # NOTE: For now, I am disabling this because the WebArenaLite data already gives us the ids of the elements that we can interact with.
            if attr_name == self._id_name:
                # if (
                #     node.name in DefaultVisitor._INTERACTABLE_TAGS
                #     or node.attrs.get("role") in DefaultVisitor._INTERACTABLE_TAGS
                # ):
                #     node.attrs["id"] = node.attrs.pop(attr_name)
                # else:
                #     del node.attrs[attr_name]
                continue

            if attr_name == "aria-label":
                node.attrs["label"] = node.attrs.pop(attr_name)
                continue

            if attr_name == "aria-selected":
                node.attrs["selected"] = node.attrs.pop(attr_name)
                continue

            if attr_name == "aria-expanded":
                node.attrs["expanded"] = node.attrs.pop(attr_name)
                continue

            if attr_name not in DefaultVisitor._ATTRS_ALLOWLIST:
                del node.attrs[attr_name]
                continue

            attr_value = node.attrs[attr_name]

            if not isinstance(attr_value, str):
                # Right now we don't have a good way of handling these
                continue

            if attr_name == "href":
                # Map the url into the local domain and also trim it to a reasonable length
                node.attrs[attr_name] = attr_value[:100]
                continue

            if attr_name == "rel":
                del node.attrs[attr_name]
                continue

            if len(attr_value.strip()) == 0:
                del node.attrs[attr_name]
                continue

            if attr_name == "contenteditable" and attr_value != "true":
                del node.attrs[attr_name]
                continue

            if (
                attr_name == "role"
                and attr_value in DefaultVisitor._COMMON_IGNORED_ROLE_VALUES
            ):
                del node.attrs[attr_name]
                continue

            if node.name == "a":
                if attr_name == "role" and attr_value in ("button", "link"):
                    del node.attrs[attr_name]
                continue

            if node.name == "button":
                if attr_name == "type" and attr_value == "button":
                    del node.attrs[attr_name]
                if attr_name == "role" and attr_value == "button":
                    del node.attrs[attr_name]
                continue

            if node.name == "div":
                if attr_name == "role" and attr_value == "button":
                    node.name = "button"
                    del node.attrs[attr_name]
                continue

            if node.name == "input":
                if attr_name == "type" and attr_value == "text":
                    del node.attrs[attr_name]
                continue

    @staticmethod
    def _prune_children(node: Tag) -> None:
        if node.name == "svg":
            node.clear()

    def _has_no_attrs(self, node: Tag) -> bool:
        """
        We count the i attribute as an attribute as well. Hence, we don't remove none of the elements that have an id attribute.
        """
        return len(node.attrs) == 0

    def _collapse_node(self, node: Tag) -> None:
        if len(node) != 1:
            return

        if node.name in DefaultVisitor._NEVER_DECOMPOSE_TAGS:
            return

        only_child = node.contents[0]
        if (
            node.name in DefaultVisitor._SMOOTHABLE_TAGS
            and self._has_no_attrs(node)
            and isinstance(only_child, NavigableString)
            and node.attrs.get(self._id_name) not in self._dont_remove_ids
        ):
            # We can just unwrap the text node, because it only contains text anyways.
            node.unwrap()
            return

        # Only there for the type checker
        if not isinstance(only_child, Tag):
            return

        if (
            node.name in DefaultVisitor._COLLAPSIBLE_TAGS
            and self._has_no_attrs(node)
            and node.attrs.get(self._id_name) not in self._dont_remove_ids
        ):
            node.unwrap()
            return

        if (
            only_child.name in DefaultVisitor._COLLAPSIBLE_TAGS
            and self._has_no_attrs(only_child)
            and only_child.attrs.get(self._id_name) not in self._dont_remove_ids
        ):
            only_child.unwrap()

    @staticmethod
    def _consolidate_text(node: Tag) -> None:
        """Extracts all NavigableStrings from the children of the given node and
        replaces them with a single NavigableString containing the concatenated text.
        """
        consolidated_strings = []

        for i in range(len(node) - 1, -1, -1):
            child = node.contents[i]
            if isinstance(child, NavigableString):
                consolidated_strings.append(child.strip())
                child.extract(i)

        if len(consolidated_strings) > 0:
            node.append(" ".join(reversed(consolidated_strings)))


class StructuralDestructionVisitor(NodeVisitor):
    """This thing just unwraps everything that is not interactive basically. Onyl use this if the context length is too much and this is your only option. Prefer this over just trimming the HTML to the context length since it might require hovering and stuff."""

    @override
    def visit_tag_after_children(self, node: Tag, *, self_hidden: bool) -> None:
        DefaultVisitor._consolidate_text(node)

        # Remove all elements that are not interactive
        if node.name in DefaultVisitor._NEVER_DECOMPOSE_TAGS or node.name in (
            "body",
            "html",
        ):
            return

        if node.attrs.get(self._id_name) not in self._dont_remove_ids:
            try:
                node.unwrap()
            except Exception:
                pass


#################################################################
#                    HTML Cleaner Class                         #
#################################################################


class HTMLCleaner:
    _html: str
    _id_name: str  # This is the name of the id in the html, e.g. "backend_node_id"
    _hidden_element_ids: AbstractSet[str]
    _self_hidden_element_ids: AbstractSet[str]
    _dont_remove_ids: AbstractSet[str]

    def __init__(
        self,
        **kwargs: Unpack[HTMLCleanerArgs],
    ) -> None:
        self._html = kwargs["html"]
        self._id_name = kwargs["id_name"]
        self._hidden_element_ids = (
            hidden_element_ids
            if (hidden_element_ids := kwargs.get("hidden_element_ids")) is not None
            else set()
        )
        self._self_hidden_element_ids = (
            self_hidden_element_ids
            if (self_hidden_element_ids := kwargs.get("self_hidden_element_ids"))
            is not None
            else set()
        )
        self._dont_remove_ids = (
            dont_remove_ids
            if (dont_remove_ids := kwargs.get("dont_remove_ids")) is not None
            else set()
        )

    def clean_html(
        self,
        add_the_destructive_visitor: bool = False,
        prettify: bool = True,
        strip_annotation: bool = False,
    ) -> str:
        document = BeautifulSoup(self._html, "html.parser")
        body = document.body
        if body is None:
            body = document

        visitors: list[NodeVisitor] = [
            DefaultVisitor(self._id_name, self._dont_remove_ids)
        ]
        if add_the_destructive_visitor:
            visitors.append(
                StructuralDestructionVisitor(self._id_name, self._dont_remove_ids)
            )
        for visitor in visitors:
            self._traverse(body, visitor)

        if body.decomposed:
            return ""

        simplified_html = (
            str(body.prettify()).strip() if prettify else str(body).strip()
        )

        if strip_annotation:
            # Remove all the id="..." attribute optionally
            simplified_html = re.sub(r'\s+id="([^"]+)"', "", simplified_html)

        return simplified_html

    def _traverse(self, node: Tag, visitor: NodeVisitor) -> None:
        node_custom_id = node.attrs.get(self._id_name)
        hidden = node_custom_id in self._hidden_element_ids

        if not visitor.visit_tag_before_children(node, hidden=hidden):
            return

        # Loop backwards using index, because we're removing elements as we go, which
        # would break the iteration otherwise.
        for i in range(len(node) - 1, -1, -1):
            child = node.contents[i]

            if isinstance(child, Tag):
                self._traverse(child, visitor)
            elif isinstance(child, NavigableString):
                visitor.visit_string(child, index=i)

        self_hidden = node_custom_id in self._self_hidden_element_ids

        visitor.visit_tag_after_children(node, self_hidden=self_hidden)
