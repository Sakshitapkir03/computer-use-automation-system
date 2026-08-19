"""
Thin google-genai wrapper for the discovery LLM.

Only discovery.py imports this module. replay/ must never import it.
"""
from __future__ import annotations

import base64
import os
import uuid
from dataclasses import dataclass

from google import genai
from google.genai import types

_DEFAULT_MODEL = "gemini-3.6-flash"
_MAX_TOKENS = 4096


# ---------------------------------------------------------------------------
# ToolCall — duck-type replacement for anthropic.types.ToolUseBlock.
# discovery.py accesses .name, .input, and .id — nothing else.
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict
    id: str


# ---------------------------------------------------------------------------
# Locator sub-schema — reused in every tool that targets an element.
# ---------------------------------------------------------------------------

_LOCATOR_SCHEMA: dict = {
    "type": "object",
    "description": (
        "Element locator with a mandatory two-strategy fallback chain. "
        "For inputs with no accessible name (no label, no aria-label), "
        "css_fallback on the name attribute MUST be the primary strategy. "
        "Always include a fallback that differs from the primary."
    ),
    "properties": {
        "strategy": {
            "type": "string",
            "enum": ["role_name", "aria_label", "text_exact", "css_fallback", "xpath"],
            "description": (
                "role_name → get_by_role(role, name=value); "
                "aria_label → get_by_label(value); "
                "text_exact → get_by_text(value, exact=True); "
                "css_fallback → page.locator(CSS selector); "
                "xpath → page.locator('xpath=<expression>') — use for structural "
                "targeting such as //tr[td[normalize-space()='Label']]/td[2]"
            ),
        },
        "value": {
            "type": "string",
            "description": (
                "Accessible name for role_name; label text for aria_label; "
                "exact visible text for text_exact; CSS attribute selector "
                "(e.g. input[name='member_id']) for css_fallback. "
                "Prefer attribute selectors over positional selectors."
            ),
        },
        "role": {
            "type": "string",
            "description": (
                "ARIA role — required when strategy=role_name. "
                "Examples: button, textbox, combobox, link, cell, row."
            ),
        },
        "fallback": {
            "type": "object",
            "description": "Fallback strategy tried if primary fails to resolve.",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": ["role_name", "aria_label", "text_exact", "css_fallback", "xpath"],
                },
                "value": {"type": "string"},
                "role": {"type": "string"},
            },
            "required": ["strategy", "value"],
        },
    },
    "required": ["strategy", "value"],
}


# ---------------------------------------------------------------------------
# Tool definitions — one per Step action type, plus goal_complete / report_stuck.
# ---------------------------------------------------------------------------

_DISCOVERY_FUNCTIONS: list[types.FunctionDeclaration] = [
    types.FunctionDeclaration(
        name="navigate",
        description="Navigate the browser to a URL.",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to navigate to."},
            },
            "required": ["url"],
        },
    ),
    types.FunctionDeclaration(
        name="click",
        description="Click a button, link, or other clickable element.",
        parameters={
            "type": "object",
            "properties": {
                "locator": _LOCATOR_SCHEMA,
                "reversible": {
                    "type": "boolean",
                    "description": (
                        "True for almost all clicks. "
                        "False ONLY for the final irreversible confirmation action "
                        "(e.g. a 'Confirm — Open Account' button). "
                        "Replay will gate irreversible clicks behind auto_confirm=True."
                    ),
                },
            },
            "required": ["locator", "reversible"],
        },
    ),
    types.FunctionDeclaration(
        name="type",
        description=(
            "Fill a text input or select an option from a <select> element. "
            "For <select>, value must be the visible option label (not the value attribute). "
            "Use {param_name} syntax for values that should be parameterised "
            "(e.g. '{member_id}' not '12345')."
        ),
        parameters={
            "type": "object",
            "properties": {
                "locator": _LOCATOR_SCHEMA,
                "value": {
                    "type": "string",
                    "description": (
                        "Text to type or option label to select. "
                        "Use {param_name} for any value a caller should be able to vary."
                    ),
                },
                "reversible": {
                    "type": "boolean",
                    "description": "Typing into a form field is almost always true.",
                },
            },
            "required": ["locator", "value", "reversible"],
        },
    ),
    types.FunctionDeclaration(
        name="read",
        description="Read the visible text of an element and record it as a named output.",
        parameters={
            "type": "object",
            "properties": {
                "locator": _LOCATOR_SCHEMA,
                "output_key": {
                    "type": "string",
                    "description": (
                        "Name under which the read value is stored. "
                        "You MUST use this exact string as a key in goal_complete outputs — "
                        "omitting it from goal_complete causes the run to fail."
                    ),
                },
            },
            "required": ["locator", "output_key"],
        },
    ),
    types.FunctionDeclaration(
        name="wait_for",
        description=(
            "Wait for an element to become visible before proceeding. "
            "Use after navigation or form submission when the next element "
            "may take a moment to appear."
        ),
        parameters={
            "type": "object",
            "properties": {"locator": _LOCATOR_SCHEMA},
            "required": ["locator"],
        },
    ),
    types.FunctionDeclaration(
        name="goal_complete",
        description=(
            "Call this when the goal is fully achieved and all outputs have been read. "
            "Provide a checkpoint assertion that future replay runs can use to verify success."
        ),
        parameters={
            "type": "object",
            "properties": {
                "outputs": {
                    "type": "array",
                    "description": (
                        "One entry per read step. "
                        "MUST NOT be empty if you called read."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "description": "Exact output_key string used in the read step.",
                            },
                            "value": {
                                "type": "string",
                                "description": "Actual text read from the page.",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["str", "int", "decimal", "bool"],
                            },
                            "description": {
                                "type": "string",
                                "description": "Short human-readable label.",
                            },
                        },
                        "required": ["key", "value", "type", "description"],
                    },
                },
                "checkpoint": {
                    "type": "object",
                    "description": (
                        "Assertion used to verify the capability succeeded during replay. "
                        "text_present checks that expected is a substring of the element's text. "
                        "url_matches checks that expected is a substring of the page URL."
                    ),
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["element_visible", "text_present", "url_matches"],
                        },
                        "expected": {
                            "type": "string",
                            "description": (
                                "Text substring, URL pattern, or visible indicator of success."
                            ),
                        },
                        "locator": {
                            **_LOCATOR_SCHEMA,
                            "description": "Required for element_visible and text_present.",
                        },
                    },
                    "required": ["kind", "expected"],
                },
            },
            "required": ["outputs", "checkpoint"],
        },
    ),
    types.FunctionDeclaration(
        name="report_stuck",
        description=(
            "Call this when you cannot make progress — element not found, "
            "unexpected page state, or repeated action failures. "
            "A human operator will take over the live session."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "Clear explanation of what you tried and why you cannot continue."
                    ),
                },
            },
            "required": ["reason"],
        },
    ),
]

_DISCOVERY_TOOL = types.Tool(function_declarations=_DISCOVERY_FUNCTIONS)

_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode="ANY"),
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Wrapper around the google-genai API for discovery.
    Replay never imports or instantiates this class.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = genai.Client(
            api_key=api_key or os.environ["GEMINI_API_KEY"]
        )

    def decide(
        self,
        messages: list[dict],
        system: str,
    ) -> ToolCall:
        """
        Send the current conversation to the model and return its function call.

        tool_config mode="ANY" forces the model to always call a function —
        it cannot emit free text without a function call in the discovery loop.
        This keeps the loop state machine simple: every assistant turn is
        exactly one function call.

        Raises google.genai.errors.APIError on non-retryable API failures.
        Returns a ToolCall with the .name / .input / .id interface that
        discovery.py expects (duck-type equivalent of anthropic.types.ToolUseBlock).
        """
        contents, _ = _translate_messages(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system,
                tools=[_DISCOVERY_TOOL],
                tool_config=_TOOL_CONFIG,
                max_output_tokens=self._max_tokens,
            ),
        )
        if not response.candidates:
            raise RuntimeError(
                "Model returned no candidates. "
                f"Prompt feedback: {response.prompt_feedback!r}"
            )
        candidate = response.candidates[0]
        for part in candidate.content.parts:
            if part.function_call is not None:
                fc = part.function_call
                return ToolCall(
                    name=fc.name,
                    input=fc.args if isinstance(fc.args, dict) else dict(fc.args or {}),
                    id=getattr(fc, "id", None) or f"fc_{uuid.uuid4().hex[:8]}",
                )
        raise RuntimeError(
            f"Model returned no function call "
            f"(finish_reason={candidate.finish_reason!r})."
        )


# ---------------------------------------------------------------------------
# Message translation: Anthropic-format message list → Gemini Content list
# ---------------------------------------------------------------------------

def _translate_messages(
    messages: list[dict],
) -> tuple[list[types.Content], dict[str, str]]:
    """
    Convert discovery.py's Anthropic-format message list to Gemini Content objects.

    Anthropic shapes handled:
      assistant — content list contains ToolCall dataclass instances
      user      — content list contains dicts with type in
                  {text, image, tool_result}

    Gemini shapes emitted:
      role="model"  — one function_call Part per ToolCall
      role="user"   — function_response Parts in their own Content (required by
                      the Gemini API); text + inline_data Parts in a second Content

    Returns (contents, id_to_name) where id_to_name maps tool-call id → tool
    name so FunctionResponse.name can be filled from tool_result.tool_use_id.
    """
    # Build id_to_name from all assistant messages in a first pass so that
    # tool_result items (which appear in later user messages) can resolve names.
    id_to_name: dict[str, str] = {}
    for msg in messages:
        if msg["role"] == "assistant":
            for item in msg["content"]:
                if isinstance(item, ToolCall):
                    id_to_name[item.id] = item.name

    contents: list[types.Content] = []

    for msg in messages:
        role = msg["role"]
        raw_content = msg["content"]

        if role == "assistant":
            parts: list[types.Part] = []
            for item in raw_content:
                if isinstance(item, ToolCall):
                    parts.append(
                        types.Part(
                            function_call=types.FunctionCall(
                                name=item.name,
                                args=item.input,
                                id=item.id,
                            )
                        )
                    )
            if parts:
                contents.append(types.Content(role="model", parts=parts))

        else:  # user
            # Separate function_response parts from text/image parts.
            # The Gemini API requires function_response parts to appear before
            # any subsequent text so we emit them in their own Content first.
            fn_parts: list[types.Part] = []
            other_parts: list[types.Part] = []

            for item in raw_content:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")

                if item_type == "tool_result":
                    fn_name = id_to_name.get(item["tool_use_id"], "unknown")
                    fn_parts.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fn_name,
                                response={"result": item.get("content", "")},
                            )
                        )
                    )

                elif item_type == "text":
                    other_parts.append(types.Part(text=item["text"]))

                elif item_type == "image":
                    src = item.get("source", {})
                    if src.get("type") == "base64":
                        raw_bytes = base64.b64decode(src["data"])
                        other_parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=src.get("media_type", "image/png"),
                                    data=raw_bytes,
                                )
                            )
                        )

            if fn_parts:
                contents.append(types.Content(role="user", parts=fn_parts))
            if other_parts:
                contents.append(types.Content(role="user", parts=other_parts))

    return contents, id_to_name
