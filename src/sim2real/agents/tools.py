"""Tools an agent can call: a typed argument model, a description, and a handler.

Every call is validated against the argument model before the handler runs, so a
malformed or truncated input from the model never reaches a handler. Expected failures
(rule violations, missing files, bad values) come back to the agent as error results it
can act on; unexpected exceptions propagate, because they are bugs.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from sim2real.errors import RuleViolation
from sim2real.store.event_store import ConcurrencyConflict
from sim2real.models.types import MediaPart, ToolSpec


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False
    end_run: bool = False
    """Stop the agent's invocation after this turn, e.g. once it waits for a job."""
    media: tuple[MediaPart, ...] = ()
    """Images and PDFs shown to the model with the text."""


class ToolFailure(Exception):
    """Raised by a handler for a failure the agent should see and can correct."""


EXPECTED_FAILURES = (ToolFailure, RuleViolation, PermissionError, FileNotFoundError, NotADirectoryError,
                     IsADirectoryError, ValueError, KeyError, ConcurrencyConflict)
"""Failures the agent sees as error results; ConcurrencyConflict means the record stayed busy, so trying again
works."""


def plain_schema(schema: dict) -> dict:
    """The JSON schema with ``$ref`` definitions expanded inline and titles removed.

    Not every provider resolves references in tool schemas; an inline schema works everywhere.
    """
    definitions = schema.get("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = walk(definitions[node["$ref"].rsplit("/", 1)[-1]])
                return {**target, **{k: walk(v) for k, v in node.items() if k != "$ref"}}
            # A string "title" is an annotation; a property named "title" has a dict value and stays.
            return {k: walk(v) for k, v in node.items()
                    if k != "$defs" and not (k == "title" and isinstance(v, str))}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(schema)


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[Any], ToolOutcome | str]

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description,
                        input_schema=plain_schema(self.arguments.model_json_schema()))

    def invoke(self, raw: dict) -> ToolOutcome:
        try:
            args = self.arguments.model_validate(raw)
        except ValidationError as exc:
            problems = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
            return ToolOutcome(json.dumps({"invalid_arguments": problems}), is_error=True)
        try:
            outcome = self.handler(args)
        except EXPECTED_FAILURES as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            return ToolOutcome(f"{type(exc).__name__}: {message}", is_error=True)
        return outcome if isinstance(outcome, ToolOutcome) else ToolOutcome(outcome)
