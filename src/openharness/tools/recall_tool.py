"""Recall tool — semantic search over the knowledge base.

The agent calls this during sessions to find relevant knowledge from past
dream extractions without cramming everything into the system prompt.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)


class RecallToolInput(BaseModel):
    """Arguments for knowledge recall."""

    query: str = Field(description="Natural language query describing what you need to remember")
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return",
    )
    types: str | None = Field(
        default=None,
        description="Optional comma‑separated types to filter: PREFERENCE,FACT,PATTERN,DECISION,RELATIONSHIP,CAVEAT,SELF_IMPROVEMENT",
    )


class RecallTool(BaseTool):
    """Recall relevant knowledge from long‑term memory using semantic search."""

    name = "recall"
    description = (
        "Recall relevant knowledge from long-term memory using semantic search. "
        "Use this when you need to remember Jeremy's preferences, past decisions, "
        "or patterns learned from previous sessions."
    )
    input_model = RecallToolInput

    def __init__(self, knowledge_store: object | None = None) -> None:
        super().__init__()
        self._store = knowledge_store

    def is_read_only(self, arguments: RecallToolInput) -> bool:
        return True

    async def execute(
        self, arguments: RecallToolInput, context: ToolExecutionContext
    ) -> ToolResult:
        from openharness.services.knowledge_store import get_knowledge_store

        store = get_knowledge_store()
        if store is None:
            return ToolResult(output="Knowledge store not available in this session.")

        type_list: list[str] | None = None
        if arguments.types:
            type_list = [t.strip() for t in arguments.types.split(",") if t.strip()]

        results = await store.recall(
            arguments.query,
            top_k=min(arguments.top_k, 20),
            types=type_list,
            min_confidence=0.5,
            threshold=0.65,
            session_id=context.metadata.get("session_id"),
        )
        if not results:
            return ToolResult(output="No relevant knowledge found.")

        return ToolResult(output=_format_recall_results(results))


def _format_recall_results(results: list[Any]) -> str:
    lines = ["# Recalled Knowledge"]
    for r in results:
        sim = r.similarity or 0.0
        lines.append("")
        lines.append(
            f"## [{r.knowledge_type}] {r.title} (relevance: {sim:.2f})"
        )
        lines.append(r.content)

    lines.append("")
    lines.append(
        f"_{len(results)} result{'s' if len(results) > 1 else ''} from "
        f"long‑term memory._"
    )
    return "\n".join(lines)
