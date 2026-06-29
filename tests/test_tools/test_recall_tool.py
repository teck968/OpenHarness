"""Tests for RecallTool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openharness.tools.base import ToolExecutionContext
from openharness.tools.recall_tool import RecallTool, RecallToolInput, _format_recall_results
from openharness.services.knowledge_store import KnowledgeUnit


@pytest.fixture
def mock_store():
    """Return a mock KnowledgeStore for recall tests."""
    return MagicMock()


@pytest.fixture
def context(tmp_path):
    """Minimal ToolExecutionContext with session metadata."""
    return ToolExecutionContext(
        cwd=tmp_path,
        metadata={"session_id": "test-session-123"},
    )


@pytest.mark.asyncio
async def test_recall_tool_no_store_returns_error(tmp_path):
    """Without a knowledge store, recall returns an error message."""
    from unittest.mock import patch

    with patch(
        "openharness.services.knowledge_store.get_knowledge_store",
        return_value=None,
    ):
        tool = RecallTool()
        result = await tool.execute(
            RecallToolInput(query="test query"),
            ToolExecutionContext(cwd=tmp_path, metadata={"session_id": "test"}),
        )

    assert "not available" in result.output.lower()


@pytest.mark.asyncio
async def test_recall_tool_with_results(mock_store, context):
    """Recall returns formatted results when the store has matches."""
    from unittest.mock import patch

    mock_store.recall = _make_async_mock([
        KnowledgeUnit(
            id=1,
            knowledge_type="PREFERENCE",
            title="git commits",
            content="Jeremy prefers atomic commits.",
            confidence=0.9,
            source="dream",
            similarity=0.95,
        ),
    ])

    with patch(
        "openharness.services.knowledge_store.get_knowledge_store",
        return_value=mock_store,
    ):
        tool = RecallTool()
        result = await tool.execute(
            RecallToolInput(query="how do commits work?"),
            context,
        )

    output = result.output
    assert "PREFERENCE" in output
    assert "git commits" in output
    assert "Jeremy prefers atomic commits" in output
    assert "relevance: 0.95" in output


@pytest.mark.asyncio
async def test_recall_tool_empty_results(mock_store, context):
    """Recall returns 'no relevant knowledge' when store has no matches."""
    from unittest.mock import patch

    mock_store.recall = _make_async_mock([])

    with patch(
        "openharness.services.knowledge_store.get_knowledge_store",
        return_value=mock_store,
    ):
        tool = RecallTool()
        result = await tool.execute(
            RecallToolInput(query="unmatched query"),
            context,
        )

    assert "no relevant knowledge" in result.output.lower()


@pytest.mark.asyncio
async def test_recall_tool_type_filter(mock_store, context):
    """Recall passes type filter to the store."""
    from unittest.mock import patch

    mock_store.recall = _make_async_mock([])

    with patch(
        "openharness.services.knowledge_store.get_knowledge_store",
        return_value=mock_store,
    ):
        tool = RecallTool()
        await tool.execute(
            RecallToolInput(query="query", types="PREFERENCE,FACT"),
            context,
        )

    # Check that recall was called with the parsed type list
    call_args = mock_store.recall.call_args
    assert call_args is not None
    kwargs = call_args.kwargs
    assert kwargs.get("types") == ["PREFERENCE", "FACT"]


@pytest.mark.asyncio
async def test_recall_tool_default_top_k(mock_store, context):
    """Default top_k is 5."""
    from unittest.mock import patch

    mock_store.recall = _make_async_mock([])

    with patch(
        "openharness.services.knowledge_store.get_knowledge_store",
        return_value=mock_store,
    ):
        tool = RecallTool()
        await tool.execute(
            RecallToolInput(query="q"),
            context,
        )

    kwargs = mock_store.recall.call_args.kwargs
    assert kwargs.get("top_k") == 5


def test_is_read_only():
    """Recall is always read-only."""
    tool = RecallTool()
    assert tool.is_read_only(RecallToolInput(query="test")) is True


def test_format_results_structure():
    """_format_recall_results produces markdown with all expected sections."""
    units = [
        KnowledgeUnit(
            id=1,
            knowledge_type="FACT",
            title="test",
            content="Some content.",
            similarity=0.91,
        ),
    ]
    output = _format_recall_results(units)

    assert "# Recalled Knowledge" in output
    assert "[FACT]" in output
    assert "test" in output
    assert "relevance: 0.91" in output
    assert "1 result from long" in output.lower()


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_async_mock(return_value):
    """Return an async mock that resolves to `return_value`."""
    from unittest.mock import AsyncMock

    mock = AsyncMock()
    mock.return_value = return_value
    return mock
