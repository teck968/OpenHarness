"""Tests for KnowledgeStore."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openharness.services.embedding import EmbeddingService
from openharness.services.knowledge_store import KnowledgeStore, KnowledgeUnit


@pytest.fixture
def mock_conn():
    """Return a psycopg2 connection double with a cursor factory."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    return conn


@pytest.fixture
def mock_embedding():
    """Return a mock EmbeddingService."""
    emb = MagicMock(spec=EmbeddingService)

    async def _embed_one(_text):
        return [0.5] * 1536
    emb.embed_one = _embed_one
    return emb


@pytest.mark.asyncio
async def test_insert_creates_unit(mock_conn, mock_embedding):
    """insert() returns the new row id."""
    mock_conn.cursor.return_value.fetchone.return_value = (42,)
    store = KnowledgeStore(mock_conn, mock_embedding)

    unit = KnowledgeUnit(
        knowledge_type="FACT",
        title="test fact",
        content="Jeremy works on Tuesdays",
        confidence=0.9,
        source="dream",
        source_session_id="abc123",
    )

    result = await store.insert(unit)

    assert result == 42
    mock_conn.commit.assert_called()


@pytest.mark.asyncio
async def test_insert_skips_duplicate(mock_conn, mock_embedding):
    """insert() returns None when ON CONFLICT DO NOTHING fires."""
    mock_conn.cursor.return_value.fetchone.return_value = None
    store = KnowledgeStore(mock_conn, mock_embedding)

    unit = KnowledgeUnit(
        knowledge_type="PREFERENCE",
        title="title",
        content="content",
    )

    result = await store.insert(unit)

    assert result is None


@pytest.mark.asyncio
async def test_recall_returns_results(mock_conn, mock_embedding):
    """recall() returns KnowledgeUnit objects for matching rows."""
    cur = mock_conn.cursor.return_value
    cur.fetchall.return_value = [
        (1, "PREFERENCE", "git", "Jeremy uses git", [], 0.8,
         "dream", "sess1", 5, 0.92),
        (2, "FACT", "python", "Python is used", [], 0.7,
         "dream", "sess2", 3, 0.88),
    ]
    store = KnowledgeStore(mock_conn, mock_embedding)

    results = await store.recall(
        "git workflow", top_k=3, threshold=0.7,
        session_id="test-session",
    )

    assert len(results) == 2
    assert results[0].id == 1
    assert results[0].knowledge_type == "PREFERENCE"
    assert results[0].title == "git"
    assert results[0].similarity == 0.92
    assert results[1].id == 2


@pytest.mark.asyncio
async def test_recall_empty_result(mock_conn, mock_embedding):
    """recall() returns empty list when no rows match."""
    mock_conn.cursor.return_value.fetchall.return_value = []
    store = KnowledgeStore(mock_conn, mock_embedding)

    results = await store.recall("nothing matches", threshold=0.9)

    assert results == []


@pytest.mark.asyncio
async def test_recall_logs_when_session_id(mock_conn, mock_embedding):
    """recall() inserts into oh_knowledge_recalls when session_id is given."""
    cur = mock_conn.cursor.return_value
    cur.fetchall.return_value = [
        (1, "FACT", "x", "y", [], 0.9, "dream", "sess1", 0, 0.95),
    ]
    store = KnowledgeStore(mock_conn, mock_embedding)

    await store.recall("query", session_id="log-test")

    # Verify an INSERT into oh_knowledge_recalls was made
    execute_calls = [c[0][0] for c in cur.execute.call_args_list if c[0]]
    recall_insert = [sql for sql in execute_calls
                     if "oh_knowledge_recalls" in sql]
    assert len(recall_insert) >= 1


@pytest.mark.asyncio
async def test_recall_no_log_without_session_id(mock_conn, mock_embedding):
    """recall() does not log when session_id is omitted."""
    cur = mock_conn.cursor.return_value
    cur.fetchall.return_value = [
        (1, "FACT", "x", "y", [], 0.9, "dream", "sess1", 0, 0.95),
    ]
    store = KnowledgeStore(mock_conn, mock_embedding)

    await store.recall("query")
    # session_id=None -> log should not be called

    execute_calls = [c[0][0] for c in cur.execute.call_args_list if c[0]]
    recall_insert = [sql for sql in execute_calls
                     if "oh_knowledge_recalls" in sql]
    assert len(recall_insert) == 0


@pytest.mark.asyncio
async def test_recall_with_embedding_skips_api_call(mock_conn, mock_embedding):
    """recall_with_embedding() uses the provided vector directly."""
    cur = mock_conn.cursor.return_value
    cur.fetchall.return_value = [
        (3, "PATTERN", "deploy", "Fridays", ["deploy"], 0.6,
         "dream", "sess3", 1, 0.95),
    ]
    store = KnowledgeStore(mock_conn, mock_embedding)

    precomputed = [0.3] * 1536
    results = await store.recall_with_embedding(
        precomputed, top_k=3, threshold=0.7,
    )

    assert len(results) == 1
    assert results[0].id == 3
