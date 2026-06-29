"""Tests for EmbeddingService."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from openharness.services.embedding import EmbeddingService


@pytest.mark.asyncio
async def test_embed_openai():
    """EmbeddingService.embed() calls OpenAI client correctly."""
    fake_embedding = [0.1] * 1536

    with patch(
        "openai.AsyncOpenAI", create=True
    ) as mock_ai:
        mock_client = mock_ai.return_value
        mock_client.embeddings.create = _make_async_mock(
            return_value=_fake_openai_response(fake_embedding)
        )

        service = EmbeddingService(
            backend="openai",
            model="text-embedding-3-small",
            api_key="sk-test",
        )

        results = await service.embed(["hello world"])

    assert len(results) == 1
    assert len(results[0]) == 1536
    assert results[0] == fake_embedding


@pytest.mark.asyncio
async def test_embed_one_returns_single():
    """embed_one returns a flat list, not nested."""
    fake_embedding = [0.2] * 1536

    with patch(
        "openai.AsyncOpenAI", create=True
    ) as mock_ai:
        mock_client = mock_ai.return_value
        mock_client.embeddings.create = _make_async_mock(
            return_value=_fake_openai_response(fake_embedding)
        )

        service = EmbeddingService(
            backend="openai",
            api_key="sk-test",
        )

        result = await service.embed_one("test")

    assert len(result) == 1536
    assert result == fake_embedding


@pytest.mark.asyncio
async def test_embed_multiple():
    """embed() handles multiple inputs in one batch."""
    embeddings = [[0.1 * i] * 1536 for i in range(3)]

    with patch(
        "openai.AsyncOpenAI", create=True
    ) as mock_ai:
        mock_client = mock_ai.return_value
        mock_client.embeddings.create = _make_async_mock(
            return_value=_fake_openai_response(*embeddings)
        )

        service = EmbeddingService(backend="openai", api_key="sk-test")
        results = await service.embed(["a", "b", "c"])

    assert len(results) == 3
    assert results == embeddings


@pytest.mark.asyncio
async def test_embed_empty_list():
    """embed() with empty input returns empty list without API call."""
    # The AsyncOpenAI client is never imported because embed()
    # returns early with an empty list.
    service = EmbeddingService(backend="openai", api_key="sk-test")
    results = await service.embed([])

    assert results == []


def test_unknown_backend_raises():
    """Unknown backend string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        EmbeddingService(backend="nonexistent")


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_async_mock(return_value):
    """Return an async mock that resolves to `return_value`."""
    from unittest.mock import AsyncMock

    mock = AsyncMock()
    mock.return_value = return_value
    return mock


def _fake_openai_response(*embeddings):
    """Return a response object matching OpenAI's embeddings.create shape."""
    from types import SimpleNamespace

    data = [SimpleNamespace(embedding=emb) for emb in embeddings]
    return SimpleNamespace(data=data, usage=SimpleNamespace(total_tokens=4))
