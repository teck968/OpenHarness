"""Embedding service for generating vector embeddings from text.

Supports OpenAI and local backends.  Used by KnowledgeStore and dreaming
pipeline for semantic recall and cross‑session reinforcement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings from text using a configurable backend."""

    def __init__(
        self,
        backend: Literal["openai", "local"] = "openai",
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        self._backend_name = backend
        self._model = model
        self._dimensions = dimensions

        if backend == "openai":
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
        elif backend == "local":
            self._client = None  # lazy load
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

        self._local_model: object | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for one or more texts."""
        if not texts:
            return []
        if self._backend_name == "openai":
            return await self._embed_openai(texts)
        return await self._embed_local(texts)

    async def embed_one(self, text: str) -> list[float]:
        """Return a single embedding."""
        results = await self.embed([text])
        return results[0]

    async def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
        )
        return [item.embedding for item in response.data]

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(self._model)

        embeddings = await asyncio.to_thread(
            self._local_model.encode, texts, normalize_embeddings=True
        )
        return embeddings.tolist()
