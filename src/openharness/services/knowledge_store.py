"""Knowledge store — CRUD + semantic recall for oh_knowledge.

Wraps psycopg2 (sync) for DB operations and EmbeddingService (async) for
vector generation.  Call sites in `DreamingExecutor` hold a psycopg2
connection and create a KnowledgeStore with it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import psycopg2.extensions

from openharness.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# Global singleton — injected by the PG session backend at startup so the
# recall tool can access it without being passed through the call chain.
_instance: "KnowledgeStore | None" = None


def get_knowledge_store() -> "KnowledgeStore | None":
    """Return the current KnowledgeStore singleton, or None."""
    return _instance


def set_knowledge_store(store: "KnowledgeStore") -> None:
    """Set the global KnowledgeStore singleton."""
    global _instance
    _instance = store


@dataclass
class KnowledgeUnit:
    """A single unit of extracted knowledge."""

    knowledge_type: str
    title: str
    content: str
    confidence: float = 1.0
    source: str = "manual"
    source_session_id: str | None = None
    tags: list[str] = field(default_factory=list)
    id: int | None = None
    embedding: list[float] | None = None
    created_at: float | None = None
    recall_count: int = 0
    similarity: float | None = None


class KnowledgeStore:
    """CRUD + semantic recall for oh_knowledge.

    Uses a sync psycopg2 connection (same pattern as DreamingExecutor)."""

    def __init__(
        self,
        conn: "psycopg2.extensions.connection",
        embedding: EmbeddingService,
    ) -> None:
        self._conn = conn
        self._embedding = embedding

    # ── insert ──────────────────────────────────────────────────────────

    async def insert(self, unit: KnowledgeUnit) -> int | None:
        """Insert a knowledge unit (with embedding) and return its id.

        Returns None if the insert was skipped (duplicate content_hash)."""
        import hashlib

        embedding = await self._embedding.embed_one(
            f"{unit.title}: {unit.content}"
        )
        content_hash = self._compute_content_hash(unit.content)

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO oh_knowledge
               (knowledge_type, title, content, tags, confidence, source,
                source_session_id, embedding, content_hash, source_evidences)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
               ON CONFLICT (content_hash) WHERE NOT archived
               DO NOTHING
               RETURNING id""",
            (
                unit.knowledge_type,
                unit.title,
                unit.content,
                unit.tags,
                unit.confidence,
                unit.source,
                unit.source_session_id,
                embedding,
                content_hash,
                json.dumps(
                    [{"session_id": unit.source_session_id, "turns": []}]
                    if unit.source_session_id
                    else []
                ),
            ),
        )
        row = cur.fetchone()
        self._conn.commit()
        if row:
            logger.debug("Inserted knowledge unit id=%s", row[0])
            return int(row[0])
        return None

    # ── recall ──────────────────────────────────────────────────────────

    async def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        types: list[str] | None = None,
        min_confidence: float = 0.3,
        threshold: float = 0.45,
        session_id: str | None = None,
    ) -> list[KnowledgeUnit]:
        """Semantic recall — query → embedding → vector search → results."""
        query_embedding = await self._embedding.embed_one(query)

        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, knowledge_type, title, content, tags, confidence,
                      source, source_session_id, recall_count,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM oh_knowledge
               WHERE archived = FALSE
                 AND 1 - (embedding <=> %s::vector) > %s
                 AND confidence >= %s
                 AND (%s::text[] IS NULL OR knowledge_type = ANY(%s::text[]))
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (
                query_embedding,
                query_embedding,
                threshold,
                min_confidence,
                types,
                types,
                query_embedding,
                top_k,
            ),
        )
        rows = cur.fetchall()

        units: list[KnowledgeUnit] = []
        for row in rows:
            units.append(
                KnowledgeUnit(
                    id=row[0],
                    knowledge_type=row[1],
                    title=row[2],
                    content=row[3],
                    tags=row[4] or [],
                    confidence=row[5],
                    source=row[6],
                    source_session_id=row[7],
                    recall_count=row[8],
                    similarity=row[9],
                )
            )

        # Bump recall_count and last_recalled_at for returned units
        if units:
            ids = [u.id for u in units if u.id is not None]
            cur.execute(
                """UPDATE oh_knowledge
                   SET recall_count = recall_count + 1,
                       last_recalled_at = now()
                   WHERE id = ANY(%s)""",
                (ids,),
            )
            self._conn.commit()

            # Log the recall for analytics
            if session_id:
                distances = [u.similarity or 0.0 for u in units]
                self._log_recall(
                    session_id, query, query_embedding, ids, distances
                )

        return units

    async def recall_with_embedding(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        types: list[str] | None = None,
        min_confidence: float = 0.3,
        threshold: float = 0.45,
    ) -> list[KnowledgeUnit]:
        """Recall using a pre‑computed embedding (avoids extra API call)."""
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, knowledge_type, title, content, tags, confidence,
                      source, source_session_id, recall_count,
                      1 - (embedding <=> %s::vector) AS similarity
               FROM oh_knowledge
               WHERE archived = FALSE
                 AND 1 - (embedding <=> %s::vector) > %s
                 AND confidence >= %s
                 AND (%s::text[] IS NULL OR knowledge_type = ANY(%s::text[]))
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (
                embedding,
                embedding,
                threshold,
                min_confidence,
                types,
                types,
                embedding,
                top_k,
            ),
        )
        rows = cur.fetchall()
        units = []
        for row in rows:
            units.append(
                KnowledgeUnit(
                    id=row[0],
                    knowledge_type=row[1],
                    title=row[2],
                    content=row[3],
                    tags=row[4] or [],
                    confidence=row[5],
                    source=row[6],
                    source_session_id=row[7],
                    recall_count=row[8],
                    similarity=row[9],
                )
            )
        return units

    # ── recall logging ─────────────────────────────────────────────────────

    def _log_recall(
        self,
        session_id: str,
        query_text: str,
        query_embedding: list[float],
        recalled_ids: list[int],
        distances: list[float],
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO oh_knowledge_recalls
               (session_id, query_text, query_embedding, recalled_ids, distances)
               VALUES (%s, %s, %s, %s, %s)""",
            (session_id, query_text, query_embedding, recalled_ids, distances),
        )
        self._conn.commit()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _compute_content_hash(self, content: str) -> str:
        import hashlib

        normalized = " ".join(content.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
