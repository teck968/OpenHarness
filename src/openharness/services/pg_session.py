"""Postgres-backed session storage for OpenHarness.

Stores full session transcripts in normalized tables with pgvector support,
replacing the flat JSON file backend while keeping the same ``SessionBackend``
Protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from openharness.api.usage import UsageSnapshot

log = logging.getLogger(__name__)
from openharness.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    sanitize_conversation_messages,
)
from openharness.services.session_backend import SessionBackend
from openharness.utils.fs import atomic_write_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 8

_SQL_CREATE_TABLES = [
    # ── meta ────────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    # ── sessions ────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_sessions (
        session_id    TEXT PRIMARY KEY,
        session_key   TEXT,
        cwd           TEXT NOT NULL,
        project_name  TEXT NOT NULL,
        model         TEXT NOT NULL,
        system_prompt TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        ended_at      TIMESTAMPTZ,
        message_count INTEGER NOT NULL DEFAULT 0,
        total_tokens  BIGINT NOT NULL DEFAULT 0,
        current_epoch INTEGER NOT NULL DEFAULT 0,
        dream_run_id  BIGINT
    )
    """,
    # ── messages ────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_messages (
        message_id  BIGSERIAL PRIMARY KEY,
        session_id  TEXT NOT NULL REFERENCES oh_sessions(session_id)
                        ON DELETE CASCADE,
        turn_index  INTEGER NOT NULL,
        role        TEXT NOT NULL,
        epoch       INTEGER NOT NULL DEFAULT 0,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON oh_messages(session_id, turn_index)",
    # ── content_blocks ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_content_blocks (
        block_id       BIGSERIAL PRIMARY KEY,
        message_id     BIGINT NOT NULL REFERENCES oh_messages(message_id)
                           ON DELETE CASCADE,
        block_index    INTEGER NOT NULL,
        block_type     TEXT NOT NULL,
        text_content   TEXT,
        tool_use_id    TEXT,
        tool_name      TEXT,
        tool_input     JSONB,
        result_content TEXT,
        is_error       BOOLEAN NOT NULL DEFAULT FALSE,
        media_type     TEXT,
        source_path    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_blocks_message ON oh_content_blocks(message_id, block_index)",
    "CREATE INDEX IF NOT EXISTS idx_blocks_tool_use_id ON oh_content_blocks(tool_use_id) WHERE tool_use_id IS NOT NULL",
    # ── usage_snapshots ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_usage_snapshots (
        id               BIGSERIAL PRIMARY KEY,
        session_id       TEXT NOT NULL REFERENCES oh_sessions(session_id)
                              ON DELETE CASCADE,
        turn_index       INTEGER NOT NULL,
        recorded_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        input_tokens     INTEGER NOT NULL DEFAULT 0,
        output_tokens    INTEGER NOT NULL DEFAULT 0,
        cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
        cost_usd         NUMERIC(10, 6) NOT NULL DEFAULT 0,
        provider         TEXT,
        model            TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_usage_session ON oh_usage_snapshots(session_id)",
    # ── dream_runs ────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_dream_runs (
        run_id              BIGSERIAL PRIMARY KEY,
        started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
        finished_at         TIMESTAMPTZ,
        status              TEXT NOT NULL DEFAULT 'running',
        sessions_processed  INTEGER NOT NULL DEFAULT 0,
        knowledge_extracted INTEGER NOT NULL DEFAULT 0,
        error_message       TEXT
    )
    """,
    # ── dreamed_messages ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_dreamed_messages (
        session_id      TEXT NOT NULL REFERENCES oh_sessions(session_id)
                            ON DELETE CASCADE,
        last_message_id BIGINT NOT NULL,
        dreamed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (session_id)
    )
    """,
    # ── knowledge ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS oh_knowledge (
        id                BIGSERIAL PRIMARY KEY,
        knowledge_type    TEXT NOT NULL,
        title             TEXT NOT NULL,
        content           TEXT NOT NULL,
        tags              TEXT[] DEFAULT '{}',
        confidence        REAL NOT NULL DEFAULT 1.0,
        source            TEXT,
        source_session_id TEXT,
        source_evidences  JSONB NOT NULL DEFAULT '[]'::jsonb,
        scope             TEXT NOT NULL DEFAULT 'global',
        embedding         VECTOR(1536),
        created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_recalled_at  TIMESTAMPTZ,
        recall_count      INTEGER NOT NULL DEFAULT 0,
        content_hash      TEXT NOT NULL,
        archived          BOOLEAN NOT NULL DEFAULT FALSE
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_content_hash ON oh_knowledge(content_hash) WHERE NOT archived",
    "CREATE INDEX IF NOT EXISTS idx_knowledge_type ON oh_knowledge(knowledge_type) WHERE NOT archived",
]

_DB_MODULE_AVAILABLE = True
try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]
    _DB_MODULE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_name_from_cwd(cwd: str | Path) -> str:
    return Path(cwd).resolve().name


def _block_to_row(
    block: TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock,
    block_index: int,
) -> dict[str, Any]:
    """Convert a content block into a dict suitable for INSERT."""
    base = {
        "block_index": block_index,
        "text_content": None,
        "tool_use_id": None,
        "tool_name": None,
        "tool_input": None,
        "result_content": None,
        "is_error": False,
        "media_type": None,
        "source_path": None,
    }
    if isinstance(block, TextBlock):
        base["block_type"] = "text"
        base["text_content"] = block.text
    elif isinstance(block, ImageBlock):
        base["block_type"] = "image"
        base["media_type"] = block.media_type
        base["source_path"] = block.source_path or ""
    elif isinstance(block, ToolUseBlock):
        base["block_type"] = "tool_use"
        base["tool_use_id"] = block.id
        base["tool_name"] = block.name
        base["tool_input"] = json.dumps(block.input)
    else:  # ToolResultBlock
        base["block_type"] = "tool_result"
        base["tool_use_id"] = block.tool_use_id
        base["result_content"] = block.content
        base["is_error"] = block.is_error
    return base


def _row_to_block(row: dict[str, Any]) -> TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock:
    """Reconstruct a content block from a database row."""
    block_type = row["block_type"]
    if block_type == "text":
        return TextBlock(text=row["text_content"] or "")
    if block_type == "image":
        return ImageBlock(
            media_type=row["media_type"] or "image/png",
            data="",
            source_path=row["source_path"] or "",
        )
    if block_type == "tool_use":
        return ToolUseBlock(
            id=row["tool_use_id"] or "",
            name=row["tool_name"] or "",
            input=row["tool_input"] if isinstance(row["tool_input"], dict) else json.loads(row["tool_input"] or "{}"),
        )
    # tool_result
    return ToolResultBlock(
        tool_use_id=row["tool_use_id"] or "",
        content=row["result_content"] or "",
        is_error=bool(row["is_error"]),
    )


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _migrate_schema(conn: "psycopg2.extensions.connection") -> None:
    """Ensure the Postgres schema is up to date."""
    with conn.cursor() as cur:
        # Create meta table first if it doesn't exist (bootstrap case)
        cur.execute(_SQL_CREATE_TABLES[0])

        try:
            cur.execute(
                "SELECT value FROM oh_meta WHERE key = 'schema_version'"
            )
            row = cur.fetchone()
            current = int(row[0]) if row else 0
        except Exception:
            current = 0

        if current >= SCHEMA_VERSION:
            return

        # Run base table creation (idempotent)
        for sql in _SQL_CREATE_TABLES[1:]:
            cur.execute(sql)

        # v1 → v2: add session_key column
        if current < 2:
            cur.execute(
                "ALTER TABLE oh_sessions ADD COLUMN IF NOT EXISTS session_key TEXT"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_key "
                "ON oh_sessions(session_key) WHERE session_key IS NOT NULL"
            )

        # v2 → v3: add epoch columns for compaction/clear support
        if current < 3:
            cur.execute(
                "ALTER TABLE oh_sessions ADD COLUMN IF NOT EXISTS current_epoch INTEGER NOT NULL DEFAULT 0"
            )
            cur.execute(
                "ALTER TABLE oh_messages ADD COLUMN IF NOT EXISTS epoch INTEGER NOT NULL DEFAULT 0"
            )
            cur.execute(
                "DROP INDEX IF EXISTS idx_messages_session"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON oh_messages(session_id, epoch, turn_index)"
            )

        # v3 → v4: add dream tracking — oh_dream_runs table + dream_run_id FK
        if current < 4:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oh_dream_runs (
                    run_id              BIGSERIAL PRIMARY KEY,
                    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    finished_at         TIMESTAMPTZ,
                    status              TEXT NOT NULL DEFAULT 'running',
                    sessions_processed  INTEGER NOT NULL DEFAULT 0,
                    knowledge_extracted INTEGER NOT NULL DEFAULT 0,
                    error_message       TEXT
                )
            """)
            cur.execute(
                "ALTER TABLE oh_sessions ADD COLUMN IF NOT EXISTS dream_run_id BIGINT"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_dream_run "
                "ON oh_sessions(dream_run_id) WHERE dream_run_id IS NOT NULL"
            )

        # v4 → v5: add dreamed_messages and knowledge tables
        if current < 5:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oh_dreamed_messages (
                    session_id      TEXT NOT NULL REFERENCES oh_sessions(session_id)
                                        ON DELETE CASCADE,
                    last_message_id BIGINT NOT NULL,
                    dreamed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (session_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oh_knowledge (
                    id                BIGSERIAL PRIMARY KEY,
                    knowledge_type    TEXT NOT NULL,
                    title             TEXT NOT NULL,
                    content           TEXT NOT NULL,
                    tags              TEXT[] DEFAULT '{}',
                    confidence        REAL NOT NULL DEFAULT 1.0,
                    source            TEXT,
                    source_session_id TEXT,
                    embedding         VECTOR(1536),
                    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
                    last_recalled_at  TIMESTAMPTZ,
                    recall_count      INTEGER NOT NULL DEFAULT 0,
                    content_hash      TEXT NOT NULL,
                    archived          BOOLEAN NOT NULL DEFAULT FALSE
                )
            """)
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_content_hash "
                "ON oh_knowledge(content_hash) WHERE NOT archived"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_type "
                "ON oh_knowledge(knowledge_type) WHERE NOT archived"
            )

        # v5 → v6: add source_evidences JSONB and scope columns.
        # Backfill existing rows: single‑entry evidence array from
        # source_session_id; scope defaults to 'global' until dreamer sets it.
        if current < 6:
            cur.execute(
                "ALTER TABLE oh_knowledge ADD COLUMN IF NOT EXISTS "
                "source_evidences JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
            cur.execute(
                "ALTER TABLE oh_knowledge ADD COLUMN IF NOT EXISTS "
                "scope TEXT NOT NULL DEFAULT 'global'"
            )
            cur.execute(
                """UPDATE oh_knowledge
                   SET source_evidences =
                       CASE WHEN source_session_id IS NOT NULL
                       THEN jsonb_build_array(
                           jsonb_build_object(
                               'session_id', source_session_id,
                               'turns', '[]'::jsonb
                           )
                       )
                       ELSE '[]'::jsonb
                       END
                   WHERE source_evidences = '[]'::jsonb"""
            )

        # v6 → v7: add session_id to oh_dream_runs for per‑session in‑flight guard
        if current < 7:
            cur.execute(
                "ALTER TABLE oh_dream_runs ADD COLUMN IF NOT EXISTS session_id TEXT"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_dream_runs_session "
                "ON oh_dream_runs(session_id)"
            )

        # v7 → v8: HNSW index for knowledge embeddings + recall log table
        if current < 8:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_embedding "
                "ON oh_knowledge USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = 16, ef_construction = 200)"
            )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oh_knowledge_recalls (
                    id              BIGSERIAL PRIMARY KEY,
                    session_id      TEXT NOT NULL,
                    query_text      TEXT NOT NULL,
                    query_embedding VECTOR(1536),
                    recalled_ids    BIGINT[] NOT NULL,
                    distances       REAL[] NOT NULL,
                    recalled_at     TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

        if current == 0:
            cur.execute(
                "INSERT INTO oh_meta (key, value) VALUES ('schema_version', %s)",
                (str(SCHEMA_VERSION),),
            )
        else:
            cur.execute(
                "UPDATE oh_meta SET value = %s WHERE key = 'schema_version'",
                (str(SCHEMA_VERSION),),
            )

        conn.commit()
        logger.info("pg_session: schema migrated to version %s", SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class PostgresSessionBackend:
    """Session backend that stores transcripts in a Postgres database.

    Implements the ``SessionBackend`` Protocol.  Connections are lazy —
    the database is not contacted until the first save or load call.
    """

    def __init__(self, dsn: str, *, ca_bundle: str | None = None) -> None:
        if not _DB_MODULE_AVAILABLE:
            raise RuntimeError(
                "psycopg2 is required for PostgresSessionBackend. "
                "Install with: pip install openharness-ai[postgres]"
            )
        self._dsn = dsn
        self._ca_bundle = ca_bundle
        self._conn: Any = None
        self._dreaming_executor: Any = None
        self._dreaming_event_loop: Any = None

    def set_dreaming(self, executor: Any, *, loop: Any = None) -> None:
        """Register a :class:`~openharness.services.dreaming.DreamingExecutor`
        for milestone‑triggered dreaming.  *loop* is the event loop used to
        schedule background dream tasks."""
        self._dreaming_executor = executor
        if loop is None:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        self._dreaming_event_loop = loop

    # ── connection management ───────────────────────────────────────────────

    def _ensure_connection(self) -> "psycopg2.extensions.connection":
        if self._conn is not None and not self._conn.closed:
            return self._conn

        conn_kwargs: dict[str, Any] = {}
        if self._ca_bundle:
            conn_kwargs["sslrootcert"] = self._ca_bundle
            conn_kwargs["sslmode"] = "verify-full"

        self._conn = psycopg2.connect(self._dsn, **conn_kwargs)
        _migrate_schema(self._conn)
        return self._conn

    def close(self) -> None:
        """Close the database connection if open."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def end_session(
        self, session_id: str, message_count: int, cwd_str: str, project_name: str,
    ) -> None:
        """Mark a session as ended and trigger session‑end dreaming if warranted."""
        if self._conn is None:
            return
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE oh_sessions SET ended_at = now() WHERE session_id = %s AND ended_at IS NULL",
            (session_id,),
        )
        self._conn.commit()

        # Session‑end dreaming trigger
        if self._dreaming_executor is not None and self._dreaming_event_loop is not None:
            executor = self._dreaming_executor

            # Get last dreamed count
            cur.execute(
                "SELECT last_message_id FROM oh_dreamed_messages WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            last_dreamed = 0
            if row is not None:
                cur.execute(
                    "SELECT COUNT(*) FROM oh_messages WHERE session_id = %s AND message_id <= %s",
                    (session_id, row[0]),
                )
                count_row = cur.fetchone()
                if count_row:
                    last_dreamed = count_row[0]

            if executor.should_dream_on_end(message_count, last_dreamed):
                log.info(
                    "Dream session-end triggered: session=%s count=%d last_dreamed=%d",
                    session_id, message_count, last_dreamed,
                )
                loop = self._dreaming_event_loop
                asyncio.ensure_future(
                    executor.run_for_session(
                        session_id,
                        cwd=cwd_str,
                        project_name=project_name,
                    ),
                    loop=loop,
                )

    def _maybe_trigger_dream(
        self, session_id: str, message_count: int, prev_count: int,
        project_name: str, cwd_str: str,
    ) -> None:
        """Check milestones and schedule a background dream if triggered."""
        if not self._dreaming_executor or not self._dreaming_event_loop:
            return

        executor = self._dreaming_executor
        conn = self._conn

        # Get total message count from DB (not context window size, which
        # shrinks on compaction — we need the cumulative count across all epochs)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM oh_messages WHERE session_id = %s",
            (session_id,),
        )
        total_count = cur.fetchone()[0]

        # Get last dreamed count
        cur.execute(
            "SELECT last_message_id FROM oh_dreamed_messages WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        last_dreamed = 0
        if row is not None:
            cur.execute(
                "SELECT COUNT(*) FROM oh_messages WHERE session_id = %s AND message_id <= %s",
                (session_id, row[0]),
            )
            count_row = cur.fetchone()
            if count_row:
                last_dreamed = count_row[0]

        if not executor.should_dream(total_count, last_dreamed):
            return

        # Guard against concurrent dream runs for this session
        cur.execute(
            "SELECT 1 FROM oh_dream_runs WHERE status = 'running' AND session_id = %s LIMIT 1",
            (session_id,),
        )
        if cur.fetchone():
            log.info(
                "Dream skipped: a run is already in progress for session=%s",
                session_id,
            )
            return

        log.info(
            "Dream milestone triggered: session=%s count=%d last_dreamed=%d",
            session_id, total_count, last_dreamed,
        )

        loop = self._dreaming_event_loop
        asyncio.ensure_future(
            executor.run_for_session(
                session_id,
                cwd=cwd_str,
                project_name=project_name,
            ),
            loop=loop,
        )

    # ── SessionBackend Protocol ─────────────────────────────────────────────

    def get_session_dir(self, cwd: str | Path) -> Path:
        """Return a path for markdown exports (PG backend has no session dir)."""
        return Path(cwd).resolve() / ".openharness" / "sessions"

    def save_snapshot(
        self,
        *,
        cwd: str | Path,
        model: str,
        system_prompt: str,
        messages: list[ConversationMessage],
        usage: UsageSnapshot,
        session_id: str | None = None,
        session_key: str | None = None,
        dream_run_id: int | None = None,
        tool_metadata: dict[str, object] | None = None,
    ) -> Path:
        """Persist a session snapshot to Postgres.

        Detects epoch boundaries: when the message list shrinks (compaction
        or ``/clear``), a new epoch is started.  Pre‑compaction messages are
        preserved in the database forever; only the current epoch is returned
        by ``load_latest`` so the LLM context window stays tight.
        """
        conn = self._ensure_connection()
        messages = sanitize_conversation_messages(messages)
        sid = session_id or uuid4().hex[:12]
        project_name = _project_name_from_cwd(cwd)
        new_count = len(messages)
        cwd_str = str(Path(cwd).resolve())

        with conn.cursor() as cur:
            # 1. Read existing session state
            cur.execute(
                "SELECT message_count, current_epoch FROM oh_sessions WHERE session_id = %s",
                (sid,),
            )
            existing = cur.fetchone()
            prev_count = existing[0] if existing else 0
            current_epoch = existing[1] if existing else 0
            total_tokens = usage.total_tokens

            # 2. Detect epoch boundary (compaction or /clear shrank the list)
            if existing and prev_count > 0 and new_count <= prev_count:
                current_epoch += 1
                prev_count = 0  # re-insert all messages in new epoch

            # 3. Upsert session row
            if existing:
                cur.execute(
                    """UPDATE oh_sessions
                       SET session_key = %s, cwd = %s, project_name = %s,
                           model = %s, system_prompt = %s,
                           message_count = %s, current_epoch = %s,
                           total_tokens = oh_sessions.total_tokens + %s,
                           dream_run_id = COALESCE(oh_sessions.dream_run_id, %s)
                       WHERE session_id = %s""",
                    (
                        session_key,
                        cwd_str,
                        project_name,
                        model,
                        system_prompt,
                        new_count,
                        current_epoch,
                        total_tokens,
                        dream_run_id,
                        sid,
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO oh_sessions
                           (session_id, session_key, cwd, project_name,
                            model, system_prompt, message_count, total_tokens,
                            current_epoch, dream_run_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        sid,
                        session_key,
                        cwd_str,
                        project_name,
                        model,
                        system_prompt,
                        new_count,
                        total_tokens,
                        current_epoch,
                        dream_run_id,
                    ),
                )

            # 4. Insert messages at the current epoch
            new_messages = messages[prev_count:]
            for i, message in enumerate(new_messages):
                turn_index = prev_count + i
                cur.execute(
                    """INSERT INTO oh_messages (session_id, turn_index, role, epoch)
                       VALUES (%s, %s, %s, %s)
                       RETURNING message_id""",
                    (sid, turn_index, message.role, current_epoch),
                )
                message_id = cur.fetchone()[0]

                # Insert content blocks for this message
                for bi, block in enumerate(message.content):
                    row = _block_to_row(block, bi)
                    row["message_id"] = message_id
                    cur.execute(
                        """INSERT INTO oh_content_blocks
                               (message_id, block_index, block_type,
                                text_content, tool_use_id, tool_name,
                                tool_input, result_content, is_error,
                                media_type, source_path)
                           VALUES (%(message_id)s, %(block_index)s, %(block_type)s,
                                   %(text_content)s, %(tool_use_id)s, %(tool_name)s,
                                   %(tool_input)s, %(result_content)s, %(is_error)s,
                                   %(media_type)s, %(source_path)s)""",
                        row,
                    )

            # 5. Record usage snapshot
            cur.execute(
                """INSERT INTO oh_usage_snapshots
                       (session_id, turn_index, input_tokens, output_tokens,
                        cache_hit_tokens, cost_usd, provider, model)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    sid,
                    new_count - 1,
                    getattr(usage, "input_tokens", 0),
                    getattr(usage, "output_tokens", 0),
                    getattr(usage, "cache_hit_tokens", 0),
                    getattr(usage, "cost_usd", 0),
                    getattr(usage, "provider", None),
                    getattr(usage, "model", None),
                ),
            )

            conn.commit()

            # ── dreaming trigger ─────────────────────────────────────────
            if self._dreaming_executor is not None and self._dreaming_event_loop is not None:
                try:
                    self._maybe_trigger_dream(sid, new_count, prev_count, project_name, cwd_str)
                except Exception:
                    log.exception("Dream trigger check failed for %s", sid)

        return Path(str(Path(cwd).resolve())) / ".openharness" / "sessions" / f"session-{sid}.json"

    def load_latest(self, cwd: str | Path) -> dict | None:
        """Load the most recently created session."""
        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM oh_sessions ORDER BY created_at DESC LIMIT 1"
            )
            session = cur.fetchone()
            if session is None:
                return None

            return self._build_snapshot_dict(cur, session)

    def list_snapshots(self, cwd: str | Path, limit: int = 20) -> list[dict]:
        """List recent sessions, newest first."""
        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT session_id, created_at, message_count, model
                   FROM oh_sessions
                   ORDER BY created_at DESC
                   LIMIT %s""",
                (limit,),
            )
            rows = cur.fetchall()
        return [
            {
                "session_id": row["session_id"],
                "summary": "",  # populated by caller if needed
                "message_count": row["message_count"],
                "model": row["model"],
                "created_at": row["created_at"].timestamp()
                if hasattr(row["created_at"], "timestamp")
                else row["created_at"],
            }
            for row in rows
        ]

    def load_by_id(self, cwd: str | Path, session_id: str) -> dict | None:
        """Load a specific session by ID."""
        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM oh_sessions WHERE session_id = %s",
                (session_id,),
            )
            session = cur.fetchone()
            if session is None:
                return None
            return self._build_snapshot_dict(cur, session)

    def load_latest_for_session_key(self, session_key: str) -> dict | None:
        """Load latest session for a given Discord/gateway session_key."""
        conn = self._ensure_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM oh_sessions WHERE session_key = %s "
                "ORDER BY created_at DESC LIMIT 1",
                (session_key,),
            )
            session = cur.fetchone()
            if session is None:
                return None
            return self._build_snapshot_dict(cur, session)

    def export_markdown(
        self,
        *,
        cwd: str | Path,
        messages: list[ConversationMessage],
    ) -> Path:
        """Export the current transcript as a markdown file."""
        session_dir = self.get_session_dir(cwd)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / "transcript.md"
        parts = ["# Session Transcript"]
        for message in messages:
            parts.append(f"\n## {message.role.capitalize()}\n")
            text = message.text.strip()
            if text:
                parts.append(text)
        atomic_write_text(path, "\n".join(parts).strip() + "\n")
        return path

    # ── internal helpers ────────────────────────────────────────────────────

    def _build_snapshot_dict(self, cur, session: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct the snapshot payload dict from normalized tables.

        Only messages from the session's ``current_epoch`` are returned so
        the runtime LLM context window stays tight.  Pre‑compaction epochs
        remain in the database for dreaming / knowledge extraction.
        """
        sid = session["session_id"]
        current_epoch = session.get("current_epoch", 0)

        # Load messages for the current epoch only
        cur.execute(
            """SELECT message_id, turn_index, role
               FROM oh_messages
               WHERE session_id = %s AND epoch = %s
               ORDER BY turn_index""",
            (sid, current_epoch),
        )
        message_rows = cur.fetchall()

        messages: list[dict[str, Any]] = []
        for mrow in message_rows:
            cur.execute(
                """SELECT block_type, text_content, tool_use_id, tool_name,
                          tool_input, result_content, is_error,
                          media_type, source_path
                   FROM oh_content_blocks
                   WHERE message_id = %s
                   ORDER BY block_index""",
                (mrow["message_id"],),
            )
            block_rows = cur.fetchall()
            content: list[dict[str, Any]] = []
            for brow in block_rows:
                bt = brow["block_type"]
                if bt == "text":
                    content.append({"type": "text", "text": brow["text_content"] or ""})
                elif bt == "image":
                    content.append({
                        "type": "image",
                        "media_type": brow["media_type"] or "image/png",
                        "data": "",
                        "source_path": brow["source_path"] or "",
                    })
                elif bt == "tool_use":
                    ti = brow["tool_input"]
                    if isinstance(ti, str):
                        try:
                            ti = json.loads(ti)
                        except (json.JSONDecodeError, TypeError):
                            ti = {}
                    content.append({
                        "type": "tool_use",
                        "id": brow["tool_use_id"] or "",
                        "name": brow["tool_name"] or "",
                        "input": ti or {},
                    })
                elif bt == "tool_result":
                    content.append({
                        "type": "tool_result",
                        "tool_use_id": brow["tool_use_id"] or "",
                        "content": brow["result_content"] or "",
                        "is_error": bool(brow["is_error"]),
                    })
            messages.append({"role": mrow["role"], "content": content})

        # Load latest usage snapshot (for the return payload)
        usage: dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
        cur.execute(
            """SELECT input_tokens, output_tokens
               FROM oh_usage_snapshots
               WHERE session_id = %s
               ORDER BY recorded_at DESC
               LIMIT 1""",
            (sid,),
        )
        urow = cur.fetchone()
        if urow:
            usage = {"input_tokens": urow["input_tokens"], "output_tokens": urow["output_tokens"]}

        summary = ""
        for msg in messages:
            if msg["role"] == "user":
                for block in msg.get("content", []):
                    if block.get("type") == "text":
                        summary = block.get("text", "").strip()[:80]
                        break
                if summary:
                    break

        return {
            "session_id": sid,
            "cwd": session["cwd"],
            "model": session["model"],
            "system_prompt": session["system_prompt"] or "",
            "messages": messages,
            "usage": usage,
            "tool_metadata": {},
            "created_at": session["created_at"].timestamp()
            if hasattr(session["created_at"], "timestamp")
            else time.time(),
            "summary": summary,
            "message_count": session["message_count"],
        }
