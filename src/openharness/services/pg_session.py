"""Postgres-backed session storage for OpenHarness.

Stores full session transcripts in normalized tables with pgvector support,
replacing the flat JSON file backend while keeping the same ``SessionBackend``
Protocol.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from openharness.api.usage import UsageSnapshot
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

SCHEMA_VERSION = 2

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
        total_tokens  BIGINT NOT NULL DEFAULT 0
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
        tool_metadata: dict[str, object] | None = None,
    ) -> Path:
        """Persist a session snapshot to Postgres."""
        conn = self._ensure_connection()
        messages = sanitize_conversation_messages(messages)
        sid = session_id or uuid4().hex[:12]
        project_name = _project_name_from_cwd(cwd)
        new_count = len(messages)
        cwd_str = str(Path(cwd).resolve())

        with conn.cursor() as cur:
            # 1. Upsert session row (track the previous message count)
            cur.execute(
                "SELECT message_count FROM oh_sessions WHERE session_id = %s",
                (sid,),
            )
            existing = cur.fetchone()
            prev_count = existing[0] if existing else 0

            total_tokens = usage.total_tokens

            if existing:
                cur.execute(
                    """UPDATE oh_sessions
                       SET session_key = %s, cwd = %s, project_name = %s,
                           model = %s, system_prompt = %s,
                           message_count = %s,
                           total_tokens = oh_sessions.total_tokens + %s
                       WHERE session_id = %s""",
                    (
                        session_key,
                        cwd_str,
                        project_name,
                        model,
                        system_prompt,
                        new_count,
                        total_tokens,
                        sid,
                    ),
                )
            else:
                cur.execute(
                    """INSERT INTO oh_sessions
                           (session_id, session_key, cwd, project_name,
                            model, system_prompt, message_count, total_tokens)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        sid,
                        session_key,
                        cwd_str,
                        project_name,
                        model,
                        system_prompt,
                        new_count,
                        total_tokens,
                    ),
                )

            # 2. Insert only new messages (those after prev_count)
            # If the message count shrank, compaction replaced old messages
            # with summaries — purge and re-insert everything.
            if new_count <= prev_count:
                cur.execute(
                    "DELETE FROM oh_messages WHERE session_id = %s",
                    (sid,),
                )
                prev_count = 0

            new_messages = messages[prev_count:]
            for i, message in enumerate(new_messages):
                turn_index = prev_count + i
                cur.execute(
                    """INSERT INTO oh_messages (session_id, turn_index, role)
                       VALUES (%s, %s, %s)
                       RETURNING message_id""",
                    (sid, turn_index, message.role),
                )
                message_id = cur.fetchone()[0]

                # 3. Insert content blocks for this message
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

            # 4. Record usage snapshot
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
        """Reconstruct the snapshot payload dict from normalized tables."""
        sid = session["session_id"]

        # Load messages
        cur.execute(
            """SELECT message_id, turn_index, role
               FROM oh_messages
               WHERE session_id = %s
               ORDER BY turn_index""",
            (sid,),
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
