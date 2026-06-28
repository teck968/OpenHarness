"""Load and compact session transcripts from Postgres for dream prompts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── compaction thresholds ─────────────────────────────────────────────────

ASSISTANT_FULL_MAX = 500          # chars — full text below this
ASSISTANT_HEAD = 300              # chars — keep at start of long messages
ASSISTANT_TAIL = 200              # chars — keep at end of long messages
TOOL_RESULT_MAX = 200             # chars — truncate results above this
TRUNCATION_MARKER = "[TRUNCATED]"


# ── public api ─────────────────────────────────────────────────────────────

def load_compacted_transcript(
    conn: Any,
    session_id: str,
) -> str:
    """Return a compacted transcript of *session_id* for the dream prompt.

    Queries ``oh_messages`` + ``oh_content_blocks`` across all epochs.
    Applies head+tail truncation to long assistant texts and truncates
    large tool results.
    """
    rows = _load_messages(conn, session_id)
    session_info = _load_session_info(conn, session_id)
    lines: list[str] = []

    # header
    project = session_info.get("project_name", "?")
    model = session_info.get("model", "?")
    lines.append(f"[Session: {session_id} | Project: {project} | Model: {model}]")

    for msg in rows:
        role = msg["role"]
        label = _role_label(role)
        for block in msg.get("blocks", []):
            text = _compact_block(block, role)
            if text is not None:
                lines.append(f"--- Turn {msg['turn_index']} ({label}) ---")
                lines.append(text)

    return "\n".join(lines)


def load_full_transcript(
    conn: Any,
    session_id: str,
) -> str:
    """Return the *full* transcript — no compaction.  Written to disk for
    the dream child to ``read_file`` truncated sections on demand."""
    rows = _load_messages(conn, session_id)
    session_info = _load_session_info(conn, session_id)
    lines: list[str] = []
    project = session_info.get("project_name", "?")
    model = session_info.get("model", "?")
    lines.append(f"[Session: {session_id} | Project: {project} | Model: {model}] [FULL TRANSCRIPT]")

    for msg in rows:
        role = msg["role"]
        label = _role_label(role)
        for block in msg.get("blocks", []):
            text = _full_block(block, role)
            if text is not None:
                lines.append(f"--- Turn {msg['turn_index']} ({label}) ---")
                lines.append(text)

    return "\n".join(lines)


# ── internal helpers ───────────────────────────────────────────────────────

def _load_messages(conn: Any, session_id: str) -> list[dict[str, Any]]:
    """Query messages + content blocks and return structured rows."""
    cur = conn.cursor()
    cur.execute(
        """SELECT m.turn_index, m.role, cb.block_index, cb.block_type,
                  cb.text_content, cb.tool_use_id, cb.tool_name, cb.tool_input,
                  cb.result_content, cb.is_error
           FROM oh_messages m
           JOIN oh_content_blocks cb ON cb.message_id = m.message_id
           WHERE m.session_id = %s
           ORDER BY m.epoch, m.turn_index, cb.block_index""",
        (session_id,),
    )
    # group blocks by turn_index
    turns: dict[int, dict[str, Any]] = {}
    for row in cur.fetchall():
        ti = row[0]
        if ti not in turns:
            turns[ti] = {"turn_index": ti, "role": row[1], "blocks": []}
        turns[ti]["blocks"].append({
            "block_type": row[3],
            "text_content": row[4],
            "tool_use_id": row[5],
            "tool_name": row[6],
            "tool_input": row[7],
            "result_content": row[8],
            "is_error": row[9],
        })
    return [turns[k] for k in sorted(turns)]


def _load_session_info(conn: Any, session_id: str) -> dict[str, str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT project_name, model FROM oh_sessions WHERE session_id = %s",
        (session_id,),
    )
    row = cur.fetchone()
    if row is None:
        return {"project_name": "unknown", "model": "unknown"}
    return {"project_name": row[0], "model": row[1]}


def _role_label(role: str) -> str:
    return {"user": "user", "assistant": "assistant", "system": "system"}.get(role, role)


def _compact_block(block: dict[str, Any], role: str) -> str | None:
    bt = block["block_type"]
    text = block.get("text_content") or ""
    tool_name = block.get("tool_name") or ""
    tool_input = block.get("tool_input")
    result = block.get("result_content") or ""

    if bt == "text":
        if role == "user":
            return text
        if len(text) <= ASSISTANT_FULL_MAX:
            return text
        # head + tail
        return text[:ASSISTANT_HEAD] + " " + TRUNCATION_MARKER + " " + text[-ASSISTANT_TAIL:]

    if bt == "tool_use":
        inp = _compact_json(tool_input) if tool_input else ""
        return f"tool: {tool_name}({inp})"

    if bt == "tool_result":
        if result:
            out = result[:TOOL_RESULT_MAX]
            if len(result) > TOOL_RESULT_MAX:
                out += " " + TRUNCATION_MARKER
            return f"> {out.strip()}"
        return None

    if bt == "image":
        return None  # skip images

    return None


def _full_block(block: dict[str, Any], role: str) -> str | None:
    bt = block["block_type"]
    text = block.get("text_content") or ""
    tool_name = block.get("tool_name") or ""
    tool_input = block.get("tool_input")
    result = block.get("result_content") or ""

    if bt == "text":
        return text
    if bt == "tool_use":
        inp = _compact_json(tool_input) if tool_input else ""
        return f"tool: {tool_name}({inp})"
    if bt == "tool_result":
        return f"> {result.strip()}" if result else None
    if bt == "image":
        return None
    return None


def _compact_json(obj: object) -> str:
    """Compact a JSON-serializable object as a short string."""
    import json
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) > 500:
        s = s[:500] + "..."
    return s
