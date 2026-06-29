"""Dreaming executor — orchestrates knowledge extraction from session transcripts.

Called from:
- ``save_snapshot`` (milestone trigger)
- Session teardown (session‑end trigger)
- Cron daemon ``/dream --sweep`` command
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from openharness.services.dreaming.transcript import load_compacted_transcript, load_full_transcript
from openharness.services.dreaming.prompt import build_dream_prompt

log = logging.getLogger(__name__)

# ── milestone constants ───────────────────────────────────────────────────

# Doubling milestones: 10, 20, 40, 80, then every 100 thereafter
_MILESTONES = {10, 20, 40, 80}
_CAP_INTERVAL = 100
_CHILD_TASK_TIMEOUT = 600  # seconds — large transcripts can take DeepSeek several minutes


# ── public api ─────────────────────────────────────────────────────────────

class DreamingExecutor:
    """Orchestrate a dream run for a single session."""

    def __init__(self, conn: Any, *, workspace: Path) -> None:
        self._conn = conn
        self._workspace = Path(workspace)

    # ── milestone detection ──────────────────────────────────────────────

    def next_milestone_for(self, message_count: int) -> int | None:
        """Return the next milestone *message_count* will cross, or None."""
        if message_count < 10:
            return None
        if message_count in _MILESTONES:
            return message_count
        if message_count < 80:
            # find next doubling milestone
            for m in sorted(_MILESTONES):
                if message_count > m and message_count <= m * 2:
                    # check if the count crosses it
                    prev = m
                    next_m = m * 2
                    if next_m in _MILESTONES and message_count >= next_m:
                        return next_m
                    # Actually simplify: check if message_count is between milestones
            return None
        # capped phase: every 100 after 80
        base = ((message_count - 80 + _CAP_INTERVAL - 1) // _CAP_INTERVAL) * _CAP_INTERVAL + 80
        if message_count >= base and (message_count - base) < _CAP_INTERVAL:
            return base
        # check if we just crossed
        last_milestone = 80 + ((message_count - 80) // _CAP_INTERVAL) * _CAP_INTERVAL
        if message_count >= last_milestone:
            # this was already triggered or is the current one
            pass
        return None

    def should_dream(self, message_count: int, last_dreamed_count: int) -> bool:
        """Return True if *message_count* has crossed a milestone since
        *last_dreamed_count*."""
        if message_count < 10:
            return False
        next_m = _next_milestone_after(last_dreamed_count)
        return next_m is not None and message_count >= next_m

    # ── session‑end trigger ──────────────────────────────────────────────

    def should_dream_on_end(self, message_count: int, last_dreamed_count: int) -> bool:
        """Return True if the session‑end trigger should fire.

        Triggers when the delta since last dream exceeds half the milestone
        interval (50 messages in the capped phase).
        """
        threshold = min(_CAP_INTERVAL // 2, 50)
        return (message_count - last_dreamed_count) >= threshold

    # ── main entry point ─────────────────────────────────────────────────

    async def run_for_session(
        self,
        session_id: str,
        *,
        cwd: str | Path,
        project_name: str,
        dream_run_id: int | None = None,
    ) -> dict[str, Any]:
        """Execute a full dream run for *session_id*.

        Returns a result dict with keys: ``status``, ``created``, ``updated``,
        ``deprecated``, ``reinforced``, ``errors``, ``summary``.
        """
        result: dict[str, Any] = {
            "status": "error",
            "created": 0,
            "updated": 0,
            "deprecated": 0,
            "reinforced": 0,
            "errors": [],
            "summary": "",
            "extractions_found": 0,
        }

        # 1. Create dream_run record
        if dream_run_id is None:
            dream_run_id = self._create_dream_run(session_id)

        # 1b. Advance high-water mark immediately so concurrent milestone
        # checks don't re-trigger while this dream is still running.
        self._update_dreamed_messages(session_id)

        # 2. Load transcript
        try:
            compacted = load_compacted_transcript(self._conn, session_id)
            full_text = load_full_transcript(self._conn, session_id)
        except Exception as exc:
            log.error("Failed to load transcript for %s: %s", session_id, exc)
            result["errors"].append(f"transcript load: {exc}")
            self._finish_dream_run(dream_run_id, "error", error=str(exc))
            return result

        # 3. Write full transcript to dream workspace
        transcript_dir = self._workspace / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        (transcript_dir / "full.txt").write_text(full_text, encoding="utf-8")

        # 4. Load prior extractions (same session only)
        prior_extractions = self._load_prior_extractions(session_id)

        # 5. Build prompt
        prompt_text = build_dream_prompt(
            compacted_transcript=compacted,
            prior_extractions=prior_extractions,
            project_name=project_name,
        )

        # 6. Spawn child process
        try:
            child_output = await self._spawn_child(prompt_text, cwd)
        except subprocess.TimeoutExpired:
            log.error("Dream child timed out for %s", session_id)
            result["errors"].append("child timeout")
            self._finish_dream_run(dream_run_id, "timeout")
            return result
        except Exception as exc:
            log.error("Dream child failed for %s: %s", session_id, exc)
            result["errors"].append(f"child spawn: {exc}")
            self._finish_dream_run(dream_run_id, "error", error=str(exc))
            return result

        # 7. Parse JSON response
        extraction_data = self._parse_response(child_output, result, dream_run_id=dream_run_id)

        # Steps 8‑10 (reinforce + extractions + embedding backfill) wrapped
        # in a timeout so stuck API calls don't leave orphaned running rows.
        try:
            async with asyncio.timeout(_CHILD_TASK_TIMEOUT):
                # 8. Cross‑session reinforcement
                if extraction_data:
                    await self._cross_session_reinforce(session_id, extraction_data, result)

                # 9. Process extractions
                if extraction_data:
                    self._process_extractions(session_id, extraction_data, result)

                # 10. Backfill embeddings for any newly-created units
                if extraction_data and result.get("created", 0) > 0:
                    try:
                        from openharness.auth.storage import load_credential
                        from openharness.services.embedding import EmbeddingService
                        api_key = load_credential("openai", "api_key")
                        if api_key:
                            emb_svc = EmbeddingService(
                                backend="openai",
                                model="text-embedding-3-small",
                                api_key=api_key,
                            )
                            await self._backfill_new_embeddings(session_id, emb_svc)
                    except Exception:
                        log.exception("Embedding backfill failed for session %s", session_id)
        except asyncio.TimeoutError:
            log.error("Dream post‑processing timed out for session %s", session_id)
            result["errors"].append("post‑processing timeout")
            self._finish_dream_run(dream_run_id, "timeout")
            return result
        except Exception:
            log.exception("Dream post‑processing crashed for session %s", session_id)
            result["errors"].append("post‑processing error")
            self._finish_dream_run(dream_run_id, "error")
            return result

        # 11. Record result
        extractions_applied = (
            result["created"]
            + result["updated"]
            + result["deprecated"]
            + result["reinforced"]
        )
        self._finish_dream_run(
            dream_run_id,
            "completed",
            extractions_found=result["extractions_found"],
            extractions_applied=extractions_applied,
        )

        log.info(
            "Dream run %d completed: %d extractions found, "
            "%d created, %d updated, %d reinforced, %d deprecated, "
            "%d duplicates, %d errors",
            dream_run_id,
            result["extractions_found"],
            result["created"],
            result["updated"],
            result["reinforced"],
            result["deprecated"],
            result["extractions_found"] - extractions_applied - len(result["errors"]),
            len(result["errors"]),
        )

        result["status"] = "completed"
        return result

    # ── internal helpers ─────────────────────────────────────────────────

    def _heal_stuck_runs(self) -> None:
        """On startup, mark every 'running' row as 'error'.

        Any 'running' row at startup is necessarily orphaned — the gateway
        process that spawned the child died, so the child is dead too.
        """
        try:
            cur = self._conn.cursor()
            cur.execute(
                """UPDATE oh_dream_runs
                   SET finished_at = now(),
                       status = 'error',
                       error_message = 'healed on startup: previous gateway instance crashed'
                   WHERE status = 'running'""",
            )
            healed = cur.rowcount
            if healed:
                self._conn.commit()
                log.warning(
                    "Healed %d stuck dream run(s) on startup (running -> error)",
                    healed,
                )
            else:
                self._conn.rollback()
        except Exception:
            log.exception("Failed to heal stuck dream runs on startup")

    def _create_dream_run(self, session_id: str) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO oh_dream_runs (status, session_id) VALUES ('running', %s) RETURNING run_id",
            (session_id,),
        )
        run_id = cur.fetchone()[0]
        self._conn.commit()
        return run_id

    def _finish_dream_run(
        self,
        run_id: int,
        status: str,
        *,
        extractions_found: int = 0,
        extractions_applied: int = 0,
        error: str | None = None,
    ) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE oh_dream_runs
               SET finished_at = now(), status = %s,
                   knowledge_extracted = %s,
                   extractions_found = %s,
                   error_message = %s
               WHERE run_id = %s""",
            (status, extractions_applied, extractions_found, error, run_id),
        )
        self._conn.commit()

    def _load_prior_extractions(self, session_id: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            """SELECT content_hash, knowledge_type, title, content, confidence
               FROM oh_knowledge
               WHERE source_session_id = %s AND archived = false
               ORDER BY updated_at DESC""",
            (session_id,),
        )
        return [
            {
                "content_hash": row[0],
                "knowledge_type": row[1],
                "title": row[2],
                "content": row[3],
                "confidence": row[4],
            }
            for row in cur.fetchall()
        ]

    async def _spawn_child(self, prompt_text: str, cwd: str | Path) -> str:
        """Spawn `ohmo --print-file` and return stdout."""
        # Write prompt to temp file (avoids Windows 32K command-line limit)
        transcript_dir = self._workspace / "transcript"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = transcript_dir / "prompt.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "ohmo",
            "--print-file",
            str(prompt_file),
            "--max-turns", "20",
            "--workspace",
            str(self._workspace),
            "--cwd",
            str(Path(cwd).resolve()),
            "--denied-tools",
            "bash,write_file,edit_file,notebook_edit",
        ]

        log.info("Dream child spawn: cwd=%s workspace=%s", cwd, self._workspace)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=_CHILD_TASK_TIMEOUT
            )
        except asyncio.TimeoutError:
            process.kill()
            raise subprocess.TimeoutExpired(cmd, _CHILD_TASK_TIMEOUT)

        if process.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"child exit={process.returncode}: {err_text}")

        return stdout.decode("utf-8", errors="replace")

    def _parse_response(
        self, child_output: str, result: dict[str, Any], dream_run_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Parse child output as JSON.  Returns None on failure.

        Saves raw output to the dream workspace under transcript/response.txt
        for post-mortem debugging when parsing fails.
        """
        # Save raw output before any parsing (debuggability)
        try:
            transcript_dir = self._workspace / "transcript"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"-{dream_run_id}" if dream_run_id else ""
            (transcript_dir / f"response{suffix}.txt").write_text(
                child_output, encoding="utf-8"
            )
        except Exception:
            pass  # non-critical

        text = child_output.strip()
        # Extract JSON block — model may output analysis text before/after
        # Look for ```json ... ``` fences first
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        # If still no JSON, try to find the outermost braces
        if not text.startswith("{"):
            brace_start = text.find("{")
            if brace_start >= 0:
                brace_end = text.rfind("}") + 1
                if brace_end > brace_start:
                    text = text[brace_start:brace_end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            log.error("Failed to parse dream child JSON: %s", exc)
            result["errors"].append(f"json parse: {exc}")
            return None

        if not isinstance(data, dict) or "extractions" not in data:
            result["errors"].append("missing 'extractions' key in response")
            log.error("Dream child response missing 'extractions' key")
            return None

        extractions = data.get("extractions", [])
        log.info(
            "Dream child parsed: %d extractions in response",
            len(extractions) if isinstance(extractions, list) else 0,
        )
        return data

    async def _cross_session_reinforce(
        self,
        session_id: str,
        data: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """For each 'create' extraction, check if similar knowledge exists
        across all sessions via embedding similarity.  If a match is found
        (cosine similarity >= 0.60), convert the action to 'reinforce' so
        the existing unit gets new evidence rather than creating a duplicate."""
        try:
            from openharness.services.knowledge_store import get_knowledge_store
        except ImportError:
            return
        store = get_knowledge_store()
        if store is None:
            return

        extractions = data.get("extractions", [])
        for i, unit in enumerate(extractions):
            if not isinstance(unit, dict):
                continue
            if unit.get("action", "create") != "create":
                continue

            unit_type = unit.get("type", "FACT")
            title = unit.get("title", "")
            content = unit.get("content", "")
            query_text = f"{title}: {content}"

            try:
                matches = await store.recall(
                    query_text,
                    top_k=3,
                    types=[unit_type],
                    min_confidence=0.3,
                    threshold=0.60,
                )
            except Exception:
                continue

            if matches and matches[0].similarity and matches[0].similarity >= 0.60:
                best = matches[0]
                if best.source_session_id != session_id:
                    # Found similar knowledge from another session — reinforce
                    unit["action"] = "reinforce"
                    unit["id"] = best.id
                    log.info(
                        "Cross‑session reinforce: '%s' matches unit %d "
                        "(similarity=%.2f, from session %s)",
                        title, best.id, best.similarity, best.source_session_id,
                    )

    def _process_extractions(
        self,
        session_id: str,
        data: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Iterate extractions and upsert into oh_knowledge."""
        extractions = data.get("extractions", [])
        result["summary"] = data.get("summary", "")
        result["extractions_found"] = len(extractions)
        total = len(extractions)

        for i, unit in enumerate(extractions):
            if not isinstance(unit, dict):
                result["errors"].append("non-dict extraction item")
                log.warning("Dream extraction %d/%d: REJECTED (non-dict)", i + 1, total)
                continue

            # Validate required fields
            source_turns = unit.get("source_turns", [])
            if not source_turns or not isinstance(source_turns, list):
                result["errors"].append(
                    f"rejected '{unit.get('title', '?')}': missing source_turns"
                )
                log.warning(
                    "Dream extraction %d/%d: REJECTED '%s' (missing source_turns)",
                    i + 1, total, unit.get("title", "?")[:80],
                )
                continue

            action = unit.get("action", "create")
            if action not in ("create", "update", "deprecate", "reinforce"):
                result["errors"].append(f"unknown action '{action}'")
                log.warning(
                    "Dream extraction %d/%d: REJECTED '%s' (unknown action '%s')",
                    i + 1, total, unit.get("title", "?")[:80], action,
                )
                continue

            title = unit.get("title", "?")[:80]
            try:
                disposition = None
                if action == "create":
                    disposition = self._upsert_create(session_id, unit, result)
                elif action == "update":
                    disposition = self._upsert_update(session_id, unit, result)
                elif action == "deprecate":
                    disposition = self._upsert_deprecate(unit, result)
                elif action == "reinforce":
                    disposition = self._upsert_reinforce(session_id, unit, result)

                if disposition:
                    log.info(
                        "Dream extraction %d/%d: '%s' -> %s",
                        i + 1, total, title, disposition,
                    )
            except Exception as exc:
                result["errors"].append(f"upsert error: {exc}")
                log.exception("Upsert failed for unit %s", title)

    def _compute_content_hash(self, content: str) -> str:
        """SHA-256 hex of normalized content."""
        import hashlib
        normalized = " ".join(content.split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def _backfill_new_embeddings(
        self, session_id: str, emb_svc: "EmbeddingService",
    ) -> int:
        """Generate embeddings for knowledge units that have NULL embedding.

        Called after _process_extractions to fill embeddings for new units
        inserted by the dreaming pipeline (which bypasses KnowledgeStore).
        Returns the number of rows backfilled.
        """
        cur = self._conn.cursor()
        cur.execute(
            """SELECT id, title, content FROM oh_knowledge
               WHERE embedding IS NULL AND archived = FALSE
               AND source_session_id = %s""",
            (session_id,),
        )
        null_rows = cur.fetchall()
        if not null_rows:
            return 0

        count = 0
        for kid, title, content in null_rows:
            text = f"{title}: {content}"
            try:
                vec = await emb_svc.embed_one(text)
            except Exception:
                log.exception("Failed to embed knowledge unit id=%s", kid)
                continue
            cur.execute(
                "UPDATE oh_knowledge SET embedding = %s WHERE id = %s",
                (vec, kid),
            )
            count += 1
        self._conn.commit()
        if count:
            log.info("Embedding backfill: %d rows for session %s", count, session_id)
        return count

    def _upsert_create(
        self, session_id: str, unit: dict[str, Any], result: dict[str, Any]
    ) -> str:
        content = unit.get("content", "")
        content_hash = self._compute_content_hash(content)
        evidence = json.dumps([{"session_id": session_id, "turns": unit["source_turns"]}])

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO oh_knowledge
               (knowledge_type, title, content, confidence, source, source_session_id,
                source_evidences, scope, content_hash)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
               ON CONFLICT (content_hash) WHERE NOT archived
               DO NOTHING""",
            (
                unit.get("type", "FACT"),
                unit.get("title", ""),
                content,
                unit.get("confidence", 0.3),
                "dream",
                session_id,
                evidence,
                unit.get("scope", "global"),
                content_hash,
            ),
        )
        if cur.rowcount:
            result["created"] += 1
            self._conn.commit()
            return "CREATED"
        else:
            # Duplicate — find which existing unit has this hash
            cur.execute(
                "SELECT id FROM oh_knowledge WHERE content_hash = %s AND NOT archived",
                (content_hash,),
            )
            row = cur.fetchone()
            existing_id = row[0] if row else "?"
            self._conn.commit()  # still commit (rollback not needed for DO NOTHING, but consistent)
            return f"DUPLICATE (content_hash matches id={existing_id})"

    def _upsert_update(
        self, session_id: str, unit: dict[str, Any], result: dict[str, Any]
    ) -> str:
        unit_id = unit.get("id")
        if not unit_id:
            result["errors"].append("update missing 'id'")
            return "ERROR (missing id)"

        source_turns = unit["source_turns"]
        new_evidence = json.dumps({"session_id": session_id, "turns": source_turns})

        cur = self._conn.cursor()
        cur.execute(
            """UPDATE oh_knowledge
               SET content = COALESCE(%s, content),
                   confidence = COALESCE(%s, confidence),
                   scope = COALESCE(%s, scope),
                   updated_at = now(),
                   source_evidences = source_evidences || %s::jsonb
               WHERE content_hash = %s AND archived = false""",
            (
                unit.get("content"),
                unit.get("confidence"),
                unit.get("scope"),
                new_evidence,
                unit_id,
            ),
        )
        if cur.rowcount:
            result["updated"] += 1
            self._conn.commit()
            return f"UPDATED (id={unit_id})"
        else:
            self._conn.commit()
            return f"SKIPPED (id={unit_id} not found)"

    def _upsert_deprecate(self, unit: dict[str, Any], result: dict[str, Any]) -> str:
        unit_id = unit.get("id")
        if not unit_id:
            result["errors"].append("deprecate missing 'id'")
            return "ERROR (missing id)"
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE oh_knowledge SET archived = true, updated_at = now() WHERE content_hash = %s",
            (unit_id,),
        )
        if cur.rowcount:
            result["deprecated"] += 1
            self._conn.commit()
            return f"DEPRECATED (id={unit_id})"
        else:
            self._conn.commit()
            return f"SKIPPED (id={unit_id} not found)"

    def _upsert_reinforce(
        self, session_id: str, unit: dict[str, Any], result: dict[str, Any]
    ) -> str:
        unit_id = unit.get("id")
        if not unit_id:
            result["errors"].append("reinforce missing 'id'")
            return "ERROR (missing id)"
        new_evidence = json.dumps(
            {"session_id": session_id, "turns": unit["source_turns"]}
        )
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE oh_knowledge
               SET source_evidences = source_evidences || %s::jsonb,
                   updated_at = now(),
                   confidence = COALESCE(%s, confidence)
               WHERE content_hash = %s AND archived = false""",
            (new_evidence, unit.get("confidence"), unit_id),
        )
        if cur.rowcount:
            result["reinforced"] += 1
            self._conn.commit()
            return f"REINFORCED (id={unit_id})"
        else:
            self._conn.commit()
            return f"SKIPPED (id={unit_id} not found)"

    def _update_dreamed_messages(self, session_id: str) -> None:
        """Set the high‑water mark to the current message count for this session.

        Uses COUNT directly to avoid message_id gaps (dream child sessions
        consume global id sequence but don't belong to this session_id).
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM oh_messages WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            cur.execute(
                """INSERT INTO oh_dreamed_messages (session_id, last_message_id)
                   VALUES (%s, %s)
                   ON CONFLICT (session_id) DO UPDATE SET last_message_id = %s,
                      dreamed_at = now()""",
                (session_id, row[0], row[0]),
            )
        self._conn.commit()


# ── milestone math helpers ────────────────────────────────────────────────

def _next_milestone_after(count: int) -> int | None:
    """Return the first milestone strictly greater than *count*."""
    for m in sorted(_MILESTONES):
        if count < m:
            return m
    # capped phase
    if count < 80:
        return 80
    base = 80 + ((count - 80) // _CAP_INTERVAL + 1) * _CAP_INTERVAL
    return base
