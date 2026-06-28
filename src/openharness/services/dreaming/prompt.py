"""Build the dreaming prompt from a compacted transcript and prior extractions."""

from __future__ import annotations

import json
from typing import Any

_CORE_PROMPT = """You are an offline knowledge extraction agent.  You are reviewing the complete
transcript of an ohmo coding session and must extract durable knowledge units.

A knowledge unit is something true about the user, the project, or a pattern
that should persist beyond this session.

Extract TWO kinds of knowledge:

1. DISCRETE FACTS — stated explicitly in a message.
2. EMERGENT PATTERNS — demonstrated across multiple turns.  Not stated once,
   but visible in what was accepted vs. rejected, in coding style choices, in
   workflow rhythms.

For emergent patterns: after reading the full transcript, reflect on the
narrative arc.  What does this session reveal about how the user works?

The transcript below may show `[TRUNCATED]` markers where long assistant messages
or tool results were compacted to save context.  The full uncompacted transcript
is available at `transcript/full.txt` — use `read_file` to inspect any truncated
section that seems important.

Classify each unit into exactly one type:
- PREFERENCE: the user expressed a preference (tools, style, workflow)
- FACT: a fact about the project, codebase, or environment
- PATTERN: a recurring pattern or convention the agent should follow
- DECISION: a design or architecture decision the user made
- RELATIONSHIP: how files, systems, or concepts relate to each other
- CAVEAT: a constraint, limitation, or "watch out for" note
- SELF_IMPROVEMENT: a behavior pattern the agent observed in its own actions
  that could be more effective or efficient. Must cite specific turns showing
  the suboptimal pattern. Example: "In turns 15-17, the agent ran three
  separate grep commands before using lsp, which would have been faster."

For scope, decide per unit:
- "global" for user preferences, environment facts, and self‑improvement items
  that apply regardless of project.
- "project:<name>" for codebase‑specific knowledge where <name> is the
  project name from the session header. Most FACT, PATTERN, DECISION,
  RELATIONSHIP, and CAVEAT units will be project‑scoped.

For each unit, output a JSON object:
{
  "action": "create" | "update" | "deprecate" | "reinforce",
  "id": "<content_hash from oh_knowledge>" (for update/deprecate/reinforce),
  "type": "PREFERENCE",
  "title": "short, searchable title",
  "content": "the knowledge itself, 1-3 sentences",
  "confidence": 0.0-1.0,
  "rationale": "why this confidence level",
  "scope": "global" | "project:<project_name>",
  "source_turns": [3, 7]   // REQUIRED for every unit. Turn numbers that support this extraction.
                            // Without this, the unit WILL be rejected.
}

IMPORTANT: source_turns is not optional. If you cannot cite specific turns,
do not extract the unit. For SELF_IMPROVEMENT, this is especially critical —
no uncited self‑criticism.

New action: "reinforce" — for an existing unit where you found additional
evidence in this session but are not changing the content or confidence.
Append the new turn numbers to the unit's source_evidences. If the new
evidence raises confidence, use "update" with the new confidence instead.

Cross‑session reinforcement is deferred to a future stage.  For now, only
same‑session prior extractions are provided below.

Return: {"extractions": [...], "summary": "1-2 sentence summary of this dream run"}"""

_SELF_CORRECTION_TEMPLATE = """
PRIOR EXTRACTIONS FROM THIS SESSION
====================================
The following knowledge was extracted during earlier dream runs of this
same session.  Review each one against the full transcript now available:

- **Keep it** if still accurate → do not include it in output.
- **Update it** if context has changed → include with "action": "update",
  the updated confidence, and the rationale for the change.
- **Deprecate it** if proven wrong → include with "action": "deprecate".
- **Reinforce it** if you see additional confirming evidence → include with
  "action": "reinforce" and the new turn numbers in source_turns.

Do NOT reinforce without new evidence.  Do NOT create near‑duplicates of
existing units — use "reinforce" or "update" on the existing unit instead.

Cross‑session reinforcement is deferred to Stage 3 (embeddings enable
topical retrieval).  For now, only same‑session priors are provided.

EXISTING EXTRACTIONS (this session):
{priors_json}
"""


def build_dream_prompt(
    *,
    compacted_transcript: str,
    prior_extractions: list[dict[str, Any]] | None = None,
    project_name: str = "unknown",
) -> str:
    """Build the full dreaming prompt.

    *compacted_transcript* is the output of
    :func:`~openharness.services.dreaming.transcript.load_compacted_transcript`.

    *prior_extractions* are rows from ``oh_knowledge`` for the same session
    (collected by the executor).  Each dict should have at minimum:
    ``id``, ``type``, ``title``, ``content``, ``confidence``.
    """
    parts: list[str] = []

    parts.append(_CORE_PROMPT)

    # self‑correction block
    if prior_extractions:
        priors_for_prompt = []
        for row in prior_extractions:
            priors_for_prompt.append({
                "id": row.get("content_hash", row.get("id", "")),
                "type": row.get("knowledge_type", row.get("type", "FACT")),
                "title": row.get("title", ""),
                "content": row.get("content", ""),
                "confidence": row.get("confidence", 0.5),
            })
        parts.append(
            _SELF_CORRECTION_TEMPLATE.format(
                priors_json=json.dumps(priors_for_prompt, indent=2, ensure_ascii=False)
            )
        )

    # transcript
    parts.append("")
    parts.append("=" * 60)
    parts.append("SESSION TRANSCRIPT")
    parts.append("=" * 60)
    parts.append(compacted_transcript)

    return "\n".join(parts)
