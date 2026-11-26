"""Language-model backends used by the agent.

Two interchangeable backends implement the same two methods:

* ``ClaudeLLM`` calls Claude through the official ``anthropic`` SDK and asks for JSON that
  matches a schema (structured outputs). Used when an Anthropic API key is configured.
* ``OfflineLLM`` is a deterministic, rule-based stand-in that builds its answers *only* from
  the retrieved knowledge-base chunks. It lets the whole system (and the evaluation suite)
  run reproducibly with no network or API key.

Both backends receive the same retrieved chunks, so retrieval changes what either one says.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Protocol

from pawpal_system import Pet

from .guardrails import applicability_issue
from .knowledge import RetrievedChunk, tokenize

log = logging.getLogger(__name__)

GENERIC_TERMS = frozenset({"dog", "cat", "rabbit", "bird", "pet", "puppy", "kitten", "my"})
CLAUDE_MODEL =os.environ.get("PAWPAL_CLAUDE_MODEL", "claude-opus-5-5")


class LLMError(RuntimeError):
    """Raised when a backend fails to produce a usable response."""


class LLM(Protocol):
    """Interface every backend implements."""

    name: str

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> dict:
        """Return {"answer": str, "sources": [chunk ids], "answerable": bool}."""

    def propose_tasks(self, pet: Pet, request: str, chunks: list[RetrievedChunk],
                      feedback: str | None = None) -> list[dict]:
        """Return a list of task dicts (description, time, duration_minutes, priority,
        frequency, source, rationale)."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered source list for a prompt."""
    blocks = []
    for r in chunks:
        c = r.chunk
        tasks = "\n".join(f"  suggested task: {t.description} | {t.time} | {t.duration_minutes} min"
                          f" | {t.priority} | {t.frequency}" for t in c.tasks)
        blocks.append(f"<source id=\"{c.id}\" species=\"{c.species}\" title=\"{c.label}\">\n"
                      f"{c.text}\n{tasks}\n</source>")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------------------
# Offline backend
# --------------------------------------------------------------------------------------
class OfflineLLM:
    """Deterministic extractive backend; no network required."""

    name = "offline"

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> dict:
        """Pick the retrieved sentences that best overlap the question and cite them."""
        # Species words match almost every sentence, so they don't count as evidence.
        q_terms = set(tokenize(question)) - GENERIC_TERMS
        scored = []
        for rank, r in enumerate(chunks):
            for i, sentence in enumerate(re.split(r"(?<=[.!?])\s+", r.chunk.text)):
                overlap = len(q_terms & set(tokenize(sentence)))
                if overlap:
                    # Prefer high overlap, then higher-ranked chunks, then earlier sentences.
                    scored.append((-overlap, rank, i, sentence.strip(), r.chunk.id))
        if not scored:
            return {"answer": "", "sources": [], "answerable": False}
        top_overlap = -min(scored)[0]
        best = [s for s in sorted(scored) if -s[0] >= max(1, top_overlap * 0.6)][:3]
        best.sort(key=lambda s: (s[1], s[2]))  # restore reading order
        sources = list(dict.fromkeys(s[4] for s in best))
        return {"answer": " ".join(s[3] for s in best), "sources": sources, "answerable": True}

    def propose_tasks(self, pet: Pet, request: str, chunks: list[RetrievedChunk],
                      feedback: str | None = None) -> list[dict]:
        """Turn TASK templates from relevant retrieved chunks into task proposals."""
        proposals: list[dict] = []
        seen: set[str] = set()
        for r in chunks:
            c = r.chunk
            if c.species not in ("all", pet.species.lower()):
                continue  # never borrow another species' routine
            for t in c.tasks:
                if t.description.lower() in seen:
                    continue
                if applicability_issue(c.id, t.description, pet, request):
                    continue  # the agent would reject it anyway; don't propose it
                seen.add(t.description.lower())
                rationale = re.split(r"(?<=[.!?])\s+", c.text)[0]
                proposals.append({
                    "description": t.description, "time": t.time,
                    "duration_minutes": t.duration_minutes, "priority": t.priority,
                    "frequency": t.frequency, "source": c.id, "rationale": rationale,
                })
        return proposals


# --------------------------------------------------------------------------------------
# Claude backend
# --------------------------------------------------------------------------------------
ANSWER_SYSTEM = """You are the PawPal+ pet-care assistant.
Answer the owner's question using ONLY the numbered sources provided. Cite the ids of the
sources you used in "sources". If the sources do not contain the answer, set "answerable"
to false and say you don't know. Never invent medication names or doses; for anything that
sounds like a health problem, tell the owner to contact a veterinarian. Keep answers under
90 words and write in plain language."""

PLAN_SYSTEM = """You are the PawPal+ care planner. Propose concrete daily care tasks for one pet.
Rules:
- Base every task on the provided sources and put the id of the source it came from in "source".
- Only use sources for the pet's species or for "all" species.
- Times are 24-hour "HH:MM". Durations are whole minutes between 1 and 240.
- priority is one of high, medium, low (medication, food, water, and hygiene are high).
- frequency is one of once, daily, weekly.
- Do not add medications or vet visits unless the owner's request mentions them, and never
  include doses, human medications, or foods that are toxic to pets.
- Prefer times mentioned by the owner. Do not duplicate tasks the pet already has.
- Propose at most 8 tasks."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "sources", "answerable"],
    "additionalProperties": False,
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "time": {"type": "string"},
                    "duration_minutes": {"type": "integer"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "frequency": {"type": "string", "enum": ["once", "daily", "weekly"]},
                    "source": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["description", "time", "duration_minutes", "priority",
                             "frequency", "source", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}


class ClaudeLLM:
    """Backend that calls Claude with schema-constrained JSON output."""

    name = "claude"

    def __init__(self, client=None, model: str = CLAUDE_MODEL) -> None:
        """Create the backend; pass a client to inject a fake one in tests."""
        if client is None:
            import anthropic  # imported lazily so offline mode never needs the package

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def _call(self, system: str, user: str, schema: dict) -> dict:
        try:
            import anthropic
            api_errors: tuple = (anthropic.APIError,)
        except ImportError:  # only possible with an injected fake client
            api_errors = ()
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=4000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": schema}},
                # If a safety classifier declines, let the API retry on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except api_errors as exc:
            raise LLMError(f"Claude API error: {exc}") from exc
        log.info("Claude call: model=%s stop_reason=%s usage=%s", self.model,
                 response.stop_reason, getattr(response, "usage", None))
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude response was cut off (max_tokens)")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("Claude returned no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Claude returned invalid JSON: {exc}") from exc

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> dict:
        """Answer a question grounded in the retrieved chunks."""
        user = f"Sources:\n{format_context(chunks)}\n\nOwner's question: {question}"
        return self._call(ANSWER_SYSTEM, user, ANSWER_SCHEMA)

    def propose_tasks(self, pet: Pet, request: str, chunks: list[RetrievedChunk],
                      feedback: str | None = None) -> list[dict]:
        """Propose tasks grounded in the retrieved chunks."""
        existing = ", ".join(f"{t.description} at {t.time}" for t in pet.get_pending_tasks()) or "none"
        user = (f"Pet: {pet.name}, a {pet.age}-year-old {pet.species}.\n"
                f"Existing pending tasks: {existing}\n\n"
                f"Sources:\n{format_context(chunks)}\n\n"
                f"Owner's request: {request}")
        if feedback:
            user += f"\n\nReviewer feedback on your previous proposal (fix this): {feedback}"
        return self._call(PLAN_SYSTEM, user, PLAN_SCHEMA).get("tasks", [])


def get_llm(preference: str | None = None) -> LLM:
    """Pick a backend: 'claude', 'offline', or 'auto' (default; env PAWPAL_LLM).

    'auto' uses Claude when an API key is configured and the SDK is installed, otherwise the
    offline backend. Any failure to build the Claude client falls back to offline.
    """
    preference = (preference or os.environ.get("PAWPAL_LLM", "auto")).lower()
    if preference == "offline":
        return OfflineLLM()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if preference == "claude" or (preference == "auto" and has_key):
        try:
            return ClaudeLLM()
        except Exception as exc:  # missing package, bad config, ...
            log.warning("Could not create Claude backend (%s); using offline backend", exc)
    return OfflineLLM()
