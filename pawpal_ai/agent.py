"""The PawPal+ care assistant: retrieval-augmented Q&A and an agentic care planner.

Planner loop (``CareAssistant.plan_care``)::

    1. PLAN      break the owner's request into care topics (feeding, exercise, ...)
    2. RETRIEVE  search the knowledge base once per topic
    3. PROPOSE   ask the LLM backend for tasks grounded in the retrieved sources
    4. VALIDATE  guardrails: schema, safety, grounding, applicability, duplicates
    5. ACT       place each task on a *copy* of the schedule; move it if it conflicts
    6. CHECK     self-critique: time budget, conflicts, topic coverage -> confidence score
    7. REVISE    if a topic is uncovered, retrieve more and re-propose with feedback (once)

Nothing touches the real schedule until a human approves the plan (``apply_plan``).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import date

from pawpal_system import Owner, Pet, Scheduler, Task

from . import guardrails as g
from .knowledge import Retriever, RetrievedChunk
from .llm import LLM, LLMError, OfflineLLM, get_llm

log = logging.getLogger(__name__)

# Care topics: keywords that signal the topic in a request, and the query used to retrieve it.
TOPICS: dict[str, tuple[tuple[str, ...], str]] = {
    "feeding": (("feed", "food", "meal", "eat", "breakfast", "dinner", "diet", "hay", "greens"),
                "feeding meals diet food"),
    "water": (("water", "hydrat", "drink"), "fresh water bowl hydration"),
    "exercise": (("walk", "exercise", "run", "active", "energy", "roam", "out of cage"),
                 "exercise walks activity free-roam"),
    "enrichment": (("play", "toy", "enrich", "bored", "train", "social"),
                   "play enrichment toys training social"),
    "hygiene": (("litter", "clean", "cage", "potty", "liner", "bedding"),
                "litter box cage cleaning hygiene"),
    "grooming": (("groom", "brush", "nail", "claw", "coat", "fur", "teeth", "shed"),
                 "grooming brushing nails coat"),
    "medication": (g.MEDICATION_WORDS, "giving medication on schedule flea tick prevention"),
    "vet": (g.VET_WORDS, "vet checkup vaccines"),
}
ESSENTIALS = {
    "dog": ["feeding", "water", "exercise", "enrichment"],
    "cat": ["feeding", "water", "hygiene", "enrichment"],
    "rabbit": ["feeding", "water", "exercise", "hygiene"],
    "bird": ["feeding", "enrichment", "hygiene"],
}
# Scores below this mean the knowledge base probably does not cover the question.
MIN_RELEVANCE = 0.12
MAX_ITERATIONS = 2
PLAN_CONFIDENCE_FLOOR = 0.6  # below this the UI tells the owner to review carefully


@dataclass
class Answer:
    """Result of a grounded question-answer call."""

    question: str
    text: str
    sources: list[RetrievedChunk]
    confidence: float
    status: str  # "answered" | "refused" | "emergency" | "error"
    backend: str
    notes: list[str] = field(default_factory=list)


@dataclass
class ProposedTask:
    """A validated task waiting for the owner's approval."""

    task: Task
    source: str
    rationale: str
    notes: list[str] = field(default_factory=list)


@dataclass
class PlanResult:
    """Everything the planner decided, and why."""

    pet_name: str
    request: str
    day: date
    topics: list[str]
    accepted: list[ProposedTask]
    rejected: list[tuple[str, str]]  # (description, reason)
    sources: list[RetrievedChunk]
    confidence: float
    checks: dict[str, bool]
    trace: list[str]
    backend: str
    iterations: int
    status: str = "ok"  # "ok" | "emergency" | "blocked" | "error"

    @property
    def needs_review(self) -> bool:
        """True when confidence is low enough that the owner should look closely."""
        return self.status == "ok" and self.confidence < PLAN_CONFIDENCE_FLOOR


class CareAssistant:
    """Retrieval-augmented assistant over a PawPal+ owner's pets and schedule."""

    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None) -> None:
        """Build the assistant; defaults load the knowledge base and auto-pick a backend."""
        self.retriever = retriever or Retriever.from_folder()
        self.llm = llm or get_llm()
        self.fallback = OfflineLLM()
        log.info("CareAssistant ready (backend=%s, %d chunks)", self.llm.name,
                 len(self.retriever.chunks))

    # ----------------------------------------------------------------------------------
    # Question answering
    # ----------------------------------------------------------------------------------
    def answer_question(self, question: str, species: str | None = None) -> Answer:
        """Answer a pet-care question using only retrieved knowledge-base sources."""
        try:
            question = g.validate_text_input(question, "question")
        except g.InputError as exc:
            log.warning("Rejected question: %s", exc)
            return Answer(question or "", str(exc), [], 0.0, "error", "guardrail")

        if g.detect_emergency(question):
            log.warning("Emergency detected in question: %r", question)
            sources = self.retriever.search("emergency warning signs toxic poisoning", k=2)
            return Answer(question, g.EMERGENCY_MESSAGE, sources, 1.0, "emergency", "guardrail")

        if not g.is_pet_related(question):
            log.info("Refusing off-topic question: %r", question)
            return Answer(question, "I can only help with pet-care questions (feeding, exercise, "
                          "grooming, medication schedules, safety).", [], 0.0, "refused", "guardrail")

        sources = self.retriever.search(question, species=species, k=3)
        if not sources or sources[0].score < MIN_RELEVANCE:
            log.info("Refusing: best score %.3f below %.2f",
                     sources[0].score if sources else 0.0, MIN_RELEVANCE)
            return Answer(question, "I don't have reliable information about that in my pet-care "
                          "knowledge base. Please ask your veterinarian.", sources, 0.0,
                          "refused", "guardrail")

        notes: list[str] = []
        backend = self.llm.name
        try:
            result = self.llm.answer(question, sources)
        except LLMError as exc:
            log.error("Backend %s failed (%s); falling back to offline", self.llm.name, exc)
            notes.append(f"{self.llm.name} backend failed, used offline fallback")
            result, backend = self.fallback.answer(question, sources), self.fallback.name

        valid_ids = {r.chunk.id for r in sources}
        cited = [s for s in result.get("sources", []) if s in valid_ids]
        dropped = set(result.get("sources", [])) - valid_ids
        if dropped:
            notes.append(f"removed citations not in the retrieved sources: {sorted(dropped)}")
        if not result.get("answerable") or not result.get("answer", "").strip():
            return Answer(question, "The knowledge base doesn't answer that directly. Please ask "
                          "your veterinarian.", sources, 0.2, "refused", backend, notes)
        if not cited:
            notes.append("answer had no valid citations")

        retrieval_strength = min(1.0, sources[0].score / 0.6)
        confidence = round(0.6 * retrieval_strength + 0.4 * (1.0 if cited else 0.0), 2)
        used = [r for r in sources if r.chunk.id in cited] or sources
        log.info("Answered %r with %s (confidence %.2f)", question, cited, confidence)
        return Answer(question, result["answer"].strip(), used, confidence, "answered", backend, notes)

    # ----------------------------------------------------------------------------------
    # Agentic care planner
    # ----------------------------------------------------------------------------------
    def plan_topics(self, pet: Pet, request: str) -> list[str]:
        """Step 1: decide which care topics the request is about."""
        topics = [name for name, (words, _) in TOPICS.items() if g.mentions(request, words)]
        if not topics or g.mentions(request, ("routine", "schedule", "plan", "everything", "daily")):
            for name in ESSENTIALS.get(pet.species.lower(), ["feeding", "water"]):
                if name not in topics:
                    topics.append(name)
        return topics

    def _retrieve(self, pet: Pet, request: str, topics: list[str], k: int) -> list[RetrievedChunk]:
        """Step 2: one search per topic (plus life stage), merged and de-duplicated."""
        stage = g.life_stage(pet)
        queries = [f"{TOPICS[t][1]} {pet.species}" for t in topics]
        if stage != "adult":
            queries.append({"young": "puppy kitten young meals potty",
                            "senior": "senior pets gentle walks"}[stage])
        merged: dict[str, RetrievedChunk] = {}
        for q in queries:
            for r in self.retriever.search(q, species=pet.species, k=k, min_score=0.05):
                if r.chunk.id not in merged or r.score > merged[r.chunk.id].score:
                    merged[r.chunk.id] = r
        return sorted(merged.values(), key=lambda r: r.score, reverse=True)

    def plan_care(self, owner: Owner, pet_name: str, request: str,
                  day: date | None = None) -> PlanResult:
        """Run the plan -> retrieve -> propose -> validate -> act -> check loop."""
        day = day or date.today()
        trace: list[str] = []

        def step(msg: str) -> None:
            trace.append(msg)
            log.info("[plan %s] %s", pet_name, msg)

        def fail(msg: str, status: str = "error") -> PlanResult:
            step(msg)
            return PlanResult(pet_name, request or "", day, [], [], [], [], 0.0, {}, trace,
                              "guardrail", 0, status)

        try:
            request = g.validate_text_input(request)
        except g.InputError as exc:
            return fail(f"Input rejected: {exc}")
        pet = owner.get_pet(pet_name)
        if pet is None:
            return fail(f"Input rejected: no pet named {pet_name!r}")
        if g.detect_emergency(request):
            return fail(f"Emergency language detected; no plan generated. {g.EMERGENCY_MESSAGE}",
                        "emergency")

        if reason := g.unsafe_reason(request, pet.species):
            return fail(f"Request blocked by safety guardrail: it {reason}. No plan generated; "
                        "please talk to your veterinarian.", "blocked")

        topics = self.plan_topics(pet, request)
        existing_descs = [t.description for t in pet.tasks if t.due_date == day and not t.completed]
        step(f"PLAN: topics for {pet.name} ({pet.species}, {g.life_stage(pet)}): {', '.join(topics)}")

        backend = self.llm.name
        feedback: str | None = None
        k = 2
        accepted: list[ProposedTask] = []
        rejected: list[tuple[str, str]] = []
        sources: list[RetrievedChunk] = []
        iteration = 0
        proposed_total = 0
        sandbox = copy.deepcopy(owner)  # ACT happens here, never on the real owner
        sandbox_pet = sandbox.get_pet(pet.name)
        scheduler = Scheduler(sandbox)

        while iteration < MAX_ITERATIONS:
            iteration += 1
            sources = self._retrieve(pet, request, topics, k)
            step(f"RETRIEVE (round {iteration}): " + ", ".join(f"{r.chunk.id}={r.score}" for r in sources))
            if not sources:
                return fail("No relevant knowledge-base sources found; refusing to guess.")

            try:
                raw = self.llm.propose_tasks(pet, request, sources, feedback)
            except LLMError as exc:
                step(f"PROPOSE: {self.llm.name} failed ({exc}); using offline fallback")
                raw, backend = self.fallback.propose_tasks(pet, request, sources, feedback), "offline (fallback)"
            if not isinstance(raw, list):
                raw = []
            proposed_total += len(raw)
            step(f"PROPOSE: {len(raw)} candidate task(s) from {backend}")

            new = self._validate_and_place(raw, pet, sandbox_pet, scheduler, request, day,
                                           {r.chunk.id for r in sources}, accepted, rejected, step)
            accepted.extend(new)

            covered = self._covered_topics(accepted, topics, existing_descs)
            missing = [t for t in topics if t not in covered]
            if not missing or iteration == MAX_ITERATIONS:
                break
            feedback = f"No accepted task covers: {', '.join(missing)}."
            k += 2
            step(f"REVISE: {feedback} Retrying with wider retrieval (k={k}).")

        guardrail_rejections = len(rejected)  # budget drops (added below) aren't AI mistakes
        checks = self._self_check(owner, scheduler, pet, accepted, rejected, topics, day, step,
                                  existing_descs)
        grounded = 1 - guardrail_rejections / max(1, proposed_total)
        coverage = len(self._covered_topics(accepted, topics, existing_descs)) / max(1, len(topics))
        retrieval = min(1.0, (sum(r.score for r in sources[:3]) / max(1, len(sources[:3]))) / 0.5)
        constraints = 1.0 if checks["no_conflicts"] and checks["within_budget"] else 0.5
        confidence = round(0.3 * retrieval + 0.2 * grounded + 0.3 * coverage + 0.2 * constraints, 2)
        if not accepted:
            confidence = 0.0
        step(f"CHECK: confidence={confidence} (retrieval={retrieval:.2f}, grounded={grounded:.2f}, "
             f"coverage={coverage:.2f}, constraints={constraints})")
        return PlanResult(pet.name, request, day, topics, accepted, rejected, sources, confidence,
                          checks, trace, backend, iteration)

    def _validate_and_place(self, raw, pet, sandbox_pet, scheduler, request, day, source_ids,
                            accepted, rejected, step) -> list[ProposedTask]:
        """Steps 4-5: guardrail each proposal, then place it on the sandbox schedule."""
        new: list[ProposedTask] = []
        existing = [t.description for t in sandbox_pet.tasks if t.due_date == day and not t.completed]
        seen_dups: set[str] = set()
        for item in raw:
            desc = str(item.get("description", "")).strip()[:80]

            def reject(reason: str) -> None:
                rejected.append((desc or "(blank)", reason))
                step(f"VALIDATE: rejected '{desc}': {reason}")

            if not desc:
                reject("missing description")
                continue
            if dup := g.is_duplicate(desc, existing):
                if dup.lower() != desc.lower() and desc not in seen_dups:
                    seen_dups.add(desc)
                    step(f"VALIDATE: skipped '{desc}' (duplicates existing task '{dup}')")
                continue  # already scheduled by the owner or an earlier round
            source = str(item.get("source", ""))
            if source not in source_ids:
                reject(f"not grounded: source '{source}' was not retrieved")
                continue
            if reason := g.unsafe_reason(desc, pet.species):
                reject(f"unsafe: {reason}")
                continue
            if reason := g.applicability_issue(source, desc, pet, request):
                reject(f"not applicable: {reason}")
                continue
            try:
                task = Task(desc, str(item.get("time", "")), g.clamp_duration(item.get("duration_minutes", 0)),
                            str(item.get("priority", "medium")), str(item.get("frequency", "once")),
                            due_date=day)
            except (TypeError, ValueError) as exc:
                reject(f"invalid fields: {exc}")
                continue

            notes = []
            sandbox_pet.add_task(task)
            if any(task.description in c for c in scheduler.detect_conflicts(day)):
                sandbox_pet.tasks.remove(task)
                slot = (scheduler.find_next_available_slot(task.duration_minutes, day, day_start=task.time)
                        or scheduler.find_next_available_slot(task.duration_minutes, day))
                if slot is None:
                    reject("no free time slot left in the day")
                    continue
                notes.append(f"moved from {task.time} to {slot} to avoid a conflict")
                step(f"ACT: '{desc}' conflicted at {task.time}; moved to {slot}")
                task.time = slot
                sandbox_pet.add_task(task)
            else:
                step(f"ACT: placed '{desc}' at {task.time} ({task.duration_minutes} min)")
            existing.append(desc)
            new.append(ProposedTask(task, source, str(item.get("rationale", ""))[:200], notes))
        return new

    @staticmethod
    def _covered_topics(accepted: list[ProposedTask], topics: list[str],
                        existing: list[str] = ()) -> set[str]:
        """Topics covered by a proposed task or by a task the pet already has that day."""
        texts = [f"{p.task.description} {p.source}" for p in accepted] + list(existing)
        return {t for t in topics for text in texts if g.mentions(text, TOPICS[t][0])}

    def _self_check(self, owner, scheduler, pet, accepted, rejected, topics, day, step,
                    existing_descs=()) -> dict[str, bool]:
        """Step 6: verify the sandbox plan; drop low-priority extras if over the time budget."""
        budget = owner.available_minutes

        def pending_minutes() -> int:
            return sum(t.duration_minutes for _, t in scheduler.filter_tasks(completed=False, day=day))

        sandbox_pet = scheduler.owner.get_pet(pet.name)
        new_minutes = sum(p.task.duration_minutes for p in accepted)
        step(f"CHECK: budget {budget} min; existing tasks use {pending_minutes() - new_minutes} min, "
             f"new tasks add {new_minutes} min")
        while pending_minutes() > budget:
            optional = [p for p in accepted if p.task.priority != "high"]
            if not optional:
                break
            victim = min(optional, key=lambda p: (p.task.priority_rank(), -p.task.start_minutes()))
            accepted.remove(victim)
            sandbox_pet.tasks.remove(victim.task)
            rejected.append((victim.task.description, f"over the {budget}-minute daily budget "
                                                      f"({victim.task.priority} priority dropped first)"))
            step(f"CHECK: over budget, dropped '{victim.task.description}' ({victim.task.priority})")

        new_descs = {p.task.description for p in accepted}
        conflicts = [c for c in scheduler.detect_conflicts(day) if any(d in c for d in new_descs)]
        missing = [t for t in topics if t not in self._covered_topics(accepted, topics, existing_descs)]
        checks = {
            "no_conflicts": not conflicts,
            "within_budget": pending_minutes() <= budget,
            "all_topics_covered": not missing,
            "all_tasks_cited": all(p.source for p in accepted),
        }
        step(f"CHECK: {checks}" + (f" uncovered={missing}" if missing else ""))
        return checks

    @staticmethod
    def apply_plan(owner: Owner, plan: PlanResult, approved: list[int] | None = None) -> list[Task]:
        """Human-in-the-loop step: add the approved proposals (all by default) to the real pet."""
        pet = owner.get_pet(plan.pet_name)
        if pet is None:
            raise ValueError(f"No pet named {plan.pet_name!r}")
        chosen = [p for i, p in enumerate(plan.accepted) if approved is None or i in approved]
        added = []
        for p in chosen:
            task = copy.deepcopy(p.task)
            pet.add_task(task)
            added.append(task)
        log.info("Applied %d approved task(s) to %s", len(added), pet.name)
        return added
