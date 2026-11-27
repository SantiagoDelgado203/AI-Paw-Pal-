"""Tests for the PawPal+ AI layer: retrieval, guardrails, LLM backends, and the planning agent.

All tests run offline. Claude is exercised through a fake client so we can check the exact
request we send and how we handle bad responses, without network access or an API key.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from pawpal_ai import CareAssistant, ClaudeLLM, OfflineLLM, Retriever, get_llm
from pawpal_ai import guardrails as g
from pawpal_ai.knowledge import TaskTemplate, load_chunks, tokenize
from pawpal_ai.llm import LLMError
from pawpal_system import Owner, Pet, Task

DAY = date(2026, 1, 5)


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever.from_folder()


@pytest.fixture
def assistant(retriever) -> CareAssistant:
    return CareAssistant(retriever=retriever, llm=OfflineLLM())


@pytest.fixture
def owner() -> Owner:
    o = Owner("Jordan", available_minutes=240)
    mochi = Pet("Mochi", "dog", 3)
    luna = Pet("Luna", "cat", 5)
    o.add_pet(mochi)
    o.add_pet(luna)
    mochi.add_task(Task("Morning walk", "07:00", 30, "high", "daily", due_date=DAY))
    luna.add_task(Task("Flea medication", "12:00", 5, "high", "weekly", due_date=DAY))
    return o


class FakeLLM:
    """Scripted backend: returns canned proposals/answers and records what it was given."""

    name = "fake"

    def __init__(self, proposals=None, answer=None, fail=False):
        self.proposals = proposals or []
        self._answer = answer
        self.fail = fail
        self.feedback_seen: list[str | None] = []

    def answer(self, question, chunks):
        if self.fail:
            raise LLMError("boom")
        return self._answer

    def propose_tasks(self, pet, request, chunks, feedback=None):
        self.feedback_seen.append(feedback)
        if self.fail:
            raise LLMError("boom")
        return self.proposals


def proposal(desc="Dinner", time="17:30", minutes=10, priority="high", freq="daily",
             source="dogs#feeding"):
    return {"description": desc, "time": time, "duration_minutes": minutes, "priority": priority,
            "frequency": freq, "source": source, "rationale": "test"}


# --------------------------------------------------------------------------------------
# Knowledge base + retrieval
# --------------------------------------------------------------------------------------
def test_knowledge_base_loads_chunks_and_task_templates():
    chunks = load_chunks()
    ids = {c.id for c in chunks}
    assert {"dogs#exercise-and-walks", "cats#litter-box", "safety#emergency-warning-signs"} <= ids
    walks = next(c for c in chunks if c.id == "dogs#exercise-and-walks")
    assert TaskTemplate("Morning walk", "07:00", 30, "high", "daily") in walks.tasks
    assert "TASK:" not in walks.text  # machine lines are stripped from the prose


def test_malformed_task_line_is_skipped(tmp_path):
    (tmp_path / "x.md").write_text("---\ntitle: X\nspecies: dog\n---\n## Sec\nSome text.\n"
                                   "TASK: Broken | 07:00\nTASK: Good | 08:00 | 5 | low | daily\n")
    (chunk,) = load_chunks(tmp_path)
    assert [t.description for t in chunk.tasks] == ["Good"]


def test_missing_knowledge_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "nope")


def test_tokenize_drops_stopwords_and_stems_plurals():
    assert tokenize("How often should I walk the puppies?") == ["walk", "puppy"]


@pytest.mark.parametrize("query,species,expected", [
    ("How often should I scoop the litter box?", "cat", "cats#litter-box"),
    ("how many walks does my dog need", "dog", "dogs#exercise-and-walks"),
    ("what should my parrot eat", None, "birds#diet-food-and-water"),
    ("is it ok to bathe my bunny", None, "rabbits#grooming-and-hygiene"),
    ("which foods are toxic like chocolate", None, "safety#toxic-foods-and-substances"),
])
def test_retriever_finds_the_right_section(retriever, query, species, expected):
    top = [r.chunk.id for r in retriever.search(query, species=species, k=3)]
    assert expected in top


def test_species_boost_prefers_matching_species(retriever):
    cat_top = retriever.search("feeding meals", species="cat", k=1)[0]
    dog_top = retriever.search("feeding meals", species="dog", k=1)[0]
    assert cat_top.chunk.species == "cat" and dog_top.chunk.species == "dog"


def test_empty_or_unknown_query_returns_nothing(retriever):
    assert retriever.search("") == []
    assert retriever.search("quantum chromodynamics") == []


# --------------------------------------------------------------------------------------
# Guardrails
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["My dog ate chocolate!", "my cat is having a seizure",
                                  "he collapsed in the yard", "she can't breathe properly",
                                  "the puppy swallowed grapes"])
def test_emergencies_are_detected(text):
    assert g.detect_emergency(text)


@pytest.mark.parametrize("text", ["How often should I walk my dog?", "what toys do cats like",
                                  "Which foods are toxic to cats?"])
def test_normal_questions_are_not_emergencies(text):
    assert not g.detect_emergency(text)


@pytest.mark.parametrize("desc,species,fragment", [
    ("Chocolate treat", "dog", "toxic"),
    ("Give 200 mg ibuprofen", "dog", "human medication"),
    ("Give 5 mg pill", "cat", "dose"),
    ("Bathe the rabbit", "rabbit", "never be bathed"),
    ("Put lilies by the window", "cat", "lilies"),
])
def test_unsafe_task_descriptions_are_caught(desc, species, fragment):
    assert fragment in g.unsafe_reason(desc, species)


def test_safe_task_passes_and_bathing_a_dog_is_fine():
    assert g.unsafe_reason("Evening walk", "dog") is None
    assert g.unsafe_reason("Bath time", "dog") is None


def test_input_validation():
    with pytest.raises(g.InputError):
        g.validate_text_input("   ")
    with pytest.raises(g.InputError):
        g.validate_text_input("x" * (g.MAX_INPUT_CHARS + 1))
    assert g.validate_text_input("  hi  ") == "hi"


def test_life_stage():
    assert g.life_stage(Pet("P", "dog", 0)) == "young"
    assert g.life_stage(Pet("P", "dog", 4)) == "adult"
    assert g.life_stage(Pet("P", "dog", 9)) == "senior"
    assert g.life_stage(Pet("P", "cat", 9)) == "adult"


def test_applicability_blocks_unrequested_medication_and_wrong_life_stage():
    adult = Pet("Rex", "dog", 4)
    assert g.applicability_issue("medications#giving-medication-on-schedule",
                                 "Give prescribed medication", adult, "daily routine")
    assert g.applicability_issue("medications#giving-medication-on-schedule",
                                 "Give prescribed medication", adult, "Rex takes a pill at 8") is None
    assert g.applicability_issue("life_stages#senior-pets", "Gentle short walk", adult, "routine")
    assert g.applicability_issue("medications#vet-checkups-and-vaccines", "Vet checkup", adult,
                                 "plan his walks")


def test_duplicate_detection():
    assert g.is_duplicate("dinner", ["Dinner"]) == "Dinner"
    assert g.is_duplicate("Flea and tick prevention", ["Flea medication"]) == "Flea medication"
    assert g.is_duplicate("Morning play", ["Interactive play"]) is None


# --------------------------------------------------------------------------------------
# Question answering
# --------------------------------------------------------------------------------------
def test_answer_is_grounded_and_cited(assistant):
    ans = assistant.answer_question("How often should I scoop the litter box?", "cat")
    assert ans.status == "answered"
    assert "once a day" in ans.text
    assert ans.sources[0].chunk.id == "cats#litter-box"
    assert ans.confidence >= 0.7


def test_emergency_question_skips_the_llm(retriever):
    llm = FakeLLM(fail=True)  # would raise if called
    ans = CareAssistant(retriever, llm).answer_question("My dog ate chocolate an hour ago", "dog")
    assert ans.status == "emergency"
    assert "veterinarian" in ans.text


@pytest.mark.parametrize("q", ["What is the capital of France?", "best laptop for gaming",
                               "What time is it in Tokyo?"])
def test_off_topic_questions_are_refused(assistant, q):
    assert assistant.answer_question(q).status == "refused"


def test_pet_question_with_no_matching_words_is_refused(assistant):
    ans = assistant.answer_question("Should my cat get pet insurance?")
    assert ans.status == "refused"


def test_empty_question_is_an_error(assistant):
    assert assistant.answer_question("").status == "error"


def test_backend_failure_falls_back_to_offline(retriever):
    ans = CareAssistant(retriever, FakeLLM(fail=True)).answer_question("How often should I walk my dog?", "dog")
    assert ans.status == "answered" and ans.backend == "offline"
    assert any("fallback" in n for n in ans.notes)


def test_hallucinated_citations_are_removed(retriever):
    llm = FakeLLM(answer={"answer": "Twice a day.", "sources": ["made-up#source"], "answerable": True})
    ans = CareAssistant(retriever, llm).answer_question("How often should I walk my dog?", "dog")
    assert any("removed citations" in n for n in ans.notes)
    assert ans.confidence < 0.7  # no valid citation -> lower confidence


def test_model_saying_unanswerable_is_respected(retriever):
    llm = FakeLLM(answer={"answer": "", "sources": [], "answerable": False})
    assert CareAssistant(retriever, llm).answer_question("How often should I walk my dog?").status == "refused"


# --------------------------------------------------------------------------------------
# Planning agent
# --------------------------------------------------------------------------------------
def test_offline_plan_for_dog_is_grounded_conflict_free_and_skips_existing(assistant, owner):
    plan = assistant.plan_care(owner, "Mochi", "Plan a daily routine", DAY)
    descs = [p.task.description for p in plan.accepted]
    assert plan.status == "ok"
    assert "Morning walk" not in descs  # Mochi already has it
    assert {"Breakfast", "Dinner"} <= set(descs)
    assert all(p.source.startswith(("dogs#", "routines#", "life_stages#")) for p in plan.accepted)
    assert plan.checks["no_conflicts"] and plan.checks["within_budget"]
    assert plan.confidence >= 0.8


def test_plan_does_not_touch_real_schedule_until_applied(assistant, owner):
    before = len(owner.get_all_tasks())
    plan = assistant.plan_care(owner, "Mochi", "Plan a daily routine", DAY)
    assert len(owner.get_all_tasks()) == before
    added = CareAssistant.apply_plan(owner, plan, approved=[0])
    assert len(added) == 1 and len(owner.get_all_tasks()) == before + 1


def test_cat_plan_never_borrows_dog_tasks(assistant, owner):
    plan = assistant.plan_care(owner, "Luna", "daily routine", DAY)
    assert plan.accepted
    assert not any(p.source.startswith("dogs#") for p in plan.accepted)


def test_duplicate_flea_medication_is_skipped(assistant, owner):
    plan = assistant.plan_care(owner, "Luna", "Luna needs flea medication and play time", DAY)
    assert not any("Flea" in p.task.description for p in plan.accepted)
    assert any("duplicates existing task 'Flea medication'" in s for s in plan.trace)


def test_conflicting_task_is_moved(retriever, owner):
    llm = FakeLLM([proposal("Breakfast", "07:00", source="dogs#feeding")])  # clashes with walk
    plan = CareAssistant(retriever, llm).plan_care(owner, "Mochi", "feed him", DAY)
    (p,) = plan.accepted
    assert p.task.time == "07:30" and "moved" in p.notes[0]


def test_guardrails_reject_bad_proposals(retriever, owner):
    llm = FakeLLM([
        proposal("Dinner"),                                   # ok
        proposal("Snack", source="invented#source"),          # ungrounded
        proposal("Chocolate reward", source="dogs#feeding"),  # unsafe
        proposal("Lunch", time="25:99"),                      # invalid time
        proposal("Marathon walk", minutes=600),               # absurd duration
    ])
    plan = CareAssistant(retriever, llm).plan_care(owner, "Mochi", "feeding plan", DAY)
    assert [p.task.description for p in plan.accepted] == ["Dinner"]
    reasons = " | ".join(r for _, r in plan.rejected)
    for fragment in ("not grounded", "unsafe", "invalid fields", "outside"):
        assert fragment in reasons


def test_unrequested_flea_task_is_not_applicable(retriever, owner):
    llm = FakeLLM([proposal("Flea and tick prevention", "09:00", 5,
                            source="medications#flea-tick-and-heartworm-prevention")])
    plan = CareAssistant(retriever, llm).plan_care(owner, "Mochi", "Mochi takes medication", DAY)
    assert not plan.accepted
    assert "not applicable" in plan.rejected[0][1]


def test_over_budget_drops_low_priority_first(retriever, owner):
    owner.available_minutes = 60  # walk (30) + flea (5) already use 35
    llm = FakeLLM([proposal("Dinner", minutes=10), proposal("Play fetch", "16:30", 20, "low",
                                                           source="dogs#mental-enrichment-and-training")])
    plan = CareAssistant(retriever, llm).plan_care(owner, "Mochi", "dinner and play", DAY)
    assert [p.task.description for p in plan.accepted] == ["Dinner"]
    assert "budget" in plan.rejected[-1][1]
    assert plan.checks["within_budget"]


def test_revise_loop_sends_feedback_when_topic_uncovered(retriever, owner):
    llm = FakeLLM([proposal("Dinner")])  # never covers "exercise"
    plan = CareAssistant(retriever, llm).plan_care(owner, "Luna", "feeding and exercise", DAY)
    assert plan.iterations == 2
    assert llm.feedback_seen[0] is None and "exercise" in llm.feedback_seen[1]


def test_backend_failure_in_planner_falls_back(retriever, owner):
    plan = CareAssistant(retriever, FakeLLM(fail=True)).plan_care(owner, "Mochi", "meals", DAY)
    assert plan.backend == "offline (fallback)" and plan.accepted


@pytest.mark.parametrize("pet,request_text,status", [
    ("Ghost", "daily routine", "error"),
    ("Mochi", "", "error"),
    ("Mochi", "he ate chocolate and is shaking", "emergency"),
    ("Luna", "give her 50 mg of ibuprofen", "blocked"),
])
def test_planner_input_guardrails(assistant, owner, pet, request_text, status):
    plan = assistant.plan_care(owner, pet, request_text, DAY)
    assert plan.status == status and not plan.accepted


def test_life_stage_specific_tasks(assistant):
    o = Owner("Sam", 300)
    o.add_pet(Pet("Bean", "dog", 0))
    o.add_pet(Pet("Old Rex", "dog", 12))
    young = {p.task.description for p in assistant.plan_care(o, "Bean", "daily routine", DAY).accepted}
    senior = {p.task.description for p in assistant.plan_care(o, "Old Rex", "daily routine", DAY).accepted}
    assert "Potty break" in young and "Potty break" not in senior
    assert "Gentle short walk" in senior and "Gentle short walk" not in young


# --------------------------------------------------------------------------------------
# Claude backend (fake client) and backend selection
# --------------------------------------------------------------------------------------
class FakeClient:
    def __init__(self, text, stop_reason="end_turn"):
        self.calls = []
        response = SimpleNamespace(stop_reason=stop_reason, usage=None,
                                   content=[SimpleNamespace(type="text", text=text)])
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def test_claude_backend_sends_schema_and_parses_json(retriever):
    payload = {"answer": "Twice a day.", "sources": ["dogs#exercise-and-walks"], "answerable": True}
    client = FakeClient(json.dumps(payload))
    llm = ClaudeLLM(client=client)
    chunks = retriever.search("walks dog", species="dog", k=2)
    assert llm.answer("How often?", chunks) == payload
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "dogs#exercise-and-walks" in call["messages"][0]["content"]  # retrieved context is sent


def test_claude_backend_plan_includes_feedback(retriever):
    client = FakeClient(json.dumps({"tasks": [proposal()]}))
    tasks = ClaudeLLM(client=client).propose_tasks(Pet("Mochi", "dog", 3), "meals",
                                                   retriever.search("feeding", "dog"), "cover exercise")
    assert tasks == [proposal()]
    assert "cover exercise" in client.calls[0]["messages"][0]["content"]


@pytest.mark.parametrize("text,stop", [("not json", "end_turn"), ("{}", "refusal"), ("{", "max_tokens")])
def test_claude_backend_bad_responses_raise_llm_error(retriever, text, stop):
    with pytest.raises(LLMError):
        ClaudeLLM(client=FakeClient(text, stop)).answer("q", retriever.search("walk", k=1))


def test_workspace_id_is_sent_as_header_when_set(monkeypatch):
    from pawpal_ai.llm import workspace_headers

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    assert workspace_headers() == {}
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", " wrkspc_test123 ")
    assert workspace_headers() == {"anthropic-workspace-id": "wrkspc_test123"}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    llm = ClaudeLLM()  # real SDK client, no request sent
    assert llm.client.default_headers["anthropic-workspace-id"] == "wrkspc_test123"


def test_get_llm_defaults_to_offline_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("PAWPAL_LLM", raising=False)
    assert get_llm().name == "offline"
    assert get_llm("offline").name == "offline"
