"""Reliability evaluation for the PawPal+ AI assistant.

Runs every case in evaluation/cases.json, checks it against its expectations, and writes a
Markdown report (evaluation/results_<backend>.md) plus a one-line summary to stdout.

    python evaluation/run_eval.py              # offline backend (reproducible, no API key)
    python evaluation/run_eval.py --llm claude # same cases against Claude (needs a key)
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from pawpal_ai import CareAssistant, configure_logging, get_llm  # noqa: E402
from pawpal_system import Owner, Pet, Task  # noqa: E402

DAY = date(2026, 1, 5)


def household() -> Owner:
    """A fixed evaluation household covering every species and life stage."""
    owner = Owner("Eval", available_minutes=300)
    for name, species, age in [("Mochi", "dog", 3), ("Luna", "cat", 5), ("Bean", "dog", 0),
                               ("Rex", "dog", 12), ("Thumper", "rabbit", 2), ("Kiwi", "bird", 4)]:
        owner.add_pet(Pet(name, species, age))
    owner.get_pet("Mochi").add_task(Task("Morning walk", "07:00", 30, "high", "daily", due_date=DAY))
    owner.get_pet("Luna").add_task(Task("Flea medication", "12:00", 5, "high", "weekly", due_date=DAY))
    return owner


def eval_qa(assistant: CareAssistant, case: dict) -> dict:
    ans = assistant.answer_question(case["question"], case.get("species"))
    problems = []
    if ans.status != case["expect_status"]:
        problems.append(f"status {ans.status} != {case['expect_status']}")
    ids = [r.chunk.id for r in ans.sources]
    if case.get("expect_source") and not any(i.startswith(e) for e in case["expect_source"] for i in ids):
        problems.append(f"expected source {case['expect_source']} not in {ids}")
    for phrase in case.get("must_contain", []):
        if phrase.lower() not in ans.text.lower():
            problems.append(f"answer missing '{phrase}'")
    return {"id": case["id"], "input": case["question"] or "(empty)", "status": ans.status,
            "confidence": ans.confidence, "passed": not problems, "problems": problems,
            "detail": ans.text[:110].replace("|", "/")}


def eval_plan(assistant: CareAssistant, case: dict, base: Owner) -> dict:
    owner = copy.deepcopy(base)
    if "budget" in case:
        owner.available_minutes = case["budget"]
    plan = assistant.plan_care(owner, case["pet"], case["request"], DAY)
    descs = [p.task.description.lower() for p in plan.accepted]
    problems = []
    if plan.status != case["expect_status"]:
        problems.append(f"status {plan.status} != {case['expect_status']}")
    for word in case.get("must_include", []):
        if not any(word in d for d in descs):
            problems.append(f"missing '{word}'")
    for word in case.get("must_exclude", []):
        if any(word in d for d in descs):
            problems.append(f"should not include '{word}'")
    if plan.status == "ok":
        expected = {"no_conflicts": True, "within_budget": True, "all_tasks_cited": True}
        expected.update(case.get("expect_checks", {}))
        for check, want in expected.items():
            if plan.checks.get(check) != want:
                problems.append(f"self-check {check}={plan.checks.get(check)} (expected {want})")
    return {"id": case["id"], "input": f"{case['pet']}: {case['request']}", "status": plan.status,
            "confidence": plan.confidence, "passed": not problems, "problems": problems,
            "detail": ", ".join(p.task.description for p in plan.accepted) or plan.trace[-1][:110],
            "rejected": len(plan.rejected), "iterations": plan.iterations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", choices=["offline", "claude", "auto"], default="offline")
    args = parser.parse_args(argv)
    configure_logging()

    cases = json.loads((ROOT / "evaluation" / "cases.json").read_text(encoding="utf-8"))
    assistant = CareAssistant(llm=get_llm(args.llm))
    base = household()
    qa = [eval_qa(assistant, c) for c in cases["qa"]]
    plans = [eval_plan(assistant, c, base) for c in cases["plans"]]
    rows = qa + plans

    passed = sum(r["passed"] for r in rows)
    qa_pass = sum(r["passed"] for r in qa)
    plan_pass = sum(r["passed"] for r in plans)
    answered = [r["confidence"] for r in qa if r["status"] == "answered"]
    ok_plans = [r["confidence"] for r in plans if r["status"] == "ok"]
    retrieval_cases = [c for c in cases["qa"] if c.get("expect_source")]
    retrieval_hits = sum(1 for c, r in zip(cases["qa"], qa)
                         if c.get("expect_source") and not any("expected source" in p for p in r["problems"]))
    summary = (f"{passed}/{len(rows)} cases passed (Q&A {qa_pass}/{len(qa)}, planner {plan_pass}/{len(plans)}); "
               f"retrieval hit-rate {retrieval_hits}/{len(retrieval_cases)}; "
               f"mean confidence {statistics.mean(answered):.2f} on answers, "
               f"{statistics.mean(ok_plans):.2f} on plans; backend={assistant.llm.name}")

    lines = [f"# Evaluation results ({assistant.llm.name} backend)", "",
             f"Generated by `python evaluation/run_eval.py --llm {args.llm}` on a fixed household "
             f"and fixed date ({DAY}).", "", f"**Summary:** {summary}", "",
             "## Q&A cases", "", "| ID | Input | Status | Confidence | Result | Output / problems |",
             "|---|---|---|---|---|---|"]
    for r in qa:
        verdict = "✅ Pass" if r["passed"] else "❌ Fail"
        detail = "; ".join(r["problems"]) if r["problems"] else r["detail"]
        lines.append(f"| {r['id']} | {r['input']} | {r['status']} | {r['confidence']:.2f} | {verdict} | {detail} |")
    lines += ["", "## Planner cases", "",
              "| ID | Input | Status | Confidence | Rounds | Rejected | Result | Accepted tasks / problems |",
              "|---|---|---|---|---|---|---|---|"]
    for r in plans:
        verdict = "✅ Pass" if r["passed"] else "❌ Fail"
        detail = "; ".join(r["problems"]) if r["problems"] else r["detail"]
        lines.append(f"| {r['id']} | {r['input']} | {r['status']} | {r['confidence']:.2f} | "
                     f"{r['iterations']} | {r['rejected']} | {verdict} | {detail} |")
    out = ROOT / "evaluation" / f"results_{assistant.llm.name}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for r in rows:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']:<8} {r['status']:<9} conf={r['confidence']:.2f}  {r['input'][:55]}"
              + (f"  -> {'; '.join(r['problems'])}" if r["problems"] else ""))
    print("\n" + summary)
    print(f"Report written to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
