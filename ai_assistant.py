"""Command-line interface for the PawPal+ AI care assistant.

Examples:
    python ai_assistant.py ask "How often should I scoop the litter box?"
    python ai_assistant.py plan --pet Mochi "Plan a daily routine with walks and meals"
    python ai_assistant.py plan --pet Luna --minutes 240 --apply "Luna needs flea medication"
    python ai_assistant.py demo            # scripted run of every feature (used for evidence/)

Add --llm offline|claude|auto to pick the backend (default: auto -> Claude if
ANTHROPIC_API_KEY is set, otherwise the deterministic offline backend).
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from main import build_demo_owner
from pawpal_ai import Answer, CareAssistant, PlanResult, configure_logging, get_llm
from pawpal_system import Owner, Scheduler

sys.stdout.reconfigure(encoding="utf-8")


def print_answer(ans: Answer) -> None:
    """Pretty-print a Q&A result."""
    print(f"Q: {ans.question}")
    print(f"A: {ans.text}")
    print(f"   status={ans.status}  confidence={ans.confidence:.2f}  backend={ans.backend}")
    for r in ans.sources:
        print(f"   source: [{r.chunk.id}] {r.chunk.label} (score {r.score:.2f})")
    for note in ans.notes:
        print(f"   note: {note}")


def print_plan(plan: PlanResult, show_trace: bool = True) -> None:
    """Pretty-print a planner result, including the agent trace."""
    print(f"Request for {plan.pet_name}: {plan.request!r}  ({plan.day:%a %b %d})")
    print(f"Status: {plan.status}  backend={plan.backend}  iterations={plan.iterations}  "
          f"confidence={plan.confidence:.2f}{'  ⚠️ NEEDS REVIEW' if plan.needs_review else ''}")
    if show_trace:
        print("Agent trace:")
        for line in plan.trace:
            print(f"  · {line}")
    print("Proposed tasks (awaiting approval):")
    for i, p in enumerate(plan.accepted):
        t = p.task
        extra = f"  [{'; '.join(p.notes)}]" if p.notes else ""
        print(f"  {i}. {t.time} {t.emoji()} {t.description:<26} {t.duration_minutes:>3} min "
              f"{t.priority:<6} {t.frequency:<6} <- {p.source}{extra}")
    if not plan.accepted:
        print("  (none)")
    if plan.rejected:
        print("Rejected / dropped:")
        for desc, reason in plan.rejected:
            print(f"  ✘ {desc}: {reason}")
    if plan.checks:
        print("Self-check: " + ", ".join(f"{k}={'✔' if v else '✘'}" for k, v in plan.checks.items()))


def load_owner(data: str | None, minutes: int | None) -> Owner:
    """Load the owner from a JSON file, or build the demo household."""
    owner = Owner.load_from_json(data) if data else None
    if data and owner is None:
        raise SystemExit(f"No data file at {data}")
    owner = owner or build_demo_owner()
    if minutes is not None:
        owner.available_minutes = minutes
    return owner


def run_demo(assistant: CareAssistant) -> None:
    """Scripted walkthrough used to generate the evidence files."""
    questions = [
        ("How often should I walk my dog?", "dog"),
        ("How often should I scoop the litter box?", "cat"),
        ("Can I give my dog ibuprofen for pain?", "dog"),
        ("Help! My dog ate chocolate an hour ago", "dog"),
        ("What's the best laptop for gaming?", None),
    ]
    print("=" * 72 + "\nPART 1 - Retrieval-augmented Q&A\n" + "=" * 72)
    for q, species in questions:
        print_answer(assistant.answer_question(q, species))
        print()

    print("=" * 72 + "\nPART 2 - Agentic care planner (demo household, 240 min budget)\n" + "=" * 72)
    owner = build_demo_owner()
    owner.available_minutes = 240
    requests = [
        ("Mochi", "Plan a daily routine for Mochi"),
        ("Luna", "Luna needs her flea medication and more play time"),
        ("Luna", "Give Luna 50 mg of ibuprofen with chocolate treats"),
    ]
    for pet, req in requests:
        print_plan(assistant.plan_care(owner, pet, req))
        print()

    print("=" * 72 + "\nPART 3 - Human approval: apply Mochi's plan, then show the schedule\n" + "=" * 72)
    plan = assistant.plan_care(owner, "Mochi", "Plan a daily routine for Mochi")
    added = CareAssistant.apply_plan(owner, plan)
    print(f"Owner approved {len(added)} task(s).")
    print(Scheduler(owner).format_schedule())
    new = {t.description for t in added}
    conflicts = Scheduler(owner).detect_conflicts()
    print("Conflicts involving AI-added tasks:",
          [c for c in conflicts if any(d in c for d in new)] or "none")
    print("Pre-existing conflicts (owner's own tasks):",
          [c for c in conflicts if not any(d in c for d in new)] or "none")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the chosen command."""
    parser = argparse.ArgumentParser(description="PawPal+ AI care assistant")
    parser.add_argument("--llm", choices=["auto", "offline", "claude"], default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="ask a pet-care question")
    ask.add_argument("question")
    ask.add_argument("--species", choices=["dog", "cat", "rabbit", "bird"])
    plan = sub.add_parser("plan", help="generate a care plan for one pet")
    plan.add_argument("request")
    plan.add_argument("--pet", required=True)
    plan.add_argument("--data", help="owner JSON file (default: built-in demo household)")
    plan.add_argument("--minutes", type=int, help="override the owner's daily time budget")
    plan.add_argument("--apply", action="store_true", help="approve all proposals and save to --data")
    sub.add_parser("demo", help="scripted run of every feature")
    args = parser.parse_args(argv)

    log_path = configure_logging()
    assistant = CareAssistant(llm=get_llm(args.llm))
    print(f"[backend: {assistant.llm.name} | log: {Path(log_path).name}]\n")

    if args.cmd == "ask":
        print_answer(assistant.answer_question(args.question, args.species))
    elif args.cmd == "plan":
        owner = load_owner(args.data, args.minutes)
        result = assistant.plan_care(owner, args.pet, args.request, date.today())
        print_plan(result)
        if args.apply and result.accepted:
            CareAssistant.apply_plan(owner, result)
            if args.data:
                owner.save_to_json(args.data)
            print(f"\nApplied {len(result.accepted)} task(s)." + (f" Saved to {args.data}." if args.data else ""))
        return 0 if result.status == "ok" else 1
    else:
        run_demo(assistant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
