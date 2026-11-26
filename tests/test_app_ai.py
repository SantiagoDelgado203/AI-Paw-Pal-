"""End-to-end UI test: drive the Streamlit AI-assistant tab with AppTest (offline backend)."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def _button(at, label):
    return next(b for b in at.button if b.label.startswith(label))


def _text(at, label):
    return next(t for t in at.text_input if t.label.startswith(label))


def test_ai_tab_ask_then_draft_and_approve_plan(tmp_path, monkeypatch):
    monkeypatch.setenv("PAWPAL_DATA_FILE", str(tmp_path / "data.json"))
    monkeypatch.setenv("PAWPAL_LLM", "offline")
    at = AppTest.from_file("app.py", default_timeout=30).run()
    _text(at, "Pet name").input("Mochi")
    _button(at, "Add pet").click().run()

    _text(at, "Question").input("How often should I walk my dog?")
    _button(at, "Ask").click().run()
    assert any("two walks a day" in s.value for s in at.success)

    next(r for r in at.radio if "need" in r.label).set_value("🗓️ Draft a care plan").run()
    _text(at, "What should").input("Plan a daily routine")
    _button(at, "Draft plan").click().run()
    suggestions = [c for c in at.checkbox if "·" in c.label]
    assert len(suggestions) >= 4

    suggestions[-1].uncheck().run()  # the owner rejects one suggestion
    _button(at, "✅ Add").click().run()
    assert any(f"Added {len(suggestions) - 1} AI-suggested" in s.value for s in at.success)
    assert not at.exception
    assert (tmp_path / "data.json").exists()  # approved tasks were persisted
