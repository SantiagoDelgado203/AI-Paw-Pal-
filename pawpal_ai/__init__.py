"""PawPal+ AI layer: retrieval, guardrails, LLM backends, and the care-planning agent."""

from __future__ import annotations

import logging
from pathlib import Path

from .agent import Answer, CareAssistant, PlanResult, ProposedTask
from .knowledge import Retriever
from .llm import ClaudeLLM, OfflineLLM, get_llm

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "pawpal_ai.log"


def configure_logging(log_file: str | Path = LOG_FILE, level: int = logging.INFO) -> Path:
    """Send pawpal_ai logs to a file (idempotent). Returns the log file path."""
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pawpal_ai")
    logger.setLevel(level)
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
               for h in logger.handlers):
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return log_file


__all__ = ["Answer", "CareAssistant", "ClaudeLLM", "OfflineLLM", "PlanResult", "ProposedTask",
           "Retriever", "configure_logging", "get_llm"]
