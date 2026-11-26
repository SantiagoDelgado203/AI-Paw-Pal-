"""Guardrails: input checks, emergency detection, and safety/applicability rules for AI output.

Everything here is deterministic and runs *around* the language model, so the system stays
safe even when the model is wrong, offline, or replaced.
"""

from __future__ import annotations

import re

from pawpal_system import Pet

MAX_INPUT_CHARS = 500
MIN_DURATION, MAX_DURATION = 1, 240  # minutes; anything longer is almost certainly an error

EMERGENCY_PATTERNS = [
    r"\b(ate|eaten|swallowed|got into|ingested|licked)\b.*\b(chocolate|grapes?|raisins?|xylitol|"
    r"onions?|garlic|lil(y|ies)|poison|rat bait|antifreeze)",
    r"\b(chocolate|grapes?|raisins?|xylitol|antifreeze|rat poison)\b.*\b(ate|eaten|swallowed|got into)\b",
    r"\bseizur", r"\bcollaps", r"\bnot breathing\b", r"\b(can't|cannot|trouble|struggling to) breath",
    r"\bbleeding\b", r"\bpoison", r"\bunconscious\b", r"\bheat ?stroke\b",
    r"\b(can't|cannot) (pee|urinate)\b", r"\bswollen (belly|stomach|abdomen)\b",
]
EMERGENCY_MESSAGE = (
    "⚠️ This may be an emergency. Contact your veterinarian, the nearest emergency animal "
    "clinic, or an animal poison control line right now. Do not wait to see if it improves. "
    "PawPal+ is a planning tool and cannot give medical treatment advice."
)

# Content that must never appear in an AI-generated care task.
UNSAFE_TASK_PATTERNS = {
    r"\b(chocolate|xylitol|grapes?|raisins?|onions?|garlic|alcohol|caffeine|coffee)\b":
        "mentions a food that is toxic to pets",
    r"\b(ibuprofen|acetaminophen|tylenol|advil|aspirin|naproxen|aleve)\b":
        "mentions a human medication",
    r"\b\d+(\.\d+)?\s*(mg|ml|mcg|milligrams?|tablets?)\b":
        "specifies a medication dose (only a veterinarian should set doses)",
    r"\bdouble (the )?dose\b": "suggests doubling a dose",
}
SPECIES_UNSAFE = {
    "rabbit": {r"\bbath(e|ing)?\b": "rabbits should never be bathed in water"},
    "cat": {r"\blil(y|ies)\b": "lilies are toxic to cats"},
    "bird": {r"\b(avocado|teflon|non-stick)\b": "avocado and non-stick fumes are dangerous for birds"},
}

PET_TERMS = re.compile(
    r"\b(pets?|dogs?|puppy|puppies|pup|cats?|kittens?|kitty|rabbits?|bunny|bunnies|birds?|"
    r"parrots?|parakeets?|budgies?|cockatiels?|animals?|vets?|veterinar\w*|litter|leash|"
    r"walk\w*|feed\w*|groom\w*|flea|tick|heartworm|hay|cage|paws?|fur|coat|claws?|kibble|"
    r"treats?|toxic|poison\w*|chew\w*|bark\w*|meow\w*)\b")


def is_pet_related(text: str) -> bool:
    """True if the text mentions at least one pet-care term (off-topic questions are refused)."""
    return bool(PET_TERMS.search(text.lower()))


MEDICATION_WORDS = ("med", "pill", "flea", "tick", "heartworm", "prescri", "dose", "insulin")
VET_WORDS = ("vet", "checkup", "check-up", "vaccin")


class InputError(ValueError):
    """Raised when user input fails validation."""


def validate_text_input(text: str | None, what: str = "request") -> str:
    """Strip and check user text; raise InputError if empty or too long."""
    text = (text or "").strip()
    if not text:
        raise InputError(f"Please enter a {what}.")
    if len(text) > MAX_INPUT_CHARS:
        raise InputError(f"Your {what} is too long ({len(text)} characters, max {MAX_INPUT_CHARS}).")
    return text


def detect_emergency(text: str) -> list[str]:
    """Return the emergency patterns that match the text (empty list = no emergency)."""
    lowered = text.lower()
    return [p for p in EMERGENCY_PATTERNS if re.search(p, lowered)]


def unsafe_reason(description: str, species: str) -> str | None:
    """Return why a task description is unsafe, or None if it passes."""
    lowered = description.lower()
    rules = dict(UNSAFE_TASK_PATTERNS)
    rules.update(SPECIES_UNSAFE.get(species.lower(), {}))
    for pattern, reason in rules.items():
        if re.search(pattern, lowered):
            return reason
    return None


def life_stage(pet: Pet) -> str:
    """Classify a pet as 'young', 'adult', or 'senior' from species and age."""
    if pet.age < 1:
        return "young"
    senior_age = {"dog": 8, "cat": 11, "rabbit": 6, "bird": 15}.get(pet.species.lower(), 10)
    return "senior" if pet.age >= senior_age else "adult"


def mentions(text: str, words: tuple[str, ...]) -> bool:
    """True if any of the word stems appear in the text."""
    lowered = text.lower()
    return any(w in lowered for w in words)


def applicability_issue(source_id: str, description: str, pet: Pet, request: str) -> str | None:
    """Return why a suggested task does not apply to this pet/request, or None if it applies.

    Prevents the assistant from inventing medications or vet visits the owner never mentioned,
    and from adding puppy/kitten or senior routines to pets of the wrong age.
    """
    stage = life_stage(pet)
    if source_id.startswith("life_stages#young") and stage != "young":
        return f"{pet.name} is not a young animal (age {pet.age})"
    if source_id.startswith("life_stages#senior") and stage != "senior":
        return f"{pet.name} is not a senior pet (age {pet.age})"
    is_med = source_id.startswith("medications#giving") or source_id.startswith("medications#flea") \
        or mentions(description, MEDICATION_WORDS)
    if is_med and not mentions(request, MEDICATION_WORDS):
        return "medication tasks are only added when the owner asks about medication"
    parasite = ("flea", "tick", "heartworm")
    if source_id.startswith("medications#flea") and not mentions(request, parasite):
        return "flea/tick prevention is only added when the owner asks about it"
    if source_id.startswith("medications#giving") and not (
            mentions(request, ("pill", "prescri", "insulin", "medicine", "dose"))
            or ("medication" in request.lower() and not mentions(request, parasite))):
        return "a daily prescription reminder is only added when the owner mentions one"
    is_vet = source_id.startswith("medications#vet") or mentions(description, VET_WORDS)
    if is_vet and not mentions(request, VET_WORDS):
        return "vet visits are only added when the owner asks for one"
    return None


def is_duplicate(description: str, existing: list[str]) -> str | None:
    """Return the existing task this one duplicates, or None.

    Exact (case-insensitive) matches always count. Medication tasks also count as duplicates
    when they share a keyword (e.g. "Flea and tick prevention" vs. an existing "Flea medication"),
    because a doubled medication reminder is the riskiest kind of duplicate.
    """
    lowered = description.lower()
    words = set(re.findall(r"[a-z]+", lowered)) - {"and", "the", "give", "medication", "daily"}
    for other in existing:
        if other.lower() == lowered:
            return other
        if mentions(description, MEDICATION_WORDS) and mentions(other, MEDICATION_WORDS):
            if words & set(re.findall(r"[a-z]+", other.lower())):
                return other
    return None


def clamp_duration(minutes: int) -> int:
    """Raise ValueError if a duration is outside the allowed range."""
    if not (MIN_DURATION <= int(minutes) <= MAX_DURATION):
        raise ValueError(f"duration {minutes} min is outside {MIN_DURATION}-{MAX_DURATION} min")
    return int(minutes)
