"""
Heuristic specificity check for doctor-entered red flags.

No LLM call here (deliberately, per token-budget decision): a red flag like
"if it gets worse" is nearly useless downstream, both for the patient-side
assistant and for our red-flag matching layer, which needs a concrete
symptom to match against. This flags vague entries so the intake form can
nudge the doctor to be specific, without spending any tokens.
"""

import re

VAGUE_TERMS = [
    "worse", "worsens", "worsening", "bad", "badly", "not right",
    "not good", "unusual", "abnormal", "too much", "a lot",
    "significant", "severe" # "severe" alone with no symptom named is vague
]

# A concrete symptom mention alongside a vague term usually redeems it,
# e.g. "fever gets worse" is fine, "gets worse" alone is not.
SYMPTOM_HINTS = [
    "fever", "temperature", "bleeding", "blood", "pain", "swelling",
    "redness", "discharge", "pus", "breath", "breathing", "chest",
    "confusion", "vomit", "nausea", "numb", "weak", "dizzy", "faint",
    "rash", "hives", "cough", "urin", "stool", "bowel", "incision",
    "wound", "site", "leg", "calf", "headache", "vision", "speech",
]


def check_red_flag_specificity(symptom_text: str) -> tuple[bool, str | None]:
    """Returns (is_vague, reason). is_vague=True means the intake form
    should prompt the doctor to be more specific before saving."""
    text = symptom_text.lower().strip()

    if len(text) < 6:
        return True, "This is very short — add a specific symptom."

    has_vague_term = any(re.search(rf"\b{re.escape(term)}\b", text) for term in VAGUE_TERMS)
    has_symptom_hint = any(hint in text for hint in SYMPTOM_HINTS)

    if has_vague_term and not has_symptom_hint:
        return True, (
            "This sounds general rather than a specific symptom. "
            "Try naming the exact sign, e.g. 'fever above 101°F' "
            "instead of 'if it gets worse'."
        )

    if not has_symptom_hint and not has_vague_term:
        # Doesn't match any known symptom vocabulary at all — could still be
        # valid (our hint list isn't exhaustive), so warn softly rather than
        # block. This keeps false positives from blocking legitimate entries.
        return True, (
            "Couldn't recognize a specific symptom here — double check "
            "this is concrete enough for a patient to self-assess."
        )

    return False, None