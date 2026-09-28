"""
Broad symptom-mention detection, used only to decide whether to auto-log a
patient's utterance as a note for the doctor. Deliberately MORE permissive
than red_flags.py's matching: a false positive here just adds a harmless
note, unlike a false positive in red_flags.py which would trigger a full ER
escalation. So this casts a wider net on purpose.
"""

import re

SYMPTOM_KEYWORDS = {
    "pain", "painful", "hurts", "hurting", "hurt", "ache", "aching", "sore",
    "swelling", "swollen", "swell",
    "bleeding", "bleed", "bled", "blood",
    "fever", "feverish", "chills",
    "redness", "red", "discharge", "oozing", "ooze", "pus",
    "numbness", "numb", "tingling", "tingle",
    "dizziness", "dizzy", "lightheaded", "faint",
    "nausea", "nauseous", "vomiting", "vomit", "throwing up",
    "cough", "coughing",
    "rash", "itchy", "itching", "itch", "hives",
    "tired", "fatigue", "exhausted", "weak", "weakness",
    "diarrhea", "constipation", "constipated",
    "headache", "migraine",
    "burning", "stinging", "cramping", "cramp", "throbbing",
    "shortness of breath", "trouble breathing",
}


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower())


def mentions_symptom(text: str) -> bool:
    normalized = _normalize(text)
    return any(keyword in normalized for keyword in SYMPTOM_KEYWORDS)