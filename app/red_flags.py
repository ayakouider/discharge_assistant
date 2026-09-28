"""
Red-flag detection, run on every patient utterance BEFORE the grounded LLM
Q&A is ever called. Deterministic and keyword-based by design: this is the
safety-critical path, and a hardcoded, auditable rule ("this phrase always
escalates") is more trustworthy here than an LLM's in-context judgment call.

Two tiers:
- TIER 1 (universal): built into the app, always active, independent of what
  any doctor entered. Catches severity-of-symptom patterns that warrant
  emergency care after essentially any procedure.
- TIER 2 (doctor-specified): matched against this patient's own red_flags
  entries (from current-instructions), which carry the doctor's chosen
  action (call_clinic vs go_to_er).

Biased deliberately toward over-escalating: a false positive costs an
unnecessary "contact your care team" message; a false negative could cost
a life. If in doubt, this module escalates.

NOTE / next increment: this is keyword matching only. It will miss
paraphrases it wasn't given a pattern for ("it hurts to breathe" vs.
"shortness of breath"). The planned next layer is an LLM semantic check
that classifies the utterance against the same fixed category list (not
free-form medical reasoning) to catch paraphrase — deferred for now to keep
this safety-critical path testable without any external API call.
"""

import re
from typing import Optional


# (category, phrases, spoken guidance) — go_to_er, always.
TIER1_RED_FLAGS = [
    ("breathing_difficulty", [
        "can't breathe", "cant breathe", "can not breathe",
        "trouble breathing", "difficulty breathing", "hard to breathe",
        "short of breath", "shortness of breath", "struggling to breathe",
    ]),
    ("chest_pain", [
        "chest pain", "chest pressure", "chest tightness",
        "pressure in my chest", "tightness in my chest",
    ]),
    ("uncontrolled_bleeding", [
        "can't stop the bleeding", "cant stop the bleeding",
        "won't stop bleeding", "wont stop bleeding",
        "bleeding a lot", "heavy bleeding", "soaked through the bandage",
        "blood everywhere",
    ]),
    ("stroke_signs", [
        "slurred speech", "can't speak right", "cant speak right",
        "face is drooping", "one side of my face", "can't move my arm",
        "cant move my arm", "can't move my leg", "cant move my leg",
        "sudden confusion", "can't move one side",
    ]),
    ("anaphylaxis", [
        "throat is swelling", "throat swelling", "can't swallow",
        "cant swallow", "face is swelling", "hives and trouble breathing",
        "swelling in my throat",
    ]),
    ("loss_of_consciousness", [
        "passed out", "fainted", "lost consciousness", "blacked out",
    ]),
]

# Common symptom word-families collapsed to one canonical form, so "swollen"
# matches a doctor's "swelling", "painful" matches "pain", etc. Deliberately
# hand-curated rather than a generic stemmer: symptom vocabulary here is
# bounded, and a hand-curated list is easier to audit than stemmer output.
_SYNONYM_GROUPS = {
    "swelling": {"swelling", "swollen", "swell", "swells"},
    "pain": {"pain", "painful", "hurts", "hurting", "hurt", "ache", "aching"},
    "bleeding": {"bleeding", "bleed", "bled", "blood"},
    "fever": {"fever", "feverish"},
    "redness": {"redness", "red"},
    "discharge": {"discharge", "discharging", "oozing", "ooze", "pus"},
    "numbness": {"numbness", "numb"},
    "dizziness": {"dizziness", "dizzy", "lightheaded"},
    "nausea": {"nausea", "nauseous", "vomiting", "vomit"},
    "cough": {"cough", "coughing"},
}
_SYNONYM_LOOKUP = {word: canon for canon, words in _SYNONYM_GROUPS.items() for word in words}

_STOPWORDS = {
    "the", "and", "your", "you", "this", "that", "with", "have", "has",
    "will", "from", "into", "over", "under", "than", "then", "were", "was",
    "are", "for", "not", "any", "all", "can", "could", "should", "would",
}


def _canonicalize(word: str) -> str:
    if word in _SYNONYM_LOOKUP:
        return _SYNONYM_LOOKUP[word]
    if any(ch.isdigit() for ch in word):
        # "101f" -> "101", so it lines up with a transcript saying just "101"
        match = re.match(r"^(\d+)", word)
        if match:
            return match.group(1)
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower())


def _keywords(text: str) -> set:
    words = _normalize(text).split()
    return {_canonicalize(w) for w in words if len(w) >= 3 and w not in _STOPWORDS}


def check_tier1(transcript: str) -> Optional[dict]:
    normalized = _normalize(transcript)
    for category, phrases in TIER1_RED_FLAGS:
        for phrase in phrases:
            if phrase in normalized:
                return {
                    "tier": 1,
                    "category": category,
                    "matched_phrase": phrase,
                    "action": "go_to_er",
                }
    return None


def check_tier2(transcript: str, patient_red_flags: list[dict]) -> Optional[dict]:
    """patient_red_flags: the `red_flags` list from current-instructions,
    each item having at least `symptom` and `action`."""
    transcript_words = _keywords(transcript)
    if not transcript_words:
        return None

    best_match = None
    best_overlap = 0
    for flag in patient_red_flags:
        symptom_words = _keywords(flag.get("symptom", ""))
        if not symptom_words:
            continue
        overlap = symptom_words & transcript_words
        # Require 2+ overlapping keywords (or all of them, for a
        # single-word symptom). A threshold of 1 sounds safer but isn't:
        # it made ordinary questions like "how often do I take my pain
        # medication" match a red flag of "calf pain or swelling" on the
        # single word "pain" alone, falsely triggering an ER escalation on
        # a routine medication question. Requiring 2 keeps real matches
        # (synonym normalization already handles word-form variants like
        # swollen/swelling) while cutting single-common-word false alarms.
        threshold = min(2, len(symptom_words))
        if len(overlap) >= threshold and len(overlap) > best_overlap:
            best_overlap = len(overlap)
            best_match = flag

    if best_match:
        return {
            "tier": 2,
            "category": "doctor_specified",
            "matched_phrase": best_match.get("symptom"),
            "action": best_match.get("action"),
        }
    return None


def check_red_flags(transcript: str, patient_red_flags: list[dict]) -> Optional[dict]:
    """Run both tiers. Tier 1 takes priority (universal, always ER)."""
    tier1 = check_tier1(transcript)
    if tier1:
        return tier1
    return check_tier2(transcript, patient_red_flags or [])