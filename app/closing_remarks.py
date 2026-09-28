"""
Detects a pure closing remark ("thank you", "thanks, bye") so /ask can
short-circuit to a brief canned reply instead of sending it through the
grounded-QA LLM, which would produce an odd or wasteful response to
something that isn't really a question.

Deliberately conservative: only short-circuits when the message is SHORT,
contains NO question mark, and contains NO question word — a short message
that happens to include "thanks" but is also asking something ("thanks,
when's my appointment again?") must still reach the real answer path.
Safety note: this runs only AFTER red_flags.check_red_flags() in main.py,
never before — nothing here ever affects escalation.
"""

import re

_CLOSING_KEYWORDS = {"thank", "thanks", "thx", "bye", "goodbye", "appreciate", "goodnight"}
_QUESTION_WORDS = {
    "when", "how", "what", "why", "where", "who", "which",
    "can", "could", "should", "would", "is", "are", "will",
    "do", "does", "did", "have", "has",
}
_MAX_WORDS = 6


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s']", " ", text.lower())


def is_closing_remark(text: str) -> bool:
    if "?" in text:
        return False

    words = _normalize(text).split()
    if not words or len(words) > _MAX_WORDS:
        return False
    if any(w in _QUESTION_WORDS for w in words):
        return False

    return any(any(k in w for k in _CLOSING_KEYWORDS) for w in words)


CLOSING_REPLY = "You're welcome! Let me know if you have any other questions."