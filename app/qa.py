"""
Grounded Q&A for the patient-side assistant — Gemini version.

Same contract as llm_qa.py (answer_question(question, instructions) -> str),
so switching providers is a one-line import change in main.py. See that
file's docstring for the full rationale; this module only differs in which
API it calls.

Using gemini-2.5-flash: standard (non-preview) model, free tier available,
no restrictive preview rate limits (unlike Gemini's TTS models, which are
free but capped at 2 requests/minute on the free tier — not used here).
"""

import os
import json

from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")  # can be overridden in .env

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key from "
                "aistudio.google.com/apikey and set it as an environment "
                "variable before starting the server."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a voice assistant helping a patient understand their \
own post-surgery discharge instructions, which are provided to you below as JSON.

Hard rules, no exceptions:
1. Answer ONLY using the information in the provided instructions. Never use \
outside medical knowledge, never guess, never generalize from "most patients."
2. If the answer isn't in the instructions, say so plainly and suggest they \
contact their care team — do not attempt to answer from general knowledge, \
even if you're confident it's correct in general.
3. Never diagnose, never assess whether a symptom is serious, and never \
tell the patient whether something is "normal" unless their instructions \
say so explicitly. (A separate safety system already handles anything that \
sounds like an emergency — you will not be called for those cases.)
4. Keep answers short and speakable — this is a voice conversation, not a \
document. One to three sentences unless the patient asks for a full list.
5. If the patient asks something broad ("what are my instructions?", "tell \
me everything"), do NOT recite every detail — that's too much to follow by \
ear. Instead, briefly name the categories you have for them (for example: \
medications, wound care, activity restrictions, warning signs, and their \
follow-up appointment) and invite them to pick one. Give full detail only \
when they ask about a specific category.
6. Never end a sentence mid-thought. If you're running long, finish the \
sentence you're on and offer to continue, rather than trailing off.
7. Speak directly to the patient, second person, plain language, warm but \
efficient. No clinical jargon they weren't already given.

Patient's current discharge instructions (JSON):
{instructions_json}
"""


def build_system_prompt(instructions: dict) -> str:
    def clean_items(items):
        return [
            {k: v for k, v in item.items()
             if k not in ("item_id", "set_by", "created_at", "active", "superseded_by")}
            for item in (items or [])
        ]

    clean = {
        "procedure": instructions.get("procedure_name"),
        "medications": clean_items(instructions.get("medications")),
        "activity_restrictions": clean_items(instructions.get("activity_restrictions")),
        "wound_care": clean_items(instructions.get("wound_care")),
        "red_flags": clean_items(instructions.get("red_flags")),  # context only, not detection
        "follow_up": (
            {k: v for k, v in instructions["follow_up"].items()
             if k not in ("item_id", "set_by", "created_at", "active", "superseded_by")}
            if instructions.get("follow_up") else None
        ),
        "general_notes": clean_items(instructions.get("general_notes")),
    }
    return SYSTEM_PROMPT.format(instructions_json=json.dumps(clean, indent=2, default=str))


def answer_question(question: str, instructions: dict) -> str:
    client = _get_client()
    system_prompt = build_system_prompt(instructions)

    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=800,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )
    return (response.text or "").strip()