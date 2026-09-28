import time
from datetime import datetime

# Must run BEFORE any app.* import below — several modules (create_db.py,
# qa.py, tts.py, assemblyai_client.py) read their API keys with os.getenv()
# as module-level constants at import time, so .env has to be loaded first
# or those reads will see nothing.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.create_db import init_db
from app.document import FIELD_GROUP_MODELS
from app.doc_in import CreatePatientIn, IntakeSubmission, VagueFlagResult, AskIn, TTSIn
from app.vague_check import check_red_flag_specificity
from app.red_flags import check_red_flags
from app.symptom_log import mentions_symptom
from app.closing_remarks import is_closing_remark, CLOSING_REPLY
from app.tts import synthesize_speech_stream
from app.qa import answer_question
from app.assemblyai_client import mint_temporary_token
import app.db_functions as repo
from app.db_functions import DuplicatePatientError

app = FastAPI(title="Discharge Assistant API", version="0.3.0")

# Wide-open CORS for hackathon/local development, so the static intake form
# (opened as a file:// page or served separately) can call this API.
# Tighten this to specific origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


# ---------------------------------------------------------------------------
# Patients
#
# Route ordering matters here: /patients/search and /patients/lookup are
# STATIC paths and must be declared before /patients/{code}, or FastAPI
# will try to match "search"/"lookup" as a {code} value first.
# ---------------------------------------------------------------------------

@app.post("/patients")
def create_patient(payload: CreatePatientIn):
    try:
        doc = repo.create_patient(payload.display_name, payload.procedure_name, payload.phone_number)
    except DuplicatePatientError as e:
        raise HTTPException(status_code=409, detail=str(e))
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/patients/search")
def search_patients(q: str = ""):
    patients = repo.search_patients(q)
    return {"results": patients}


@app.get("/patients/lookup")
def lookup_patient(identifier: str):
    """Patient-side login: accepts either a patient code or a phone number
    and resolves to the actual patient record either way."""
    patient = repo.get_patient_by_identifier(identifier)
    if not patient:
        raise HTTPException(status_code=404, detail="No patient found with that code or phone number")
    patient["_id"] = str(patient["_id"])
    return patient


@app.get("/patients/{code}")
def get_patient(code: str):
    patient = repo.get_patient(code)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient code not found")
    patient["_id"] = str(patient["_id"])
    return patient


# ---------------------------------------------------------------------------
# Red flag specificity check (used live by the intake form, per field,
# before the doctor saves — no LLM call, see vague_check.py)
# ---------------------------------------------------------------------------

@app.post("/red-flags/check", response_model=VagueFlagResult)
def check_red_flag(symptom: str):
    is_vague, reason = check_red_flag_specificity(symptom)
    return VagueFlagResult(symptom=symptom, is_vague=is_vague, reason=reason)


# ---------------------------------------------------------------------------
# Intake / update submission
#
# Same endpoint serves both the initial intake (Week 1) and later doctor
# updates (Week 3): any field group included in the payload supersedes the
# current active items for that group in the patient document. Field groups
# omitted from the payload are left untouched.
# ---------------------------------------------------------------------------

@app.post("/patients/{code}/intake")
def submit_intake(code: str, payload: IntakeSubmission):
    if repo.get_patient(code) is None:
        raise HTTPException(status_code=404, detail="Patient code not found")

    result = {}
    group_payloads = {
        "medications": payload.medications,
        "activity_restrictions": payload.activity_restrictions,
        "wound_care": payload.wound_care,
        "red_flags": payload.red_flags,
        "follow_up": [payload.follow_up] if payload.follow_up is not None else None,
        "general_notes": payload.general_notes,
    }

    for group, items in group_payloads.items():
        if items is None:
            continue
        raw_items = [item.model_dump() for item in items]

        # Attach the vague-flag result for red flags specifically.
        if group == "red_flags":
            for raw in raw_items:
                is_vague, _ = check_red_flag_specificity(raw["symptom"])
                raw["flagged_vague"] = is_vague

        count = repo.apply_field_group_update(code, group, payload.set_by, raw_items)
        result[group] = count

    return {"patient_code": code, "saved": result, "saved_at": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Current instructions (what the patient-side assistant reads from —
# only active/non-superseded items)
# ---------------------------------------------------------------------------

@app.get("/patients/{code}/current-instructions")
def get_current_instructions(code: str):
    instructions = repo.get_current_instructions(code)
    if instructions is None:
        raise HTTPException(status_code=404, detail="Patient code not found")
    return instructions


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Text-to-speech (Deepgram Aura-2). Streamed through rather than buffered
# server-side — see tts.py. The frontend must ALSO consume this
# progressively (not buffer the whole blob) or streaming here doesn't
# actually reduce perceived latency; see voice_assistant.html's speak().
# ---------------------------------------------------------------------------

@app.post("/tts")
def text_to_speech(payload: TTSIn):
    t0 = time.perf_counter()
    try:
        audio_stream = synthesize_speech_stream(payload.text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    # synthesize_speech_stream returns once Deepgram's response HEADERS have
    # arrived, so this is effectively Deepgram's time-to-first-byte.
    print(f"[timing] Deepgram first byte: {(time.perf_counter() - t0) * 1000:.0f} ms ({len(payload.text)} chars)")
    return StreamingResponse(audio_stream, media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# AssemblyAI streaming token
#
# The browser mic client needs a short-lived token to open a WebSocket
# straight to AssemblyAI — it must never hold the real API key. This
# endpoint mints one on the server and hands it to the frontend.
# ---------------------------------------------------------------------------

@app.get("/assemblyai/token")
def get_assemblyai_token(expires_in_seconds: int = 60):
    try:
        return mint_temporary_token(expires_in_seconds=expires_in_seconds)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---------------------------------------------------------------------------
# Patient-side conversation
#
# Order matters: red flags are checked FIRST, always — a real emergency
# phrase must escalate no matter what else the utterance also contains.
# Only if nothing escalates do we check for a pure closing remark (skips
# the LLM for "thank you"-type messages), and only then does the grounded
# Q&A model get called at all.
# ---------------------------------------------------------------------------

@app.post("/patients/{code}/ask")
def ask(code: str, payload: AskIn):
    instructions = repo.get_current_instructions(code)
    if instructions is None:
        raise HTTPException(status_code=404, detail="Patient code not found")

    red_flag = check_red_flags(payload.question, instructions.get("red_flags", []))
    if red_flag:
        if red_flag["action"] == "go_to_er":
            message = "This may be a medical emergency. Please call emergency services or go to your nearest emergency room now."
        else:
            message = "This is something you should contact your clinic about. Please call your care team now."
        return {
            "escalate": True,
            "action": red_flag["action"],
            "matched_category": red_flag["category"],
            "message": message,
        }

    if is_closing_remark(payload.question):
        return {"escalate": False, "answer": CLOSING_REPLY, "logged_symptom": False}

    t0 = time.perf_counter()
    answer = answer_question(payload.question, instructions)
    print(f"[timing] answer_question (LLM): {(time.perf_counter() - t0) * 1000:.0f} ms")

    # Auto-log: any symptom-like mention that DIDN'T trigger a red flag still
    # gets saved for the doctor to review — post-op medications can have
    # side effects worth a look even if they're not ER-urgent. The patient
    # is always told this happened; logging is never silent and never a
    # substitute for the red-flag escalation above.
    logged = False
    if mentions_symptom(payload.question):
        repo.add_symptom_report(code, payload.question)
        answer = answer + " I've made a note of this for your doctor to review."
        logged = True

    return {"escalate": False, "answer": answer, "logged_symptom": logged}


# ---------------------------------------------------------------------------
# Symptom reports — doctor-facing view of what the patient has logged.
# Append-only; nothing here is ever superseded or hidden.
# ---------------------------------------------------------------------------

@app.get("/patients/{code}/symptom-reports")
def get_symptom_reports(code: str):
    reports = repo.get_symptom_reports(code)
    if reports is None:
        raise HTTPException(status_code=404, detail="Patient code not found")
    return {"patient_code": code, "symptom_reports": reports}