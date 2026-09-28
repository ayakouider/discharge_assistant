from datetime import datetime, date
from typing import Optional
from enum import Enum
import secrets
import uuid

from pydantic import BaseModel, Field


def now():
    return datetime.now()


def new_item_id():
    return uuid.uuid4().hex


def generate_patient_code(length: int = 6):
    alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ23456789'
    return ''.join(secrets.choice(alphabet) for _ in range(length))


class EscalationLevel(str, Enum):
    call_clinic = "call_clinic"
    go_to_er = "go_to_er"


class UpdatedItem(BaseModel):
    item_id: str = Field(default_factory=new_item_id)
    set_by: str  # <-- added: was missing, so it was being silently dropped
    created_at: datetime = Field(default_factory=now)
    # Renamed from "action" -> "active": "action" collided with
    # EscalationInstructions.action (call_clinic / go_to_er) below, which
    # silently overrode this versioning flag for red flags specifically.
    active: bool = True
    superseded_by: Optional[str] = None  # fixed typo: was "superceded_by"


class MedicationInstructions(UpdatedItem):
    name: str          # was "medication_name" — aligned with doc_in.py / the intake form
    dose: str          # was "dosage" — aligned with doc_in.py / the intake form
    frequency: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    notes: Optional[str] = None


class ActivityRestrictions(UpdatedItem):
    description: str
    until_date: Optional[date] = None
    notes: Optional[str] = None


class WoundCareSteps(UpdatedItem):
    step_order: int = 0
    instruction: str
    frequency: Optional[str] = None


class EscalationInstructions(UpdatedItem):
    symptom: str
    action: EscalationLevel   # no longer collides — parent flag is now "active"
    notes: Optional[str] = None
    flagged_vague: Optional[bool] = False


class FollowUpInstructions(UpdatedItem):
    appointment_date: Optional[date] = None
    provider_name: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None


class GeneralNotes(UpdatedItem):
    note: str   # was "notes" — aligned with doc_in.py / the intake form


class SymptomReport(BaseModel):
    """Patient-logged symptom note, visible to the doctor. Deliberately NOT
    a versioned/UpdatedItem: reports are append-only history, not something
    a later entry supersedes — every report the patient makes stays visible."""
    item_id: str = Field(default_factory=new_item_id)
    created_at: datetime = Field(default_factory=now)
    note: str
    source: str = "patient"  # for future-proofing if reports can come from elsewhere


FIELD_GROUP_MODELS = {
    "medications": MedicationInstructions,
    "activity_restrictions": ActivityRestrictions,
    "wound_care": WoundCareSteps,
    "red_flags": EscalationInstructions,
    "follow_up": FollowUpInstructions,
    "general_notes": GeneralNotes,
}