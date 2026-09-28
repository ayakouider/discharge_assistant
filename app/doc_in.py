"""Pydantic request/response shapes for the doctor intake API."""

from datetime import date
from typing import Optional, List
from pydantic import BaseModel

from app.document import EscalationLevel


# ---- Inbound (doctor submits these) ----

class MedicationIn(BaseModel):
    name: str
    dose: str
    frequency: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    notes: Optional[str] = None


class ActivityRestrictionIn(BaseModel):
    description: str
    until_date: Optional[date] = None
    notes: Optional[str] = None


class WoundCareStepsIn(BaseModel):
    step_order: int = 0
    instruction: str
    frequency: Optional[str] = None


class EscalationInstructionIn(BaseModel):
    symptom: str
    action: EscalationLevel
    notes: Optional[str] = None


class FollowUpIn(BaseModel):
    appointment_date: Optional[date] = None
    provider_name: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None


class GeneralNoteIn(BaseModel):
    note: str


class CreatePatientIn(BaseModel):
    display_name: Optional[str] = None
    procedure_name: Optional[str] = None
    phone_number: Optional[str] = None


class IntakeSubmission(BaseModel):
    """Full intake payload submitted by the doctor for a patient.
    Used both for the initial intake and for updates: any field group
    included here will supersede the current active rows in that group."""
    set_by: str  # doctor name/id
    medications: Optional[List[MedicationIn]] = None
    activity_restrictions: Optional[List[ActivityRestrictionIn]] = None
    wound_care: Optional[List[WoundCareStepsIn]] = None
    red_flags: Optional[List[EscalationInstructionIn]] = None
    follow_up: Optional[FollowUpIn] = None
    general_notes: Optional[List[GeneralNoteIn]] = None


# ---- Outbound ----

class VagueFlagResult(BaseModel):
    symptom: str
    is_vague: bool
    reason: Optional[str] = None


# ---- Patient-side conversation ----

class AskIn(BaseModel):
    question: str


class TTSIn(BaseModel):
    text: str