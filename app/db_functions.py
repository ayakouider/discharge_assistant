"""
Repository layer — all MongoDB reads/writes go through here, so main.py
stays thin and the versioning logic (supersede old, insert new) lives in
exactly one place, generically, for every field group.
"""

import re
from datetime import datetime
from typing import Optional

from pymongo.errors import DuplicateKeyError

from app.create_db import patients_collection
from app.document import (
    generate_patient_code, FIELD_GROUP_MODELS, SymptomReport, now,
)


class DuplicatePatientError(Exception):
    """Raised when creating a patient would duplicate an existing phone
    number. Carries the existing patient's code so the caller can point
    the doctor at the record that already exists instead of failing blind."""
    def __init__(self, existing_code: str):
        self.existing_code = existing_code
        super().__init__(f"A patient with this phone number already exists (code: {existing_code})")


def create_patient(display_name: Optional[str], procedure_name: Optional[str],
                    phone_number: Optional[str] = None) -> dict:
    """Raises DuplicatePatientError if phone_number is given and already
    belongs to another patient — this is the "can't enter the same patient
    twice" check. A pre-check gives a friendly error message in the common
    case; the DuplicateKeyError catch below is a safety net for the rare
    race condition where two requests check at the same instant (the unique
    sparse index on phone_number, set up in create_db.py, is what actually
    guarantees this at the database level)."""
    if phone_number:
        existing = patients_collection.find_one({"phone_number": phone_number})
        if existing:
            raise DuplicatePatientError(existing["patient_code"])

    doc = {
        "patient_code": generate_patient_code(),
        "display_name": display_name,
        "procedure_name": procedure_name,
        "created_at": now(),
        # every field group starts as an empty array
        **{group: [] for group in FIELD_GROUP_MODELS},
        "symptom_reports": [],  # separate: append-only, not a versioned field group
    }
    # Only store the key when there's a real value. The unique index on
    # phone_number is sparse, and a sparse index skips documents where the
    # field is MISSING but still indexes a field explicitly set to null —
    # so storing phone_number=None would make the second patient created
    # without a phone number collide with the first on a real MongoDB.
    if phone_number:
        doc["phone_number"] = phone_number

    try:
        patients_collection.insert_one(doc)
    except DuplicateKeyError:
        existing = patients_collection.find_one({"phone_number": phone_number}) if phone_number else None
        if existing:
            raise DuplicatePatientError(existing["patient_code"])
        raise  # not a phone duplicate (e.g. a patient_code collision) — don't mislabel it
    return doc


def get_patient(code: str) -> Optional[dict]:
    return patients_collection.find_one({"patient_code": code})


def get_patient_by_identifier(identifier: str) -> Optional[dict]:
    """Used by the patient-side app: accepts either a patient code or a
    phone number and resolves to the patient record either way. Tries code
    first since that's an exact, indexed lookup; falls back to phone."""
    patient = get_patient(identifier)
    if patient:
        return patient
    return patients_collection.find_one({"phone_number": identifier})


def search_patients(query: Optional[str] = None) -> list[dict]:
    """Doctor-facing search across name, patient code, and phone number.
    An empty/missing query returns every patient — this is what powers
    "view the patient database" as well as the search box (the search box
    with nothing typed just shows the full list).

    Returns summary fields only (not full field-group history) since this
    is a browsing/list view, not a single-patient detail view."""
    projection = {
        "_id": 0, "patient_code": 1, "display_name": 1,
        "procedure_name": 1, "phone_number": 1, "created_at": 1,
    }

    if not query:
        cursor = patients_collection.find({}, projection)
    else:
        pattern = {"$regex": re.escape(query), "$options": "i"}
        cursor = patients_collection.find({
            "$or": [
                {"display_name": pattern},
                {"patient_code": pattern},
                {"phone_number": pattern},
            ]
        }, projection)

    return sorted(cursor, key=lambda p: p.get("created_at") or "", reverse=True)


def apply_field_group_update(code: str, group: str, set_by: str, raw_items: list[dict]) -> int:
    """Supersede all currently-active items in `group` for this patient, then
    append new items built from raw_items. Returns the number of new items.

    Uses a fetch -> mutate -> $set of the whole array. For a hackathon-scale
    document (a handful of items per group per patient) this is simple and
    plenty fast; a high-write-volume production version would instead do
    this with Mongo's arrayFilters ($[elem]) to avoid the read-then-write
    round trip, at the cost of more complex update expressions.
    """
    ItemModel = FIELD_GROUP_MODELS[group]

    patient = get_patient(code)
    if patient is None:
        raise ValueError(f"No patient with code {code}")

    # Build the new items first so we know their item_ids for superseded_by.
    new_items = [ItemModel(set_by=set_by, **raw).model_dump(mode="json") for raw in raw_items]
    new_ids = [item["item_id"] for item in new_items]

    existing_items = patient.get(group, [])
    updated_existing = []
    for item in existing_items:
        if item.get("active"):
            item = {**item, "active": False, "superseded_by": new_ids[0] if new_ids else None}
        updated_existing.append(item)

    final_array = updated_existing + new_items

    patients_collection.update_one(
        {"patient_code": code},
        {"$set": {group: final_array}},
    )
    return len(new_items)


def get_current_instructions(code: str) -> Optional[dict]:
    patient = get_patient(code)
    if patient is None:
        return None

    def active_items(group: str):
        return [item for item in patient.get(group, []) if item.get("active")]

    follow_up_active = active_items("follow_up")

    return {
        "patient_code": patient["patient_code"],
        "display_name": patient.get("display_name"),
        "procedure_name": patient.get("procedure_name"),
        "medications": active_items("medications"),
        "activity_restrictions": active_items("activity_restrictions"),
        "wound_care": sorted(active_items("wound_care"), key=lambda i: i.get("step_order", 0)),
        "red_flags": active_items("red_flags"),
        "follow_up": follow_up_active[-1] if follow_up_active else None,
        "general_notes": active_items("general_notes"),
    }


# ---------------------------------------------------------------------------
# Symptom reports — append-only, patient-logged, doctor-visible. Not part of
# the versioned field-group system above: nothing ever supersedes a report,
# every one the patient makes stays in the record.
# ---------------------------------------------------------------------------

def add_symptom_report(code: str, note: str) -> dict:
    report = SymptomReport(note=note).model_dump(mode="json")
    result = patients_collection.update_one(
        {"patient_code": code},
        {"$push": {"symptom_reports": report}},
    )
    if result.matched_count == 0:
        raise ValueError(f"No patient with code {code}")
    return report


def get_symptom_reports(code: str) -> Optional[list[dict]]:
    patient = get_patient(code)
    if patient is None:
        return None
    reports = patient.get("symptom_reports", [])
    return sorted(reports, key=lambda r: r.get("created_at", ""), reverse=True)