"""
MongoDB connection.

Uses the real `pymongo` driver against MONGO_URI. For local development or
CI where no MongoDB server is running, set USE_MOCK_DB=true to swap in
`mongomock` — a drop-in fake that implements the same pymongo API, so no
application code needs to change between mock and real.

To point at a real database (local mongod, Docker, or MongoDB Atlas), just
set MONGO_URI, e.g.:
MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net" uvicorn app.main:app
"""

import os

USE_MOCK_DB = os.getenv("USE_MOCK_DB", "false").lower() == "true"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "discharge_assistant")


def _make_client():
    if USE_MOCK_DB:
        import mongomock
        return mongomock.MongoClient()
    import pymongo
    return pymongo.MongoClient(MONGO_URI)


client = _make_client()
db = client[DB_NAME]
patients_collection = db["patients"]


def init_db() -> None:
    """Create indexes. Safe to call repeatedly."""
    patients_collection.create_index("patient_code", unique=True)
    # sparse=True: only documents that HAVE a phone_number are checked for
    # uniqueness, so multiple patients with no phone number at all don't
    # collide with each other (they'd otherwise all be "null", which a
    # plain unique index would reject after the first one).
    patients_collection.create_index("phone_number", unique=True, sparse=True)