"""
AssemblyAI temporary token minting.

The browser must never hold the real ASSEMBLYAI_API_KEY (anyone could read
it from page source / devtools and use it themselves). Instead, the server
exchanges the real key for a short-lived, single-use token via AssemblyAI's
REST endpoint, and only that token is sent to the browser.

Docs: https://www.assemblyai.com/docs/streaming/authenticate-with-a-temporary-token
"""

import os
import requests

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
TOKEN_URL = "https://streaming.assemblyai.com/v3/token"


def mint_temporary_token(expires_in_seconds: int = 60) -> dict:
    """Returns {"token": "...", "expires_in_seconds": ...}.

    expires_in_seconds is the REDEMPTION window (how long the browser has to
    open the WebSocket using this token) — max allowed is 600. It is NOT the
    session length; once connected, the session can run much longer
    (max_session_duration_seconds, default 3 hours), which is plenty for a
    patient conversation.
    """
    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set. Get a key from assemblyai.com "
            "and set it as an environment variable before starting the server."
        )

    response = requests.get(
        TOKEN_URL,
        params={"expires_in_seconds": expires_in_seconds},
        headers={"Authorization": ASSEMBLYAI_API_KEY},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()