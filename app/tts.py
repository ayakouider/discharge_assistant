"""
Text-to-speech via Deepgram Aura-2.

This replaces ElevenLabs TTS.

The function synthesize_speech_stream() returns an iterator of
audio chunks so FastAPI can stream the audio to the frontend
without waiting for the complete audio file.

Environment variables:

    DEEPGRAM_API_KEY=your_api_key
    DEEPGRAM_TTS_MODEL=aura-2-thalia-en

For a hackathon, the default model below is sufficient.
"""

import os
import requests


# ============================================================
# Configuration
# ============================================================

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# You can change this in .env without changing the code.
DEEPGRAM_TTS_MODEL = os.getenv(
    "DEEPGRAM_TTS_MODEL",
    "aura-2-thalia-en",
)

DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"

# One shared session for every TTS call. Without it, each request opens a
# brand-new connection to Deepgram (DNS + TCP + TLS handshake, typically
# 100-300 ms) before a single byte of audio can start. A Session keeps the
# connection alive and reuses it, so only the first request pays that cost.
_session = requests.Session()


# ============================================================
# Text-to-Speech
# ============================================================

def synthesize_speech_stream(text: str):
    """
    Convert text to speech using Deepgram Aura-2.

    Returns:
        Iterator[bytes]: streaming MP3 audio chunks.

    Raises:
        RuntimeError: if the API key is missing or Deepgram
        returns an error.
    """

    # --------------------------------------------------------
    # Validate API key
    # --------------------------------------------------------

    if not DEEPGRAM_API_KEY:
        raise RuntimeError(
            "DEEPGRAM_API_KEY is not set. "
            "Add your Deepgram API key to the .env file."
        )

    # --------------------------------------------------------
    # Validate text
    # --------------------------------------------------------

    if not text or not text.strip():
        raise RuntimeError(
            "TTS text cannot be empty."
        )

    # --------------------------------------------------------
    # Make Deepgram request
    # --------------------------------------------------------

    try:
        response = _session.post(
            DEEPGRAM_TTS_URL,
            params={
                "model": DEEPGRAM_TTS_MODEL,

                # MP3 works directly with browser audio playback.
                "encoding": "mp3",
            },
            headers={
                "Authorization": f"Token {DEEPGRAM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "text": text,
            },
            stream=True,
            timeout=30,
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Deepgram TTS: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Handle API errors
    # --------------------------------------------------------

    if not response.ok:
        try:
            error_body = response.text
        except Exception:
            error_body = "<unable to read response body>"

        raise RuntimeError(
            f"Deepgram TTS returned {response.status_code}: "
            f"{error_body}"
        )

    # --------------------------------------------------------
    # Return streaming audio
    # --------------------------------------------------------

    return response.iter_content(
        chunk_size=4096
    )