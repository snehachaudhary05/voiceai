"""
ASR Module - Speech to Text using NVIDIA Nemotron Speech ASR NIM
Records audio from microphone and transcribes it.
"""

import os
import io
import wave
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Recording settings
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.01      # RMS below this = silence
SILENCE_DURATION = 2.0        # Stop after 2 seconds of silence
MAX_DURATION = 30             # Max recording time in seconds


def record_audio() -> np.ndarray:
    """
    Records audio from the microphone.
    Automatically stops after 2 seconds of silence.
    Returns numpy array of audio samples.
    """
    print("\n Listening... (speak now, auto-stops on silence)")

    audio_chunks = []
    silence_counter = 0
    chunk_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    silence_chunks_needed = int(SILENCE_DURATION / 0.1)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype='float32', blocksize=chunk_size) as stream:
        speaking_started = False
        for _ in range(int(MAX_DURATION / 0.1)):
            chunk, _ = stream.read(chunk_size)
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > SILENCE_THRESHOLD:
                speaking_started = True
                silence_counter = 0
                audio_chunks.append(chunk.copy())
            elif speaking_started:
                audio_chunks.append(chunk.copy())
                silence_counter += 1
                if silence_counter >= silence_chunks_needed:
                    break

    if not audio_chunks:
        return None

    audio = np.concatenate(audio_chunks, axis=0).flatten()
    print(f" Recorded {len(audio) / SAMPLE_RATE:.1f}s of audio")
    return audio


def transcribe(audio: np.ndarray) -> str:
    """
    Sends audio to NVIDIA Nemotron Speech ASR NIM and returns transcript.
    """
    if audio is None or len(audio) == 0:
        return ""

    # Save audio to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')

        with open(tmp_path, 'rb') as f:
            audio_bytes = f.read()

        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
        }

        files = {
            "audio": ("audio.wav", audio_bytes, "audio/wav"),
            "model": (None, "nvidia/canary-1b"),          # Nemotron Speech ASR
            "language": (None, "auto"),                    # Auto language detection
            "response_format": (None, "text"),
        }

        response = requests.post(
            "https://integrate.api.nvidia.com/v1/audio/transcriptions",
            headers=headers,
            files=files,
            timeout=30
        )

        if response.status_code == 200:
            # response_format="text" returns plain text, not JSON
            try:
                result = response.json()
                if isinstance(result, dict):
                    text = result.get("text", "").strip()
                else:
                    text = str(result).strip()
            except Exception:
                text = response.text.strip()
            return text
        else:
            print(f" ASR Error {response.status_code}: {response.text}")
            return ""

    finally:
        os.unlink(tmp_path)


def listen_and_transcribe() -> str:
    """
    Full pipeline: record microphone → transcribe → return text.
    """
    audio = record_audio()
    if audio is None:
        print(" No audio detected.")
        return ""

    print(" Transcribing...")
    text = transcribe(audio)

    if text:
        print(f" You said: \"{text}\"")
    else:
        print(" Could not transcribe audio.")

    return text
