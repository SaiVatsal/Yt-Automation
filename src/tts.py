"""
ShortsForge TTS Module

Stage 6: Text-to-Speech synthesis with Voicebox REST API provider,
Edge-TTS automatic fallback, hash-based caching, and atempo time-stretching.
"""

import os
import hashlib
import time
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests

from src.config import get_settings


class TTSError(Exception):
    """Raised when TTS synthesis fails."""
    pass


class BaseTTS(ABC):
    """Abstract base class for TTS providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is online and available."""
        pass

    @abstractmethod
    def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """
        Synthesize text to raw audio bytes (WAV or MP3).

        Args:
            text: Text to speak
            voice: Voice ID or profile name
            speed: Speed multiplier (1.0 = normal)

        Returns:
            Audio bytes
        """
        pass


class VoiceboxTTS(BaseTTS):
    """
    Local Voicebox REST API provider.
    Connects to local Voicebox service at http://127.0.0.1:17493 (or configured base_url).
    """

    def __init__(self, base_url: Optional[str] = None, profile_id: Optional[str] = None, language: str = "en"):
        settings = get_settings()
        self.base_url = (base_url or settings.tts.voicebox.base_url).rstrip("/")
        self.profile_id = profile_id or settings.tts.voicebox.profile_id
        self.language = language or settings.tts.voicebox.language

    def is_available(self) -> bool:
        """Check if Voicebox service is running and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_profiles(self) -> List[Dict[str, Any]]:
        """List available voice profiles from Voicebox."""
        try:
            resp = requests.get(f"{self.base_url}/profiles", timeout=5.0)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """
        Submit generation request to Voicebox, poll for completion, and return audio bytes.
        """
        target_profile = voice or self.profile_id
        payload = {
            "profile_id": target_profile,
            "text": text,
            "language": self.language,
            "speed": speed
        }

        try:
            # 1. Submit generation task
            gen_resp = requests.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=10.0
            )

            if gen_resp.status_code not in (200, 201, 202):
                raise TTSError(f"Voicebox /generate returned status {gen_resp.status_code}: {gen_resp.text}")

            res_data = gen_resp.json()

            # Handle direct audio response vs task ID polling
            if "audio" in res_data:
                # If audio is directly returned (base64 or direct bytes)
                import base64
                return base64.b64decode(res_data["audio"])

            task_id = res_data.get("id") or res_data.get("task_id") or res_data.get("generation_id")
            if not task_id:
                # If binary returned directly in response content
                if gen_resp.headers.get("content-type", "").startswith("audio/"):
                    return gen_resp.content
                raise TTSError(f"Voicebox did not return a valid task_id or audio: {res_data}")

            # 2. Poll for task status
            max_wait = 60.0
            start_time = time.time()
            poll_interval = 0.5

            while time.time() - start_time < max_wait:
                status_resp = requests.get(
                    f"{self.base_url}/generate/{task_id}/status",
                    timeout=5.0
                )
                if status_resp.status_code != 200:
                    time.sleep(poll_interval)
                    continue

                status_data = status_resp.json()
                state = status_data.get("status") or status_data.get("state")

                if state in ("completed", "done", "success"):
                    # Download completed audio
                    audio_url = status_data.get("audio_url")
                    if audio_url:
                        full_audio_url = audio_url if audio_url.startswith("http") else f"{self.base_url}{audio_url}"
                        audio_fetch = requests.get(full_audio_url, timeout=15.0)
                        return audio_fetch.content
                    elif "audio" in status_data:
                        import base64
                        return base64.b64decode(status_data["audio"])
                    else:
                        # Fetch /generate/{task_id}/audio
                        dl_resp = requests.get(f"{self.base_url}/generate/{task_id}/audio", timeout=15.0)
                        if dl_resp.status_code == 200:
                            return dl_resp.content
                    raise TTSError("Voicebox task completed but audio payload not found")

                elif state in ("failed", "error"):
                    error_msg = status_data.get("error", "Unknown Voicebox error")
                    raise TTSError(f"Voicebox task {task_id} failed: {error_msg}")

                time.sleep(poll_interval)

            raise TTSError(f"Voicebox generation timed out after {max_wait}s for task {task_id}")

        except requests.RequestException as e:
            raise TTSError(f"Voicebox connection error: {e}")


class EdgeTTS(BaseTTS):
    """
    Edge-TTS fallback provider.
    Synthesizes speech using Microsoft Edge online neural TTS.
    """

    def __init__(self, voice: Optional[str] = None):
        settings = get_settings()
        self.voice = voice or settings.tts.edge_tts.voice

    def is_available(self) -> bool:
        """Edge-TTS is available as long as network is reachable."""
        return True

    def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> bytes:
        """Synthesize text using edge_tts package."""
        target_voice = voice or self.voice
        rate_str = f"+{int((speed - 1.0) * 100)}%" if speed >= 1.0 else f"-{int((1.0 - speed) * 100)}%"

        import asyncio
        import edge_tts

        async def _synthesize_async() -> bytes:
            communicate = edge_tts.Communicate(text=text, voice=target_voice, rate=rate_str)
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            return b"".join(audio_chunks)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(lambda: asyncio.run(_synthesize_async())).result()
            else:
                return loop.run_until_complete(_synthesize_async())
        except Exception as e:
            raise TTSError(f"EdgeTTS synthesis failed: {e}")


def get_tts_cache_path(text: str, voice: str, speed: float) -> Path:
    """Generate deterministic cache path for TTS audio."""
    settings = get_settings()
    cache_dir = Path(settings.paths.work_dir) / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = f"{text}|{voice}|{speed:.2f}"
    hash_str = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{hash_str}.wav"


def convert_audio_to_wav(input_bytes: bytes, output_path: Path) -> Path:
    """Convert arbitrary input audio bytes to standard 44.1kHz mono WAV using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_input = output_path.with_suffix(".tmp_in")

    try:
        with open(temp_input, "wb") as f:
            f.write(input_bytes)

        cmd = [
            "ffmpeg", "-y", "-i", str(temp_input),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return output_path
    finally:
        if temp_input.exists():
            try:
                temp_input.unlink()
            except Exception:
                pass


def adjust_audio_tempo(input_wav: Path, output_wav: Path, speed_factor: float) -> Path:
    """
    Adjust audio tempo using ffmpeg atempo filter (max 1.15x).

    Args:
        input_wav: Input WAV file
        output_wav: Output WAV file
        speed_factor: Tempo multiplier (e.g. 1.05 to 1.15)
    """
    clamped_speed = min(1.15, max(0.85, speed_factor))
    cmd = [
        "ffmpeg", "-y", "-i", str(input_wav),
        "-filter:a", f"atempo={clamped_speed:.3f}",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        str(output_wav)
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    return output_wav


def get_audio_duration(file_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path)
            ],
            capture_output=True,
            text=True,
            timeout=15
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    target_duration: Optional[float] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    High-level speech synthesis function with provider selection, fallback,
    hash caching, and duration fitting (atempo).

    Args:
        text: Text to synthesize
        voice: Voice ID / Profile
        speed: Initial speech speed
        target_duration: Maximum duration in seconds allowed for this speech segment
        output_path: Optional explicit output destination

    Returns:
        Path to finalized WAV file
    """
    settings = get_settings()
    provider_name = settings.tts.provider.lower()
    voice_name = voice or (settings.tts.voicebox.profile_id if provider_name == "voicebox" else settings.tts.edge_tts.voice)

    # 1. Check cache
    cache_path = get_tts_cache_path(text, voice_name, speed)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        source_wav = cache_path
    else:
        # 2. Select Provider with Fallback
        audio_bytes = None
        if provider_name == "voicebox":
            voicebox = VoiceboxTTS()
            if voicebox.is_available():
                try:
                    audio_bytes = voicebox.synthesize(text, voice=voice_name, speed=speed)
                except Exception as e:
                    print(f"[Warning] Voicebox synthesis failed ({e}). Falling back to Edge-TTS.")
            else:
                print(f"[Warning] Voicebox is offline at {voicebox.base_url}. Falling back to Edge-TTS.")

        if audio_bytes is None:
            # Edge-TTS fallback
            edge = EdgeTTS()
            audio_bytes = edge.synthesize(text, voice=settings.tts.edge_tts.voice, speed=speed)

        # Convert to standard WAV and save to cache
        convert_audio_to_wav(audio_bytes, cache_path)
        source_wav = cache_path

    # 3. Fit Duration if target_duration is specified
    final_output = output_path or cache_path
    actual_duration = get_audio_duration(source_wav)

    if target_duration and actual_duration > target_duration:
        needed_factor = actual_duration / target_duration
        if needed_factor > 1.0:
            speedup = min(1.15, needed_factor)
            tempo_wav = final_output.with_suffix(".tempo.wav")
            adjust_audio_tempo(source_wav, tempo_wav, speedup)
            if final_output != tempo_wav:
                tempo_wav.replace(final_output)
            return final_output

    if output_path and output_path != source_wav:
        import shutil
        shutil.copy2(source_wav, output_path)
        return output_path

    return source_wav