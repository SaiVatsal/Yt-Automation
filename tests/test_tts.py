"""
Unit tests for TTS module (Voicebox + Edge-TTS fallback).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import os
import hashlib

from src.tts import (
    BaseTTS,
    VoiceboxTTS,
    EdgeTTS,
    get_tts_cache_path,
    synthesize_speech,
    TTSError
)


class TestTTSCaching:
    """Tests for deterministic TTS caching."""

    def test_get_tts_cache_path(self):
        text = "Hello world"
        voice = "test_voice"
        speed = 1.0
        path = get_tts_cache_path(text, voice, speed)

        key = f"{text}|{voice}|{speed:.2f}"
        expected_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

        assert path.name == f"{expected_hash}.wav"


class TestVoiceboxTTS:
    """Tests for Voicebox REST API provider."""

    @patch("requests.get")
    def test_voicebox_health_check_available(self, mock_get):
        mock_get.return_value.status_code = 200
        client = VoiceboxTTS("http://127.0.0.1:17493")
        assert client.is_available() is True

    @patch("requests.get")
    def test_voicebox_health_check_unavailable(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        client = VoiceboxTTS("http://127.0.0.1:17493")
        assert client.is_available() is False

    @patch("requests.get")
    @patch("requests.post")
    def test_voicebox_synthesize_task_polling(self, mock_post, mock_get):
        # 1. Post to /generate returns task ID
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "task_abc_123"}

        # 2. Status polling: first pending, then completed with audio download url
        resp_pending = Mock(status_code=200)
        resp_pending.json.return_value = {"status": "processing"}

        resp_completed = Mock(status_code=200)
        resp_completed.json.return_value = {"status": "completed", "audio_url": "/download/abc.wav"}

        resp_audio = Mock(status_code=200, content=b"RIFF_SAMPLE_WAV_BYTES")

        mock_get.side_effect = [resp_pending, resp_completed, resp_audio]

        client = VoiceboxTTS("http://127.0.0.1:17493")
        audio_bytes = client.synthesize("Test speech", voice="default_profile")

        assert audio_bytes == b"RIFF_SAMPLE_WAV_BYTES"
        mock_post.assert_called_once()


class TestEdgeTTS:
    """Tests for EdgeTTS fallback provider."""

    def test_edge_tts_availability(self):
        client = EdgeTTS()
        assert client.is_available() is True


class TestHighLevelSynthesize:
    """Tests for high-level speech synthesis and fallback logic."""

    @patch("src.tts.convert_audio_to_wav")
    @patch("src.tts.get_audio_duration")
    @patch.object(VoiceboxTTS, "is_available", return_value=False)
    @patch.object(EdgeTTS, "synthesize", return_value=b"RAW_EDGE_BYTES")
    def test_fallback_to_edge_tts_when_voicebox_offline(
        self, mock_edge_synth, mock_vb_avail, mock_get_dur, mock_convert, tmp_path
    ):
        mock_get_dur.return_value = 2.5
        mock_convert.side_effect = lambda raw, out: out

        out_file = tmp_path / "test_out.wav"
        result = synthesize_speech(
            text="Testing fallback",
            voice="default",
            speed=1.0,
            output_path=out_file
        )

        mock_edge_synth.assert_called_once()
        assert result == out_file


if __name__ == "__main__":
    pytest.main([__file__, "-v"])