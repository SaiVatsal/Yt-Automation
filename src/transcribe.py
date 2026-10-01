"""
ShortsForge Transcription Module

Stage 3: Audio transcription using faster-whisper with word-level timestamps
and speech density calculation.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any

from faster_whisper import WhisperModel

from src.config import get_settings
from src.download import extract_audio, get_work_dir
from src.db import get_source, update_source
from src.schemas import WordTimestamp, SegmentTimestamp, TranscriptData


class TranscriptionError(Exception):
    """Raised when transcription fails."""
    pass


def get_transcript_path(video_id: str) -> Path:
    """Get the path to the transcript JSON file."""
    return get_work_dir(video_id) / "transcript.json"


def get_audio_path(video_id: str) -> Path:
    """Get the path to the extracted audio WAV file."""
    return get_work_dir(video_id) / "audio_16khz.wav"


def load_cached_transcript(video_id: str) -> Optional[TranscriptData]:
    """Load transcript from cache if exists."""
    transcript_path = get_transcript_path(video_id)
    if transcript_path.exists():
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return TranscriptData.model_validate(data)
        except (json.JSONDecodeError, TypeError, Exception):
            pass
    return None


def save_transcript(transcript: TranscriptData) -> None:
    """Save transcript to cache."""
    transcript_path = get_transcript_path(transcript.video_id)
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript.model_dump(), f, ensure_ascii=False, indent=2)


def calculate_speech_density(segments: List[SegmentTimestamp], total_duration: float) -> tuple:
    """
    Calculate speech duration and density.

    Returns:
        (speech_duration, speech_density)
    """
    speech_duration = sum(seg.end - seg.start for seg in segments)
    speech_density = speech_duration / total_duration if total_duration > 0 else 0.0
    return speech_duration, speech_density


def transcribe_video(
    video_id: str,
    video_path: Optional[Path] = None,
    force: bool = False
) -> TranscriptData:
    """
    Transcribe video audio using faster-whisper.

    Args:
        video_id: Video ID
        video_path: Path to video file (optional, will use cached if not provided)
        force: Force re-transcription even if cached

    Returns:
        TranscriptData with word-level timestamps and speech density
    """
    # Check cache first
    if not force:
        cached = load_cached_transcript(video_id)
        if cached:
            print(f"Using cached transcript for {video_id}")
            return cached

    # Get video path if not provided
    if video_path is None:
        source = get_source(video_id)
        if not source or not source.get('local_source_path'):
            raise TranscriptionError(f"No source video found for {video_id}")
        video_path = Path(source['local_source_path'])

    if not video_path.exists():
        raise TranscriptionError(f"Video file not found: {video_path}")

    # Extract audio
    audio_path = get_audio_path(video_id)
    print(f"Extracting audio to {audio_path}...")
    extract_audio(video_path, audio_path)

    # Load whisper model
    settings = get_settings()
    print(f"Loading Whisper model: {settings.whisper_model}...")
    model = WhisperModel(
        settings.whisper_model,
        device="auto",
        compute_type="default"
    )

    # Transcribe with word timestamps
    print("Transcribing...")
    segments, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=5
    )

    # Build structured transcript
    segment_list = []
    full_text_parts = []
    seg_id = 0

    for segment in segments:
        words = []
        if segment.words:
            for w in segment.words:
                words.append(WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability
                ))

        seg = SegmentTimestamp(
            id=seg_id,
            start=segment.start,
            end=segment.end,
            text=segment.text.strip(),
            words=words
        )
        segment_list.append(seg)
        full_text_parts.append(segment.text.strip())
        seg_id += 1

    # Calculate speech density
    total_duration = info.duration
    speech_duration, speech_density = calculate_speech_density(segment_list, total_duration)

    # Create transcript data
    transcript = TranscriptData(
        video_id=video_id,
        language=info.language,
        duration=total_duration,
        speech_duration=speech_duration,
        speech_density=speech_density,
        segments=segment_list,
        full_text=" ".join(full_text_parts)
    )

    # Save to cache
    save_transcript(transcript)

    # Update database
    update_source(
        video_id,
        speech_density=speech_density,
        transcript_path=str(get_transcript_path(video_id)),
        mode="transcript" if speech_density >= settings.speech_density_threshold else "vision"
    )

    print(f"Transcription complete. Language: {info.language}, Speech density: {speech_density:.2%}")
    return transcript


def get_transcript_for_range(
    transcript: TranscriptData,
    start_time: float,
    end_time: float
) -> str:
    """
    Get transcript text for a specific time range.

    Args:
        transcript: Full transcript data
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Transcript text for the range
    """
    parts = []
    for seg in transcript.segments:
        if seg.end <= start_time:
            continue
        if seg.start >= end_time:
            break
        # Segment overlaps with range
        parts.append(seg.text)

    return " ".join(parts)


def get_words_for_range(
    transcript: TranscriptData,
    start_time: float,
    end_time: float
) -> List[WordTimestamp]:
    """Get word timestamps for a specific time range."""
    words = []
    for seg in transcript.segments:
        for w in seg.words:
            if w.end <= start_time:
                continue
            if w.start >= end_time:
                break
            words.append(w)
    return words


def snap_to_word_boundary(
    transcript: TranscriptData,
    time: float,
    direction: str = "nearest"
) -> float:
    """
    Snap a timestamp to the nearest word boundary.

    Args:
        transcript: Full transcript data
        time: Timestamp to snap
        direction: "nearest", "before", or "after"

    Returns:
        Snapped timestamp
    """
    all_words = []
    for seg in transcript.segments:
        all_words.extend(seg.words)

    if not all_words:
        return time

    if direction == "before":
        # Find last word ending before or at time
        candidates = [w.end for w in all_words if w.end <= time]
        return max(candidates) if candidates else time
    elif direction == "after":
        # Find first word starting at or after time
        candidates = [w.start for w in all_words if w.start >= time]
        return min(candidates) if candidates else time
    else:
        # Nearest word boundary
        best = min(all_words, key=lambda w: min(abs(w.start - time), abs(w.end - time)))
        if abs(best.start - time) < abs(best.end - time):
            return best.start
        return best.end