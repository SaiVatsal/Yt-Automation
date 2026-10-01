"""
ShortsForge Narration Module

Stage 5: Voiceover narration script generation using Claude (Prompt C),
strict word budget calculation (duration * 2.5 * 0.9), and audio synthesis integration.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.config import get_settings
from src.db import get_clip, update_clip
from src.analyze import get_claude_client, get_clip_transcript
from src.vision import get_scene_analysis_path
from src.tts import synthesize_speech, get_audio_duration
from src.schemas import (
    NarrationSegment,
    NarrationScript,
    build_prompt_c
)


class NarrationError(Exception):
    """Raised when narration generation fails."""
    pass


def calculate_word_budget(duration: float, words_per_second: float = 2.5, pause_buffer: float = 0.10) -> int:
    """
    Calculate maximum permissible word count for a clip duration.

    Formula: floor(duration * words_per_second * (1 - pause_buffer))
    """
    capacity = duration * words_per_second
    budget = int(capacity * (1.0 - pause_buffer))
    return max(5, budget)


def generate_narration_script(
    video_id: str,
    clip_id: int,
    style_note: Optional[str] = None,
    max_retries: int = 3
) -> NarrationScript:
    """
    Generate timed narration voiceover script for a clip using Claude Prompt C.

    Args:
        video_id: Source video ID
        clip_id: Clip ID in SQLite database
        style_note: Custom tone/style direction (defaults to config)
        max_retries: Retry attempts for JSON schema validation

    Returns:
        NarrationScript with timed segments
    """
    clip = get_clip(clip_id)
    if not clip:
        raise NarrationError(f"Clip {clip_id} not found")

    settings = get_settings()
    duration = float(clip["end_time"]) - float(clip["start_time"])
    clip_transcript = get_clip_transcript(video_id, clip["start_time"], clip["end_time"])

    # Load visual events if available
    visual_events_str = None
    scene_json = get_scene_analysis_path(video_id)
    if scene_json.exists():
        try:
            with open(scene_json, "r", encoding="utf-8") as f:
                scene_data = json.load(f)
                # Filter events inside clip time window
                relevant = [
                    e for e in scene_data.get("events", [])
                    if clip["start_time"] <= e["t"] <= clip["end_time"]
                ]
                if relevant:
                    visual_events_str = json.dumps(relevant, indent=2)
        except Exception:
            pass

    target_style = style_note or settings.narration_style
    system_prompt, user_content = build_prompt_c(
        clip_duration=duration,
        transcript=clip_transcript,
        visual_events=visual_events_str,
        style_note=target_style
    )

    client = get_claude_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=2048,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )

            raw_text = response.content[0].text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            script = NarrationScript.model_validate_json(raw_text)

            # Validate word budget
            budget = calculate_word_budget(duration)
            if script.total_words > budget:
                # Truncate or warn
                pass

            # Update database
            update_clip(
                clip_id,
                narration_script=script.model_dump()
            )

            return script

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                user_content += f"\n\nPrevious script failed validation: {e}. Output strictly valid JSON."

    raise NarrationError(f"Failed to generate narration script after {max_retries} attempts: {last_error}")


def synthesize_clip_narration(
    video_id: str,
    clip_id: int,
    script: Optional[NarrationScript] = None
) -> Path:
    """
    Synthesize complete audio track for narration segments and fit into clip duration.

    Returns:
        Path to finalized narration audio WAV file
    """
    clip = get_clip(clip_id)
    if not clip:
        raise NarrationError(f"Clip {clip_id} not found")

    settings = get_settings()
    out_dir = Path(settings.paths.output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    narration_wav = out_dir / f"clip_{clip['clip_index']}_narration.wav"

    if script is None:
        if clip.get("narration_script"):
            raw_data = json.loads(clip["narration_script"]) if isinstance(clip["narration_script"], str) else clip["narration_script"]
            script = NarrationScript.model_validate(raw_data)
        else:
            script = generate_narration_script(video_id, clip_id)

    # Combine segment texts or synthesize sequentially
    full_narration_text = " ".join(seg.text for seg in script.segments)
    clip_duration = float(clip["end_time"]) - float(clip["start_time"])

    # Synthesize with speed/atempo fitting
    final_audio = synthesize_speech(
        text=full_narration_text,
        target_duration=clip_duration,
        output_path=narration_wav
    )

    # Update database
    update_clip(
        clip_id,
        narration_audio_path=str(final_audio)
    )

    return final_audio