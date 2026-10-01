"""
ShortsForge Clip Analysis Module

Stage 4: Clip extraction using Claude (Prompt A) with Pydantic validation,
retry logic, boundary snapping, and overlap removal.
"""

import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from anthropic import Anthropic

from src.config import get_settings
from src.db import get_source, create_clip, get_clips
from src.transcribe import (
    load_cached_transcript,
    get_transcript_for_range,
    snap_to_word_boundary
)
from src.schemas import (
    ClipCandidate,
    ClipExtractionResponse,
    build_prompt_a
)


class AnalysisError(Exception):
    """Raised when clip analysis fails."""
    pass


@dataclass
class ValidatedClip:
    """Validated and processed clip candidate."""
    clip_index: int
    start_time: float
    end_time: float
    title: str
    hook_text: str
    hook_score: int
    clarity_score: int
    payoff_score: int
    overall_score: float
    rationale: str
    self_contained: bool
    needs_visual_check: bool
    visual_cues: List[str]
    hashtags: List[str]
    description: str


def get_claude_client() -> Anthropic:
    """Get Anthropic client with API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise AnalysisError("ANTHROPIC_API_KEY environment variable not set")
    return Anthropic(api_key=api_key)


def call_claude_prompt_a(
    transcript: str,
    visual_notes: Optional[str] = None,
    max_retries: int = 3
) -> ClipExtractionResponse:
    """
    Call Claude with Prompt A for clip extraction.

    Args:
        transcript: Full transcript text
        visual_notes: Optional visual analysis notes
        max_retries: Maximum retry attempts for invalid JSON

    Returns:
        Parsed ClipExtractionResponse

    Raises:
        AnalysisError: If all retries fail
    """
    settings = get_settings()
    client = get_claude_client()

    system_prompt, user_content = build_prompt_a(
        transcript=transcript,
        min_len=settings.clip_min_len,
        max_len=settings.clip_max_len,
        top_n=settings.top_n,
        visual_notes=visual_notes
    )

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )

            content = response.content[0].text.strip()

            # Try to parse JSON
            try:
                # Extract JSON if wrapped in markdown
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                parsed = ClipExtractionResponse.model_validate_json(content)
                return parsed

            except Exception as e:
                last_error = f"JSON validation failed: {e}"
                # Add error feedback for retry
                if attempt < max_retries - 1:
                    user_content += f"\n\nPrevious response failed validation: {last_error}. Please return strictly valid JSON matching the schema."

        except Exception as e:
            last_error = f"API call failed: {e}"
            if attempt == max_retries - 1:
                break

    raise AnalysisError(f"Failed to get valid clip extraction after {max_retries} attempts. Last error: {last_error}")


def snap_clip_boundaries(
    transcript,
    start_time: float,
    end_time: float,
    tolerance: float = 0.5
) -> tuple:
    """
    Snap clip boundaries to word boundaries.

    Args:
        transcript: TranscriptData object
        start_time: Proposed start time
        end_time: Proposed end time
        tolerance: Maximum snap distance

    Returns:
        (snapped_start, snapped_end)
    """
    snapped_start = snap_to_word_boundary(transcript, start_time, "before")
    snapped_end = snap_to_word_boundary(transcript, end_time, "after")

    # Ensure we didn't snap too far
    if abs(snapped_start - start_time) > tolerance:
        snapped_start = start_time
    if abs(snapped_end - end_time) > tolerance:
        snapped_end = end_time

    return snapped_start, snapped_end


def filter_clips_by_duration(
    clips: List[ValidatedClip],
    min_len: float,
    max_len: float
) -> List[ValidatedClip]:
    """Filter clips by duration constraints."""
    return [
        clip for clip in clips
        if min_len <= (clip.end_time - clip.start_time) <= max_len
    ]


def remove_overlapping_clips(
    clips: List[ValidatedClip],
    max_overlap_ratio: float = 0.1
) -> List[ValidatedClip]:
    """
    Remove overlapping clips using non-maximum suppression.

    Keeps higher-scored clips when overlap exceeds threshold.
    """
    if not clips:
        return []

    # Sort by overall score descending
    sorted_clips = sorted(clips, key=lambda c: c.overall_score, reverse=True)
    selected = []

    for clip in sorted_clips:
        overlap_found = False
        for selected_clip in selected:
            # Calculate overlap
            overlap_start = max(clip.start_time, selected_clip.start_time)
            overlap_end = min(clip.end_time, selected_clip.end_time)
            overlap_duration = max(0, overlap_end - overlap_start)

            clip_duration = clip.end_time - clip.start_time
            selected_duration = selected_clip.end_time - selected_clip.start_time

            overlap_ratio = overlap_duration / min(clip_duration, selected_duration)

            if overlap_ratio > max_overlap_ratio:
                overlap_found = True
                break

        if not overlap_found:
            selected.append(clip)

    # Re-index
    for i, clip in enumerate(selected):
        clip.clip_index = i + 1

    return selected


def convert_to_validated(
    candidates: List[ClipCandidate],
    transcript
) -> List[ValidatedClip]:
    """Convert Pydantic candidates to ValidatedClip with boundary snapping."""
    validated = []

    for cand in candidates:
        # Snap boundaries
        snapped_start, snapped_end = snap_clip_boundaries(
            transcript, cand.start_time, cand.end_time
        )

        validated.append(ValidatedClip(
            clip_index=cand.clip_index,
            start_time=snapped_start,
            end_time=snapped_end,
            title=cand.title,
            hook_text=cand.hook_text,
            hook_score=cand.hook_score,
            clarity_score=cand.clarity_score,
            payoff_score=cand.payoff_score,
            overall_score=cand.overall_score,
            rationale=cand.rationale,
            self_contained=cand.self_contained,
            needs_visual_check=cand.needs_visual_check,
            visual_cues=cand.visual_cues,
            hashtags=cand.hashtags,
            description=cand.description or ""
        ))

    return validated


def analyze_video(
    video_id: str,
    visual_notes: Optional[str] = None
) -> List[ValidatedClip]:
    """
    Run full clip analysis pipeline for a video.

    Args:
        video_id: Video ID to analyze
        visual_notes: Optional visual analysis notes from vision mode

    Returns:
        List of validated, filtered, non-overlapping clips
    """
    # Load transcript
    transcript = load_cached_transcript(video_id)
    if not transcript:
        raise AnalysisError(f"No transcript found for {video_id}. Run transcription first.")

    # Get transcript text
    full_transcript = transcript.full_text

    # Call Claude for clip extraction
    print(f"Analyzing video {video_id} with Claude...")
    response = call_claude_prompt_a(full_transcript, visual_notes)

    # Convert to validated clips with boundary snapping
    validated = convert_to_validated(response.clips, transcript)

    # Filter by duration
    settings = get_settings()
    validated = filter_clips_by_duration(validated, settings.clip_min_len, settings.clip_max_len)

    # Remove overlaps
    validated = remove_overlapping_clips(validated)

    # Limit to top_n
    validated = validated[:settings.top_n]

    # Save to database
    source = get_source(video_id)
    for clip in validated:
        create_clip(
            video_id=video_id,
            clip_index=clip.clip_index,
            start_time=clip.start_time,
            end_time=clip.end_time,
            title=clip.title,
            hook_text=clip.hook_text,
            rationale=clip.rationale,
            viral_score=clip.overall_score,
            visual_cues=clip.visual_cues,
            narration_script=None  # Will be filled by narrate.py
        )

    print(f"Found {len(validated)} clip candidates for {video_id}")
    return validated


def get_clip_transcript(
    video_id: str,
    start_time: float,
    end_time: float
) -> str:
    """Get transcript text for a specific clip range."""
    transcript = load_cached_transcript(video_id)
    if not transcript:
        return ""
    return get_transcript_for_range(transcript, start_time, end_time)