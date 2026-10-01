"""
ShortsForge Pydantic Schemas

Shared data models for validation and type safety across modules.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class WordTimestamp(BaseModel):
    """Single word with timestamp."""
    word: str
    start: float
    end: float
    probability: float


class SegmentTimestamp(BaseModel):
    """Transcript segment with word timestamps."""
    id: int
    start: float
    end: float
    text: str
    words: List[WordTimestamp] = []


class TranscriptData(BaseModel):
    """Full transcript with metadata."""
    video_id: str
    language: str
    duration: float
    speech_duration: float
    speech_density: float
    segments: List[SegmentTimestamp]
    full_text: str


class ClipCandidate(BaseModel):
    """Single clip candidate from LLM extraction."""
    clip_index: int = Field(description="Sequential index 1, 2, ...")
    start_time: float = Field(description="Clip start in seconds")
    end_time: float = Field(description="Clip end in seconds")
    title: str = Field(description="Engaging title for YouTube Shorts (max 70 chars)", max_length=70)
    hook_text: str = Field(description="The first 3 seconds verbal/visual hook")
    hook_score: int = Field(description="Hook score 1-10", ge=1, le=10)
    clarity_score: int = Field(description="Clarity score 1-10", ge=1, le=10)
    payoff_score: int = Field(description="Payoff score 1-10", ge=1, le=10)
    overall_score: float = Field(description="Overall score 1-10", ge=1.0, le=10.0)
    rationale: str = Field(description="Why this clip is viral or high engagement")
    self_contained: bool = Field(description="Clip makes sense without context")
    needs_visual_check: bool = Field(description="Whether visual verification needed")
    visual_cues: Optional[List[str]] = Field(default_factory=list, description="Visual events in the segment")
    hashtags: Optional[List[str]] = Field(default_factory=list, description="Suggested hashtags")
    description: Optional[str] = Field(default=None, description="Video description")


class ClipExtractionResponse(BaseModel):
    """Response from clip extraction (Prompt A)."""
    mode: str = Field(description="'transcript' or 'vision'")
    clips: List[ClipCandidate]


class VisualEvent(BaseModel):
    """Visual event from vision analysis (Prompt B)."""
    t: float = Field(description="Timestamp in seconds")
    event: str = Field(description="Description of what happens on screen")
    intensity: int = Field(description="Action intensity 1-5", ge=1, le=5)
    key_moment: bool = Field(description="Whether this is a key moment")


class VisionAnalysisResponse(BaseModel):
    """Response from vision analysis (Prompt B)."""
    events: List[VisualEvent]
    summary: str


class NarrationSegment(BaseModel):
    """Single narration segment with timing."""
    start: float = Field(description="Start offset from clip start in seconds")
    end: float = Field(description="End offset from clip start in seconds")
    text: str = Field(description="Narration text for this segment")
    max_words: int = Field(description="Maximum words for this segment")


class NarrationScript(BaseModel):
    """Complete narration script for a clip."""
    total_words: int = Field(description="Total word count")
    segments: List[NarrationSegment]


# Prompt Templates

PROMPT_A_SYSTEM = """You are an expert short-form video editor. You receive a timestamped transcript and, optionally, visual notes. Pick the best standalone clips for YouTube Shorts.

RULES:
- Each clip is between {min_len} and {max_len} seconds.
- Start at a sentence or action boundary.
- The first 3 seconds must hook the viewer: a question, a surprising claim, a visible result, or strong action.
- End on a complete thought or clear payoff; never cut mid-explanation.
- The clip must make sense without the rest of the video; reject clips that depend on phrases like 'as I said earlier'.
- Prefer one clear idea, a useful tip, a mistake and its fix, a result reveal, a surprising fact, or a peak-action moment.
- Reject intros, outros, sponsor reads, filler, long pauses and setup without payoff.
- Clips must not overlap.
- Return at most {top_n} clips, best first.
- Use ONLY timestamps present in the input; never invent timestamps or content.
- Score each clip 1-10 on hook, clarity and payoff.
- Return ONLY valid JSON, no markdown.

Output schema:
{{
  "clips": [
    {{
      "clip_index": 1,
      "start_time": float,
      "end_time": float,
      "title": "str (max 70 chars)",
      "hook_text": "str",
      "hook_score": int,
      "clarity_score": int,
      "payoff_score": int,
      "overall_score": float,
      "rationale": "str",
      "self_contained": bool,
      "needs_visual_check": bool,
      "visual_cues": ["str"],
      "hashtags": ["str"],
      "description": "str"
    }}
  ]
}}"""


PROMPT_B_SYSTEM = """You are given frames from a video in order, each labeled with its timestamp in seconds. Describe what happens on screen moment by moment.

- Describe only what is visible; do not guess hidden information.
- Do not identify real people from their faces; name fictional characters only if a caption or on-screen text makes it clear who they are.
- Note actions, movement, impacts, on-screen text, UI changes and results.
- Mark the most important moments as key moments.
- Rate action intensity from 1 (calm) to 5 (peak).
- Return ONLY valid JSON.

Output schema:
{{
  "events": [
    {{
      "t": float,
      "event": "str",
      "intensity": int,
      "key_moment": bool
    }}
  ],
  "summary": "str"
}}"""


PROMPT_C_SYSTEM = """You write voiceover narration for a vertical short video.

INPUT: clip duration in seconds, transcript, visual events with timestamps, style note.

RULES:
- Word budget is duration x 2.5 minus 10 percent for pauses; never exceed it.
- Open with a hook sentence in the first 2 seconds.
- Do not just describe the action: explain HOW and WHY things happen and add one real insight, observation or tip, something the viewer can't get by watching alone.
- Match the timing: each segment must fit its time window and refer to what is on screen at that moment.
- Mention only what the transcript or visual events support; never invent details.
- Tone: friendly, warm, energetic, conversational.
- Short sentences, present tense, no emojis, no hashtags, no stage directions.
- Write numbers and acronyms so a voice reads them correctly (for example 'two hundred', 'A P I').
- Leave short gaps between segments.
- Return ONLY valid JSON.

Output schema:
{{
  "total_words": int,
  "segments": [
    {{
      "start": float,
      "end": float,
      "text": "str",
      "max_words": int
    }}
  ]
}}"""


def build_prompt_a(
    transcript: str,
    min_len: int,
    max_len: int,
    top_n: int,
    visual_notes: Optional[str] = None
) -> str:
    """Build Prompt A for clip extraction."""
    prompt = PROMPT_A_SYSTEM.format(
        min_len=min_len,
        max_len=max_len,
        top_n=top_n
    )

    user_content = f"Transcript:\n{transcript}\n\n"
    if visual_notes:
        user_content += f"Visual Notes:\n{visual_notes}\n\n"
    user_content += "Pick the best clips following the rules above."

    return prompt, user_content


def build_prompt_b(
    frames_description: str
) -> tuple:
    """Build Prompt B for vision analysis."""
    user_content = f"Frames with timestamps:\n{frames_description}"
    return PROMPT_B_SYSTEM, user_content


def build_prompt_c(
    clip_duration: float,
    transcript: str,
    visual_events: Optional[str] = None,
    style_note: str = "friendly, energetic explainer"
) -> tuple:
    """Build Prompt C for narration generation."""
    word_budget = int(clip_duration * 2.5 * 0.9)

    user_content = f"""Clip duration: {clip_duration:.1f} seconds
Word budget: {word_budget} words (duration × 2.5 × 0.9)
Style: {style_note}

Transcript for this clip:
{transcript}"""

    if visual_events:
        user_content += f"\n\nVisual events:\n{visual_events}"

    user_content += "\n\nWrite the narration script following the rules above."

    return PROMPT_C_SYSTEM, user_content