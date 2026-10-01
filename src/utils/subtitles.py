"""
ShortsForge Subtitle Utility

Generates styled ASS (Advanced SubStation Alpha) subtitles with word-level highlight
and karaoke timing tags for vertical YouTube Shorts.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import get_settings


def format_ass_time(seconds: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cc (centiseconds)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def generate_ass_header(width: int = 1080, height: int = 1920) -> str:
    """Generate standard ASS script header and styles."""
    settings = get_settings()
    cap = settings.caption_style

    return f"""[Script Info]
Title: ShortsForge Word Captions
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cap.font_name},{cap.font_size},{cap.primary_color},{cap.highlight_color},{cap.outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{cap.outline_width},2,2,30,30,{cap.margin_v},1
Style: Active,{cap.font_name},{cap.font_size + 4},{cap.highlight_color},&H00FFFFFF,{cap.outline_color},&H80000000,-1,0,0,0,105,105,0,0,1,{cap.outline_width + 1},2,2,30,30,{cap.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def create_word_highlight_ass(
    words: List[Dict[str, Any]],
    output_ass_path: Path,
    words_per_line: int = 4,
    clip_start_offset: float = 0.0
) -> Path:
    """
    Generate an ASS subtitle file where current active spoken word is highlighted in yellow
    and slightly scaled up.

    Args:
        words: List of dicts with 'word', 'start', 'end' (in clip-relative seconds)
        output_ass_path: Destination .ass file path
        words_per_line: Number of words displayed simultaneously in a cluster
        clip_start_offset: Base offset to subtract if words timestamps are absolute video-relative
    """
    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    header = generate_ass_header()
    dialogue_lines = []

    if not words:
        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(header)
        return output_ass_path

    # Normalize timestamps to clip-relative
    rel_words = []
    for w in words:
        s = max(0.0, w["start"] - clip_start_offset)
        e = max(s + 0.1, w["end"] - clip_start_offset)
        rel_words.append({"word": w["word"].strip(), "start": s, "end": e})

    # Group words into small clusters / lines (3-5 words)
    chunks = [rel_words[i:i + words_per_line] for i in range(0, len(rel_words), words_per_line)]

    for chunk in chunks:
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]

        # For each word in this chunk, create a sub-event where this word is highlighted
        for active_idx, active_word in enumerate(chunk):
            w_start = active_word["start"]
            w_end = active_word["end"]

            # If there's a next word, ensure end time connects cleanly
            if active_idx < len(chunk) - 1:
                w_end = min(w_end, chunk[active_idx + 1]["start"])

            start_str = format_ass_time(w_start)
            end_str = format_ass_time(w_end)

            # Build line with active word formatted in yellow/bold
            line_parts = []
            for idx, item in enumerate(chunk):
                if idx == active_idx:
                    # Active highlighted word
                    line_parts.append(f"{{\\c&H0000FFFF\\b1}}{item['word']}{{\\r}}")
                else:
                    # Inactive word
                    line_parts.append(f"{{\\c&H00FFFFFF}}{item['word']}{{\\r}}")

            styled_text = " ".join(line_parts)
            dialogue_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{styled_text}")

    content = header + "\n".join(dialogue_lines) + "\n"
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_ass_path