"""
ShortsForge Video Clipping and Mixing Module

Stage 7: ffmpeg vertical formatting (1080x1920, 9:16), blurred background / face crop,
audio ducking (duck original audio to 25%), burned-in ASS captions, and rendering.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.config import get_settings
from src.db import get_clip, update_clip, get_source
from src.utils.subtitles import create_word_highlight_ass


class ClipRenderError(Exception):
    """Raised when video rendering fails."""
    pass


def escape_ffmpeg_path(path: Path) -> str:
    """Escape file path for ffmpeg filter graph (handles Windows drive letters and backslashes)."""
    p_str = str(path.resolve()).replace("\\", "/")
    # Colons in drive letters like C:/ must be escaped as C\:/ in ffmpeg filter arguments
    if ":" in p_str:
        drive, rest = p_str.split(":", 1)
        return f"{drive}\\:{rest}"
    return p_str


def build_vertical_filtergraph(
    crop_mode: str = "blur_split",
    width: int = 1080,
    height: int = 1920,
    ass_subtitles_path: Optional[Path] = None
) -> str:
    """
    Build ffmpeg complex video filtergraph for vertical 9:16 layout and subtitle burn-in.

    Layouts:
    1. blur_split:
       - Background: scale and crop to 1080x1920, boxblur=20:5, darken by 15%
       - Foreground: scale to width 1080 (maintaining aspect ratio), overlay vertically centered
    """
    if crop_mode == "blur_split":
        # Background: scale to fill 1080x1920, crop, blur, darken
        # Foreground: scale to fit 1080 width
        vf = (
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},boxblur=20:5,eq=brightness=-0.15[bg];"
            f"[0:v]scale={width}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vcomp]"
        )
        base_v = "[vcomp]"
    else:
        # Center crop directly to 9:16
        vf = (
            f"[0:v]scale=ih*9/16:ih:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}[vcomp]"
        )
        base_v = "[vcomp]"

    if ass_subtitles_path and ass_subtitles_path.exists():
        escaped_ass = escape_ffmpeg_path(ass_subtitles_path)
        vf += f";{base_v}ass='{escaped_ass}'[vout]"
    else:
        vf += f";{base_v}copy[vout]"

    return vf


def render_clip(
    video_id: str,
    clip_id: int,
    narration_audio_path: Optional[Path] = None,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None
) -> Path:
    """
    Render a final 1080x1920 vertical Short clip with video framing, audio ducking,
    and burned-in animated word captions.

    Args:
        video_id: Source video ID
        clip_id: Clip ID from database
        narration_audio_path: Optional path to synthesized voiceover WAV/MP3
        word_timestamps: List of word timestamps for caption highlight
        output_path: Optional custom output destination

    Returns:
        Path to rendered MP4 video file
    """
    clip = get_clip(clip_id)
    if not clip:
        raise ClipRenderError(f"Clip {clip_id} not found")

    source = get_source(video_id)
    if not source or not source.get("local_source_path"):
        raise ClipRenderError(f"Source video for {video_id} not found")

    source_video_path = Path(source["local_source_path"])
    if not source_video_path.exists():
        raise ClipRenderError(f"Source video file does not exist: {source_video_path}")

    settings = get_settings()
    start_time = float(clip["start_time"])
    end_time = float(clip["end_time"])
    duration = end_time - start_time

    # Setup output paths
    out_dir = Path(settings.paths.output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dest_video = output_path or (out_dir / f"clip_{clip['clip_index']}.mp4")

    # 1. Generate Subtitle File (.ass)
    ass_path = out_dir / f"clip_{clip['clip_index']}.ass"
    if word_timestamps:
        create_word_highlight_ass(
            words=word_timestamps,
            output_ass_path=ass_path,
            clip_start_offset=start_time
        )
    else:
        ass_path = None

    # 2. Build Video Filtergraph
    v_filter = build_vertical_filtergraph(
        crop_mode=settings.video_format.crop_mode,
        width=settings.video_format.width,
        height=settings.video_format.height,
        ass_subtitles_path=ass_path
    )

    # 3. Build Audio Mixing and Ducking Command
    has_narration = narration_audio_path and narration_audio_path.exists()
    mute_bg = settings.video_format.mute_bg_music
    ducking = settings.video_format.ducking_factor

    cmd = ["ffmpeg", "-y"]

    # Input 0: Cut source video with accurate seek
    cmd.extend(["-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}", "-i", str(source_video_path)])

    # Input 1 (optional): Narration audio
    if has_narration:
        cmd.extend(["-i", str(narration_audio_path)])

    # Construct Filter Complex
    if has_narration:
        if mute_bg:
            # Completely mute background, keep only narration
            filter_complex = f"{v_filter};[1:a]volume=1.0[aout]"
        else:
            # Duck background to ducking factor (e.g. 0.25) and mix with narration
            filter_complex = (
                f"{v_filter};"
                f"[0:a]volume={ducking:.2f}[bg_aud];"
                f"[1:a]volume=1.00[narr_aud];"
                f"[bg_aud][narr_aud]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
        audio_map = "[aout]"
    else:
        filter_complex = f"{v_filter};[0:a]volume=1.0[aout]"
        audio_map = "[aout]"

    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", audio_map,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(dest_video)
    ])

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise ClipRenderError(f"FFmpeg rendering failed: {proc.stderr}")

        # Update database record
        update_clip(
            clip_id,
            output_video_path=str(dest_video),
            subtitles_ass_path=str(ass_path) if ass_path else None,
            narration_audio_path=str(narration_audio_path) if narration_audio_path else None,
            status="rendered"
        )

        return dest_video

    except subprocess.SubprocessError as e:
        raise ClipRenderError(f"FFmpeg execution failed: {e}")