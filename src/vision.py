"""
ShortsForge Vision Analysis Module

Stage 4B: Visual scene detection (PySceneDetect), audio energy peak detection (Librosa),
frame extraction (OpenCV), and Claude Vision/Event Analysis (Prompt B).
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
from scenedetect import SceneManager, open_video, ContentDetector
import librosa

from src.config import get_settings
from src.download import get_work_dir, get_source_path, extract_audio
from src.db import get_source, update_source
from src.analyze import get_claude_client
from src.schemas import (
    VisualEvent,
    VisionAnalysisResponse,
    build_prompt_b
)


class VisionError(Exception):
    """Raised when vision analysis fails."""
    pass


def get_scene_analysis_path(video_id: str) -> Path:
    """Get the path to scene analysis JSON file."""
    return get_work_dir(video_id) / "scene_analysis.json"


def get_frames_dir(video_id: str) -> Path:
    """Get the directory for extracted video frames."""
    frames_dir = get_work_dir(video_id) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


def detect_scene_cuts(video_path: Path, threshold: float = 27.0) -> List[Tuple[float, float]]:
    """
    Detect visual scene cuts using PySceneDetect.

    Returns:
        List of (start_seconds, end_seconds) tuples for each scene.
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    cuts = []
    for scene in scene_list:
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        cuts.append((start_sec, end_sec))

    return cuts


def detect_audio_energy_peaks(video_path: Path, video_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """
    Detect peak excitement/intensity moments using Librosa audio RMS envelope.

    Returns:
        List of dicts with timestamp, rms energy, and peak indicator.
    """
    work_dir = get_work_dir(video_id)
    wav_path = work_dir / "audio_energy.wav"
    extract_audio(video_path, wav_path)

    try:
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        # Find top energy frames
        peak_indices = np.argsort(rms)[-top_k:]
        peak_indices = np.sort(peak_indices)

        peaks = []
        for idx in peak_indices:
            peaks.append({
                "timestamp": round(float(times[idx]), 2),
                "energy": round(float(rms[idx]), 4)
            })
        return peaks
    except Exception as e:
        print(f"[Warning] Audio energy analysis warning: {e}")
        return []


def extract_sampled_frames(
    video_path: Path,
    video_id: str,
    fps: float = 1.0,
    max_frames: int = 40
) -> List[Dict[str, Any]]:
    """
    Extract video frames sampled uniformly or at key points with timestamps.

    Returns:
        List of dicts with 'timestamp', 'path', 'base64'
    """
    frames_dir = get_frames_dir(video_id)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VisionError(f"Could not open video file {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Step interval in video frames
    step = max(1, int(video_fps / fps))
    frame_items = []
    current_frame = 0

    while current_frame < total_frames and len(frame_items) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = round(current_frame / video_fps, 2)
        frame_filename = frames_dir / f"frame_{timestamp:.2f}s.jpg"

        # Resize frame to standard width for LLM inspection (e.g. 512px)
        h, w = frame.shape[:2]
        new_w = 512
        new_h = int(h * (new_w / w))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(str(frame_filename), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        with open(frame_filename, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        frame_items.append({
            "timestamp": timestamp,
            "path": str(frame_filename),
            "base64": b64_data
        })

        current_frame += step

    cap.release()
    return frame_items


def analyze_video_vision(
    video_id: str,
    video_path: Optional[Path] = None,
    max_retries: int = 3
) -> VisionAnalysisResponse:
    """
    Execute full Vision Mode analysis (PySceneDetect + Librosa + Claude Prompt B).

    Args:
        video_id: Video ID
        video_path: Path to local video
        max_retries: Retry attempts for JSON validation

    Returns:
        VisionAnalysisResponse with visual events and timestamps
    """
    # 1. Check Cache
    analysis_path = get_scene_analysis_path(video_id)
    if analysis_path.exists():
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return VisionAnalysisResponse.model_validate(data)
        except Exception:
            pass

    if video_path is None:
        source = get_source(video_id)
        if not source or not source.get("local_source_path"):
            raise VisionError(f"No source video for {video_id}")
        video_path = Path(source["local_source_path"])

    settings = get_settings()

    # 2. Extract scene cuts and energy peaks
    scene_cuts = detect_scene_cuts(video_path)
    energy_peaks = detect_audio_energy_peaks(video_path, video_id)
    frames = extract_sampled_frames(video_path, video_id, fps=settings.frame_fps)

    # 3. Call Claude Prompt B (Vision Mode)
    client = get_claude_client()
    system_prompt = build_prompt_b("")[0]

    # Construct Claude Multimodal message
    user_content = []
    user_content.append({
        "type": "text",
        "text": f"Here are {len(frames)} extracted video frames with timestamps, scene cuts: {scene_cuts[:10]}, and audio peaks: {energy_peaks[:5]}. Describe the moment-by-moment events strictly following the system rules."
    })

    # Add images
    for frame_info in frames[:20]:  # Limit to 20 keyframes for token efficiency
        user_content.append({
            "type": "text",
            "text": f"Timestamp: {frame_info['timestamp']}s"
        })
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_info["base64"]
            }
        })

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=settings.llm_model,
                max_tokens=4096,
                temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}]
            )

            raw_text = response.content[0].text.strip()
            # Clean markdown codeblocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            parsed = VisionAnalysisResponse.model_validate_json(raw_text)

            # Save to cache
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(parsed.model_dump(), f, indent=2)

            return parsed

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                user_content.append({
                    "type": "text",
                    "text": f"Previous JSON response failed schema validation: {e}. Please output valid JSON only."
                })

    raise VisionError(f"Claude Vision Prompt B failed after {max_retries} attempts: {last_error}")