"""
ShortsForge Download Module

Stage 2: Download video using yt-dlp with caching by video_id.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import yt_dlp

from src.config import get_settings
from src.db import get_source, update_source


class DownloadError(Exception):
    """Raised when video download fails."""
    pass


def get_work_dir(video_id: str) -> Path:
    """Get the work directory for a video."""
    settings = get_settings()
    return Path(settings.paths.work_dir) / video_id


def get_source_path(video_id: str) -> Path:
    """Get the path to the source video file."""
    work_dir = get_work_dir(video_id)
    return work_dir / "source.mp4"


def get_info_path(video_id: str) -> Path:
    """Get the path to the info JSON file."""
    work_dir = get_work_dir(video_id)
    return work_dir / "source.info.json"


def verify_video_file(filepath: Path) -> bool:
    """Verify video file exists and is valid using ffprobe."""
    if not filepath.exists() or filepath.stat().st_size == 0:
        return False

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration",
                "-of", "csv=p=0",
                str(filepath)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0 and result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def download_video(url: str, video_id: str) -> Path:
    """
    Download video using yt-dlp with caching.

    Args:
        url: YouTube video URL
        video_id: Video ID for caching

    Returns:
        Path to the downloaded video file

    Raises:
        DownloadError: If download fails
    """
    work_dir = get_work_dir(video_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    source_path = get_source_path(video_id)

    # Check cache
    if verify_video_file(source_path):
        print(f"Using cached video: {source_path}")
        return source_path

    # Configure yt-dlp
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(work_dir / 'source.%(ext)s'),
        'merge_output_format': 'mp4',
        'postprocessors': [
            {'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'},
        ],
        'writeinfojson': True,
        'writethumbnail': False,
        'quiet': False,
        'no_warnings': False,
        'progress_hooks': [progress_hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Verify download
        if not verify_video_file(source_path):
            raise DownloadError(f"Downloaded file verification failed: {source_path}")

        # Update database with local path
        update_source(video_id, local_source_path=str(source_path))

        print(f"Downloaded video to: {source_path}")
        return source_path

    except yt_dlp.DownloadError as e:
        raise DownloadError(f"yt-dlp download failed: {e}")
    except Exception as e:
        raise DownloadError(f"Unexpected error during download: {e}")


def progress_hook(d: dict) -> None:
    """Progress hook for yt-dlp downloads."""
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A')
        speed = d.get('_speed_str', 'N/A')
        print(f"\rDownloading: {percent} at {speed}", end='', flush=True)
    elif d['status'] == 'finished':
        print("\rDownload complete, processing...", flush=True)


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """
    Extract audio from video as 16kHz mono WAV for transcription.

    Args:
        video_path: Path to source video
        output_path: Path for output WAV file

    Returns:
        Path to extracted audio file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return output_path
    except subprocess.CalledProcessError as e:
        raise DownloadError(f"Audio extraction failed: {e.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise DownloadError("Audio extraction timed out")


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return 0.0