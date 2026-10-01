"""
ShortsForge YouTube Upload Module

Stage 9: YouTube Data API v3 OAuth2 authenticated resumable upload,
quota tracking (1600 units/upload), synthetic/AI media disclosure, and compliance enforcement.
"""

import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from src.config import get_settings
from src.db import (
    get_clip,
    get_source,
    create_upload,
    update_upload,
    increment_quota,
    check_quota_available
)


class YouTubeUploadError(Exception):
    """Raised when YouTube upload fails."""
    pass


# YouTube Data API v3 Scopes
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly"
]


def get_oauth_credentials() -> Credentials:
    """
    Retrieve or refresh Google OAuth2 user credentials.
    Looks for client_secrets in data/client_secrets.json or environment variables.
    Stores cached token in data/youtube_token.json.
    """
    settings = get_settings()
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    token_file = data_dir / "youtube_token.json"
    secrets_file = data_dir / "client_secrets.json"

    creds = None

    # 1. Load from cached token if exists
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
        except Exception as e:
            print(f"[Warning] Could not load token file: {e}")

    # 2. Refresh or authenticate if invalid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not secrets_file.exists():
                # Check environment variables
                client_id = os.getenv("YOUTUBE_CLIENT_ID")
                client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
                if client_id and client_secret:
                    client_config = {
                        "installed": {
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://oauth2.googleapis.com/token",
                            "redirect_uris": ["http://localhost"]
                        }
                    }
                    with open(secrets_file, "w", encoding="utf-8") as f:
                        json.dump(client_config, f, indent=2)
                else:
                    raise YouTubeUploadError(
                        f"Google OAuth client secrets not found at {secrets_file}. "
                        "Please place your client_secrets.json from Google Cloud Console in the data/ folder."
                    )

            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_file), SCOPES)
            creds = flow.run_local_server(port=0)

        # Save authorized credentials
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return creds


def get_youtube_client():
    """Build authenticated YouTube API service client."""
    creds = get_oauth_credentials()
    return build("youtube", "v3", credentials=creds)


def format_short_description(
    source_title: str,
    channel_name: str,
    source_url: str,
    has_narration: bool = True
) -> str:
    """Format compliant YouTube Shorts description with credits and AI disclosure."""
    desc_lines = [
        "#Shorts",
        "",
        f"Original Video: {source_title}",
        f"Source Channel: {channel_name}",
        f"Watch Original: {source_url}",
        ""
    ]

    if has_narration:
        desc_lines.extend([
            "---",
            "🤖 AI Voiceover: Narration synthesized with ShortsForge.",
            "Content has been transformed and edited for short-form analysis/entertainment."
        ])

    return "\n".join(desc_lines)


def upload_clip_to_youtube(
    clip_id: int,
    privacy_status: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute resumable upload of an approved clip to YouTube Shorts.

    Args:
        clip_id: ID of the clip to upload
        privacy_status: 'private', 'unlisted', or 'public' (defaults to config)

    Returns:
        Dict with uploaded YouTube video details
    """
    clip = get_clip(clip_id)
    if not clip:
        raise YouTubeUploadError(f"Clip {clip_id} does not exist.")

    source = get_source(clip["video_id"])
    if not source:
        raise YouTubeUploadError(f"Source video {clip['video_id']} not found.")

    # 1. Compliance Guard: Gating for Private Study Mode & Approval
    if source.get("is_private_study") == 1:
        raise YouTubeUploadError(
            "Upload Blocked: This source was processed under 'Private Study Mode'. "
            "Commercial or public uploading is strictly prohibited."
        )

    if clip.get("status") != "approved":
        raise YouTubeUploadError(f"Clip {clip_id} must be explicitly approved before uploading.")

    video_path = clip.get("output_video_path")
    if not video_path or not Path(video_path).exists():
        raise YouTubeUploadError(f"Rendered video file not found at {video_path}. Render clip first.")

    # 2. Daily Quota Guard
    settings = get_settings()
    needed_quota = settings.youtube.quota_per_upload
    if not check_quota_available(needed_quota):
        raise YouTubeUploadError(
            f"Daily YouTube quota limit exceeded. Need {needed_quota} units."
        )

    # 3. Prepare Metadata
    privacy = privacy_status or settings.privacy_default
    title = clip["title"]
    if not title.lower().endswith("#shorts") and "#shorts" not in title.lower():
        title = f"{title} #Shorts"[:100]

    has_narr = bool(clip.get("narration_script") or clip.get("narration_audio_path"))
    description = format_short_description(
        source_title=source.get("title", "Original Video"),
        channel_name=source.get("channel_title", "Original Creator"),
        source_url=source.get("url", ""),
        has_narration=has_narr
    )

    tags = ["Shorts", "YouTubeShorts", "ShortVideo"]
    if clip.get("hashtags"):
        try:
            extra_tags = json.loads(clip["hashtags"]) if isinstance(clip["hashtags"], str) else clip["hashtags"]
            tags.extend([t.replace("#", "") for t in extra_tags])
        except Exception:
            pass

    # Record upload initiation in DB
    upload_id = create_upload(
        clip_id=clip_id,
        title=title,
        description=description,
        tags=tags,
        privacy_status=privacy,
        synthetic_media_disclaimer=has_narr
    )

    # 4. Execute Resumable Upload
    youtube = get_youtube_client()
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "22"  # People & Blogs / General Entertainment
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        chunksize=1024 * 1024 * 5,  # 5MB chunks
        resumable=True
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    max_retries = 5
    retry_count = 0

    print(f"Uploading clip {clip_id} to YouTube Shorts ({privacy})...")

    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                print(f"Upload Progress: {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in [500, 502, 503, 504]:
                retry_count += 1
                if retry_count > max_retries:
                    update_upload(upload_id, upload_status="failed", error_message=str(e))
                    raise YouTubeUploadError(f"Upload failed after {max_retries} retries: {e}")
                sleep_sec = 2 ** retry_count
                print(f"Transient HTTP {e.resp.status}. Retrying in {sleep_sec}s...")
                time.sleep(sleep_sec)
            else:
                update_upload(upload_id, upload_status="failed", error_message=str(e))
                raise YouTubeUploadError(f"YouTube API Error ({e.resp.status}): {e}")

    # Success
    yt_video_id = response.get("id")
    increment_quota(needed_quota)

    update_upload(
        upload_id,
        youtube_video_id=yt_video_id,
        upload_status="uploaded"
    )

    print(f"Successfully uploaded to YouTube! Video ID: {yt_video_id}")
    return {
        "upload_id": upload_id,
        "youtube_video_id": yt_video_id,
        "privacy": privacy,
        "url": f"https://youtube.com/shorts/{yt_video_id}"
    }