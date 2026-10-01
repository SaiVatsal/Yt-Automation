"""
ShortsForge Compliance Module

Stage 1: Pre-download metadata fetch and permission verification.
Checks channel allowlist, Creative Commons license, manual permission, or private study mode.
"""

import yt_dlp
from typing import Optional, Dict, Any
from dataclasses import dataclass

from src.config import get_settings
from src.db import (
    create_source,
    log_compliance_check,
    get_source,
)


class ComplianceRejectionError(Exception):
    """Raised when a video fails compliance checks."""
    pass


@dataclass
class VideoMetadata:
    """Video metadata extracted from yt-dlp."""
    video_id: str
    url: str
    title: str
    channel_id: str
    channel_title: str
    duration: float
    license: Optional[str]
    is_creative_commons: bool
    raw_metadata: Dict[str, Any]


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    # yt-dlp will handle this, but we can pre-extract for logging
    ydl_opts = {'quiet': True, 'extract_flat': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('id', '')


def fetch_metadata(url: str) -> VideoMetadata:
    """
    Fetch video metadata using yt-dlp without downloading.
    Returns VideoMetadata with compliance-relevant fields.
    """
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    video_id = info.get('id', '')
    title = info.get('title', 'Unknown Title')
    channel_id = info.get('channel_id', '')
    channel_title = info.get('channel', info.get('uploader', 'Unknown Channel'))
    duration = float(info.get('duration', 0))
    license = info.get('license', '')

    # Check for Creative Commons
    is_cc = False
    if license:
        license_lower = license.lower()
        if any(cc in license_lower for cc in ['creative commons', 'cc by', 'cc-by']):
            is_cc = True

    return VideoMetadata(
        video_id=video_id,
        url=url,
        title=title,
        channel_id=channel_id,
        channel_title=channel_title,
        duration=duration,
        license=license if license else None,
        is_creative_commons=is_cc,
        raw_metadata=info
    )


def check_allowlist(channel_id: str, channel_title: str) -> bool:
    """Check if channel is on the allowlist."""
    settings = get_settings()
    allowed = settings.allowlist_channels
    return channel_id in allowed or channel_title in allowed


def check_compliance(
    url: str,
    has_permission: bool = False,
    permission_notes: Optional[str] = None,
    private_study: bool = False
) -> VideoMetadata:
    """
    Run full compliance check on a video URL.

    Args:
        url: YouTube video URL
        has_permission: User claims to have permission (manual override)
        permission_notes: Notes about the permission (required if has_permission=True)
        private_study: Process for private study only (no upload)

    Returns:
        VideoMetadata if compliance passes

    Raises:
        ComplianceRejectionError: If video fails all compliance checks
    """
    # Fetch metadata
    metadata = fetch_metadata(url)

    # Check if already processed
    existing = get_source(metadata.video_id)
    if existing:
        return metadata

    settings = get_settings()
    passed_checks = []

    # Check 1: Channel Allowlist
    allowlist_passed = check_allowlist(metadata.channel_id, metadata.channel_title)
    log_compliance_check(
        metadata.video_id, 'allowlist', allowlist_passed,
        f"Channel '{metadata.channel_title}' ({metadata.channel_id}) {'is' if allowlist_passed else 'is not'} on allowlist",
        {'channel_id': metadata.channel_id, 'channel_title': metadata.channel_title, 'allowed_channels': settings.allowlist_channels}
    )
    if allowlist_passed:
        passed_checks.append('allowlist')

    # Check 2: Creative Commons License
    cc_passed = metadata.is_creative_commons
    log_compliance_check(
        metadata.video_id, 'license', cc_passed,
        f"Video license {'is' if cc_passed else 'is not'} Creative Commons",
        {'license': metadata.license, 'is_creative_commons': cc_passed}
    )
    if cc_passed:
        passed_checks.append('creative_commons')

    # Check 3: Manual Permission
    manual_passed = has_permission and bool(permission_notes)
    log_compliance_check(
        metadata.video_id, 'manual_override', manual_passed,
        f"Manual permission {'granted' if manual_passed else 'not granted or missing notes'}",
        {'has_permission': has_permission, 'permission_notes': permission_notes}
    )
    if manual_passed:
        passed_checks.append('manual_permission')

    # Check 4: Private Study Mode
    private_study_passed = private_study
    log_compliance_check(
        metadata.video_id, 'private_study', private_study_passed,
        f"Private study mode {'enabled' if private_study_passed else 'disabled'}",
        {'private_study': private_study}
    )
    if private_study_passed:
        passed_checks.append('private_study')

    # Determine overall compliance status
    if passed_checks:
        if 'private_study' in passed_checks and len(passed_checks) == 1:
            compliance_status = 'private_study'
        else:
            compliance_status = 'passed'
    else:
        compliance_status = 'failed'

    # Create source record
    create_source(
        video_id=metadata.video_id,
        url=metadata.url,
        title=metadata.title,
        channel_id=metadata.channel_id,
        channel_title=metadata.channel_title,
        duration=metadata.duration,
        license=metadata.license,
        is_creative_commons=metadata.is_creative_commons,
        has_permission=has_permission,
        permission_notes=permission_notes,
        is_private_study=private_study,
        compliance_status=compliance_status,
        raw_metadata=metadata.raw_metadata
    )

    if compliance_status == 'failed':
        raise ComplianceRejectionError(
            f"Video '{metadata.title}' failed compliance checks. "
            f"Channel not on allowlist, not Creative Commons, no manual permission granted. "
            f"Use --private-study flag for local processing only."
        )

    return metadata


def get_compliance_status(video_id: str) -> Optional[str]:
    """Get compliance status for a video."""
    source = get_source(video_id)
    return source['compliance_status'] if source else None


def is_private_study(video_id: str) -> bool:
    """Check if video is in private study mode."""
    source = get_source(video_id)
    return source['is_private_study'] == 1 if source else False


def can_upload(video_id: str) -> bool:
    """Check if video can be uploaded to YouTube (not private study)."""
    source = get_source(video_id)
    if not source:
        return False
    return source['is_private_study'] == 0 and source['compliance_status'] == 'passed'