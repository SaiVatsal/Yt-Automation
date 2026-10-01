"""
Unit tests for YouTube upload module, quota management, and private study gating.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import os

from src.upload import (
    format_short_description,
    upload_clip_to_youtube,
    YouTubeUploadError
)
from src.db import (
    init_db,
    create_source,
    create_clip,
    approve_clip,
    get_daily_quota_used,
    increment_quota
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db_file = tmp_path / "test_upload.db"
    os.environ["SHORTSFORGE_DATABASE__PATH"] = str(db_file)
    init_db()


class TestDescriptionFormatter:
    """Tests for compliant description and disclosure."""

    def test_format_short_description(self):
        desc = format_short_description(
            source_title="Original Great Video",
            channel_name="Tech Guy",
            source_url="https://youtube.com/watch?v=123",
            has_narration=True
        )

        assert "#Shorts" in desc
        assert "Original Great Video" in desc
        assert "Tech Guy" in desc
        assert "AI Voiceover" in desc


class TestUploadGating:
    """Tests for compliance gating and quota protection."""

    def test_upload_blocked_for_private_study(self, tmp_path):
        create_source(
            video_id="study_vid",
            url="https://youtube.com/watch?v=study",
            title="Study Video",
            channel_id="UC123",
            channel_title="Creator",
            duration=60.0,
            is_private_study=True,
            compliance_status="private_study"
        )
        fake_video = tmp_path / "fake.mp4"
        fake_video.write_bytes(b"dummy")

        clip_id = create_clip(
            video_id="study_vid",
            clip_index=1,
            start_time=0.0,
            end_time=15.0,
            title="Test Clip",
            hook_text="Hook"
        )
        approve_clip(clip_id)

        from src.db import update_clip
        update_clip(clip_id, output_video_path=str(fake_video))

        with pytest.raises(YouTubeUploadError, match="Private Study Mode"):
            upload_clip_to_youtube(clip_id)

    def test_upload_blocked_for_unapproved_clip(self, tmp_path):
        create_source(
            video_id="lic_vid",
            url="https://youtube.com/watch?v=lic",
            title="Licensed Video",
            channel_id="UC123",
            channel_title="Creator",
            duration=60.0,
            is_creative_commons=True,
            compliance_status="passed"
        )
        clip_id = create_clip(
            video_id="lic_vid",
            clip_index=1,
            start_time=0.0,
            end_time=15.0,
            title="Test Clip",
            hook_text="Hook"
        )
        # Not approved yet

        with pytest.raises(YouTubeUploadError, match="must be explicitly approved"):
            upload_clip_to_youtube(clip_id)


class TestQuotaTracking:
    """Tests for daily quota increments."""

    def test_quota_tracking(self):
        assert get_daily_quota_used() == 0
        increment_quota(1600)
        assert get_daily_quota_used() == 1600
        increment_quota(1600)
        assert get_daily_quota_used() == 3200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])