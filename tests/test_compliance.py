"""
Unit tests for compliance module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.compliance import (
    fetch_metadata,
    check_allowlist,
    check_compliance,
    ComplianceRejectionError,
    VideoMetadata,
    is_private_study,
    can_upload
)
from src.db import init_db, create_source, get_source


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize test database."""
    import os
    test_db = "data/test_shortsforge.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    os.environ["SHORTSFORGE_DATABASE__PATH"] = test_db
    init_db()
    yield
    if os.path.exists(test_db):
        os.remove(test_db)


@pytest.fixture
def mock_yt_dlp():
    """Mock yt_dlp YoutubeDL."""
    with patch('src.compliance.yt_dlp.YoutubeDL') as mock_ydl:
        yield mock_ydl


@pytest.fixture
def sample_metadata():
    """Sample video metadata."""
    return {
        'id': 'test_video_123',
        'title': 'Test Video Title',
        'channel_id': 'UC_test_channel',
        'channel': 'Test Channel',
        'duration': 300.0,
        'license': 'Creative Commons Attribution license (reuse allowed)',
    }


class TestFetchMetadata:
    """Tests for fetch_metadata function."""

    def test_fetch_metadata_success(self, mock_yt_dlp, sample_metadata):
        """Test successful metadata fetch."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        from src.compliance import fetch_metadata
        result = fetch_metadata("https://youtube.com/watch?v=test_video_123")

        assert isinstance(result, VideoMetadata)
        assert result.video_id == 'test_video_123'
        assert result.title == 'Test Video Title'
        assert result.channel_id == 'UC_test_channel'
        assert result.channel_title == 'Test Channel'
        assert result.duration == 300.0
        assert result.is_creative_commons is True

    def test_fetch_metadata_no_cc_license(self, mock_yt_dlp):
        """Test metadata fetch with non-CC license."""
        metadata = {
            'id': 'test_video_123',
            'title': 'Test Video',
            'channel_id': 'UC_test',
            'channel': 'Test Channel',
            'duration': 300.0,
            'license': 'Standard YouTube License',
        }
        mock_instance = Mock()
        mock_instance.extract_info.return_value = metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        result = fetch_metadata("https://youtube.com/watch?v=test_video_123")
        assert result.is_creative_commons is False

    def test_fetch_metadata_no_license(self, mock_yt_dlp):
        """Test metadata fetch with no license field."""
        metadata = {
            'id': 'test_video_123',
            'title': 'Test Video',
            'channel_id': 'UC_test',
            'channel': 'Test Channel',
            'duration': 300.0,
        }
        mock_instance = Mock()
        mock_instance.extract_info.return_value = metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        result = fetch_metadata("https://youtube.com/watch?v=test_video_123")
        assert result.license is None
        assert result.is_creative_commons is False


class TestCheckAllowlist:
    """Tests for check_allowlist function."""

    def test_check_allowlist_by_channel_id(self):
        """Test allowlist check by channel ID."""
        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_allowed_channel']
            assert check_allowlist('UC_allowed_channel', 'Some Channel') is True
            assert check_allowlist('UC_other_channel', 'Some Channel') is False

    def test_check_allowlist_by_channel_title(self):
        """Test allowlist check by channel title."""
        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['Allowed Channel Name']
            assert check_allowlist('UC_any_id', 'Allowed Channel Name') is True
            assert check_allowlist('UC_any_id', 'Other Channel') is False


class TestCheckCompliance:
    """Tests for check_compliance function."""

    def test_compliance_allowlist_pass(self, mock_yt_dlp, sample_metadata):
        """Test compliance passes via allowlist."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_test_channel']

            result = check_compliance("https://youtube.com/watch?v=test_video_123")

            assert result.video_id == 'test_video_123'
            source = get_source('test_video_123')
            assert source is not None
            assert source['compliance_status'] == 'passed'
            assert source['is_private_study'] == 0

    def test_compliance_creative_commons_pass(self, mock_yt_dlp, sample_metadata):
        """Test compliance passes via Creative Commons license."""
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            result = check_compliance("https://youtube.com/watch?v=test_video_123")

            assert result.video_id == 'test_video_123'
            source = get_source('test_video_123')
            assert source['compliance_status'] == 'passed'
            assert source['is_creative_commons'] == 1

    def test_compliance_manual_permission_pass(self, mock_yt_dlp, sample_metadata):
        """Test compliance passes via manual permission."""
        # Remove CC license
        sample_metadata['license'] = 'Standard YouTube License'
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            result = check_compliance(
                "https://youtube.com/watch?v=test_video_123",
                has_permission=True,
                permission_notes="Written permission from creator"
            )

            assert result.video_id == 'test_video_123'
            source = get_source('test_video_123')
            assert source['compliance_status'] == 'passed'
            assert source['has_permission'] == 1
            assert source['permission_notes'] == "Written permission from creator"

    def test_compliance_private_study_mode(self, mock_yt_dlp, sample_metadata):
        """Test compliance passes via private study mode."""
        sample_metadata['license'] = 'Standard YouTube License'
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            result = check_compliance(
                "https://youtube.com/watch?v=test_video_123",
                private_study=True
            )

            assert result.video_id == 'test_video_123'
            source = get_source('test_video_123')
            assert source['compliance_status'] == 'private_study'
            assert source['is_private_study'] == 1

    def test_compliance_fails_all_checks(self, mock_yt_dlp, sample_metadata):
        """Test compliance fails when all checks fail."""
        sample_metadata['license'] = 'Standard YouTube License'
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            with pytest.raises(ComplianceRejectionError):
                check_compliance("https://youtube.com/watch?v=test_video_123")

            source = get_source('test_video_123')
            assert source['compliance_status'] == 'failed'

    def test_compliance_manual_permission_requires_notes(self, mock_yt_dlp, sample_metadata):
        """Test manual permission requires notes."""
        sample_metadata['license'] = 'Standard YouTube License'
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            # has_permission=True but no notes - should fail
            with pytest.raises(ComplianceRejectionError):
                check_compliance(
                    "https://youtube.com/watch?v=test_video_123",
                    has_permission=True,
                    permission_notes=None
                )


class TestPrivateStudyAndUpload:
    """Tests for private study and upload checks."""

    def test_is_private_study(self, mock_yt_dlp, sample_metadata):
        """Test is_private_study function."""
        sample_metadata['license'] = 'Standard YouTube License'
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            check_compliance("https://youtube.com/watch?v=test_video_123", private_study=True)
            assert is_private_study('test_video_123') is True

            # Reset for non-private study
            import os
            os.remove("data/test_shortsforge.db")
            init_db()

            check_compliance("https://youtube.com/watch?v=test_video_123", has_permission=True, permission_notes="test")
            assert is_private_study('test_video_123') is False

    def test_can_upload(self, mock_yt_dlp, sample_metadata):
        """Test can_upload function."""
        sample_metadata['license'] = 'Standard YouTube License'
        mock_instance = Mock()
        mock_instance.extract_info.return_value = sample_metadata
        mock_yt_dlp.return_value.__enter__.return_value = mock_instance

        with patch('src.compliance.get_settings') as mock_settings:
            mock_settings.return_value.allowlist_channels = ['UC_other_channel']

            # Private study - cannot upload
            check_compliance("https://youtube.com/watch?v=test_video_123", private_study=True)
            assert can_upload('test_video_123') is False

            # Reset
            import os
            os.remove("data/test_shortsforge.db")
            init_db()

            # Allowlist - can upload
            mock_settings.return_value.allowlist_channels = ['UC_test_channel']
            check_compliance("https://youtube.com/watch?v=test_video_123")
            assert can_upload('test_video_123') is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])