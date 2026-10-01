"""
Unit tests for transcription module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import os

from src.transcribe import (
    WordTimestamp,
    SegmentTimestamp,
    TranscriptData,
    calculate_speech_density,
    get_transcript_for_range,
    get_words_for_range,
    snap_to_word_boundary,
    TranscriptionError
)
from src.db import init_db, create_source, get_source


@pytest.fixture(autouse=True)
def setup_db():
    """Initialize test database."""
    test_db = "data/test_shortsforge.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    os.environ["SHORTSFORGE_DATABASE__PATH"] = test_db
    init_db()
    yield
    if os.path.exists(test_db):
        os.remove(test_db)


@pytest.fixture
def sample_transcript_data():
    """Create sample transcript data for testing."""
    words1 = [
        WordTimestamp(word="Hello", start=0.0, end=0.5, probability=0.99),
        WordTimestamp(word="world", start=0.5, end=1.0, probability=0.98),
    ]
    words2 = [
        WordTimestamp(word="This", start=2.0, end=2.2, probability=0.99),
        WordTimestamp(word="is", start=2.2, end=2.4, probability=0.97),
        WordTimestamp(word="a", start=2.4, end=2.5, probability=0.95),
        WordTimestamp(word="test", start=2.5, end=3.0, probability=0.99),
    ]
    words3 = [
        WordTimestamp(word="Goodbye", start=5.0, end=5.5, probability=0.99),
    ]

    segments = [
        SegmentTimestamp(id=0, start=0.0, end=1.0, text="Hello world", words=words1),
        SegmentTimestamp(id=1, start=2.0, end=3.0, text="This is a test", words=words2),
        SegmentTimestamp(id=2, start=5.0, end=5.5, text="Goodbye", words=words3),
    ]

    return TranscriptData(
        video_id="test_video",
        language="en",
        duration=10.0,
        speech_duration=2.5,  # 1.0 + 1.0 + 0.5
        speech_density=0.25,
        segments=segments,
        full_text="Hello world. This is a test. Goodbye."
    )


class TestSpeechDensityCalculation:
    """Tests for speech density calculation."""

    def test_calculate_speech_density(self, sample_transcript_data):
        """Test speech density calculation."""
        speech_duration, density = calculate_speech_density(
            sample_transcript_data.segments,
            sample_transcript_data.duration
        )
        assert speech_duration == 2.5
        assert density == 0.25

    def test_calculate_speech_density_zero_duration(self, sample_transcript_data):
        """Test speech density with zero total duration."""
        speech_duration, density = calculate_speech_density(
            sample_transcript_data.segments,
            0.0
        )
        assert density == 0.0

    def test_calculate_speech_density_empty_segments(self):
        """Test speech density with no segments."""
        speech_duration, density = calculate_speech_density([], 10.0)
        assert speech_duration == 0.0
        assert density == 0.0


class TestTranscriptRangeQueries:
    """Tests for getting transcript portions."""

    def test_get_transcript_for_range_full(self, sample_transcript_data):
        """Get transcript for full range."""
        text = get_transcript_for_range(sample_transcript_data, 0.0, 10.0)
        assert "Hello world" in text
        assert "This is a test" in text
        assert "Goodbye" in text

    def test_get_transcript_for_range_partial(self, sample_transcript_data):
        """Get transcript for partial range."""
        text = get_transcript_for_range(sample_transcript_data, 1.5, 3.5)
        assert "This is a test" in text
        assert "Hello world" not in text
        assert "Goodbye" not in text

    def test_get_transcript_for_range_no_overlap(self, sample_transcript_data):
        """Get transcript for range with no speech."""
        text = get_transcript_for_range(sample_transcript_data, 1.0, 1.5)
        assert text == ""

    def test_get_words_for_range(self, sample_transcript_data):
        """Get words for a time range."""
        words = get_words_for_range(sample_transcript_data, 2.0, 3.0)
        assert len(words) == 4
        assert words[0].word == "This"
        assert words[-1].word == "test"

    def test_get_words_for_range_partial_overlap(self, sample_transcript_data):
        """Get words with partial overlap."""
        words = get_words_for_range(sample_transcript_data, 2.1, 2.6)
        # Should include "is", "a", "test" (starting after 2.1)
        assert len(words) == 3
        assert words[0].word == "is"


class TestWordBoundarySnapping:
    """Tests for timestamp snapping to word boundaries."""

    def test_snap_before(self, sample_transcript_data):
        """Snap to word end before time."""
        # Time 0.3 should snap to "Hello" end at 0.5
        snapped = snap_to_word_boundary(sample_transcript_data, 0.3, "before")
        assert snapped == 0.5  # end of "Hello"

    def test_snap_after(self, sample_transcript_data):
        """Snap to word start after time."""
        # Time 0.3 should snap to "world" start at 0.5
        snapped = snap_to_word_boundary(sample_transcript_data, 0.3, "after")
        assert snapped == 0.5  # start of "world"

    def test_snap_nearest_start(self, sample_transcript_data):
        """Snap nearest to word start."""
        # Time 0.1 is closer to "Hello" start (0.0) than end (0.5)
        snapped = snap_to_word_boundary(sample_transcript_data, 0.1, "nearest")
        assert snapped == 0.0

    def test_snap_nearest_end(self, sample_transcript_data):
        """Snap nearest to word end."""
        # Time 0.4 is closer to "Hello" end (0.5) than start (0.0)
        snapped = snap_to_word_boundary(sample_transcript_data, 0.4, "nearest")
        assert snapped == 0.5

    def test_snap_no_words(self):
        """Snap with empty transcript."""
        empty_transcript = TranscriptData(
            video_id="test",
            language="en",
            duration=10.0,
            speech_duration=0.0,
            speech_density=0.0,
            segments=[],
            full_text=""
        )
        snapped = snap_to_word_boundary(empty_transcript, 5.0, "nearest")
        assert snapped == 5.0  # Should return original time


class TestTranscriptDataModel:
    """Tests for TranscriptData model validation."""

    def test_transcript_data_creation(self, sample_transcript_data):
        """Test TranscriptData can be created and serialized."""
        # Test serialization
        data_dict = sample_transcript_data.model_dump()
        assert data_dict['video_id'] == "test_video"
        assert len(data_dict['segments']) == 3

        # Test deserialization
        restored = TranscriptData(**data_dict)
        assert restored.video_id == "test_video"
        assert len(restored.segments) == 3
        assert restored.segments[0].words[0].word == "Hello"

    def test_word_timestamp_model(self):
        """Test WordTimestamp model."""
        word = WordTimestamp(word="test", start=1.0, end=1.5, probability=0.9)
        assert word.word == "test"
        assert word.start == 1.0
        assert word.end == 1.5
        assert word.probability == 0.9

    def test_segment_timestamp_model(self):
        """Test SegmentTimestamp model."""
        words = [WordTimestamp(word="hello", start=0.0, end=0.5, probability=0.99)]
        segment = SegmentTimestamp(id=0, start=0.0, end=0.5, text="hello", words=words)
        assert segment.id == 0
        assert segment.text == "hello"
        assert len(segment.words) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])