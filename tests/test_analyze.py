"""
Unit tests for clip analysis module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.analyze import (
    call_claude_prompt_a,
    snap_clip_boundaries,
    filter_clips_by_duration,
    remove_overlapping_clips,
    convert_to_validated,
    ValidatedClip,
    AnalysisError
)
from src.schemas import (
    ClipCandidate,
    ClipExtractionResponse,
    WordTimestamp,
    SegmentTimestamp,
    TranscriptData
)


@pytest.fixture
def sample_transcript():
    """Create sample transcript for testing."""
    words1 = [
        WordTimestamp(word="Welcome", start=0.0, end=0.6, probability=0.99),
        WordTimestamp(word="to", start=0.6, end=0.8, probability=0.99),
        WordTimestamp(word="the", start=0.8, end=1.0, probability=0.99),
        WordTimestamp(word="ultimate", start=1.0, end=1.5, probability=0.98),
        WordTimestamp(word="guide", start=1.5, end=2.0, probability=0.99),
    ]
    words2 = [
        WordTimestamp(word="Here", start=10.0, end=10.4, probability=0.99),
        WordTimestamp(word="is", start=10.4, end=10.6, probability=0.99),
        WordTimestamp(word="the", start=10.6, end=10.8, probability=0.99),
        WordTimestamp(word="secret", start=10.8, end=11.3, probability=0.99),
        WordTimestamp(word="trick", start=11.3, end=11.8, probability=0.99),
    ]
    words3 = [
        WordTimestamp(word="And", start=39.0, end=39.3, probability=0.99),
        WordTimestamp(word="that", start=39.3, end=39.6, probability=0.99),
        WordTimestamp(word="is", start=39.6, end=39.8, probability=0.99),
        WordTimestamp(word="how", start=39.8, end=40.0, probability=0.99),
        WordTimestamp(word="it", start=40.0, end=40.2, probability=0.99),
        WordTimestamp(word="works", start=40.2, end=40.6, probability=0.99),
    ]

    segments = [
        SegmentTimestamp(id=0, start=0.0, end=2.0, text="Welcome to the ultimate guide", words=words1),
        SegmentTimestamp(id=1, start=10.0, end=11.8, text="Here is the secret trick", words=words2),
        SegmentTimestamp(id=2, start=39.0, end=40.6, text="And that is how it works", words=words3),
    ]

    return TranscriptData(
        video_id="test_vid",
        language="en",
        duration=60.0,
        speech_duration=5.4,
        speech_density=0.09,
        segments=segments,
        full_text="Welcome to the ultimate guide. Here is the secret trick. And that is how it works."
    )


@pytest.fixture
def sample_candidates():
    """Create sample clip candidates."""
    return [
        ClipCandidate(
            clip_index=1,
            start_time=10.0,
            end_time=40.5,
            title="The Secret Trick Revealed",
            hook_text="Here is the secret trick you need to know",
            hook_score=9,
            clarity_score=8,
            payoff_score=9,
            overall_score=8.8,
            rationale="Great revelation with strong takeaway",
            self_contained=True,
            needs_visual_check=False,
            visual_cues=["Demonstration on screen"],
            hashtags=["#tips", "#viral"],
            description="Learn the secret trick now."
        ),
        ClipCandidate(
            clip_index=2,
            start_time=15.0,
            end_time=35.0,
            title="Overlap Trick",
            hook_text="Quick version of the trick",
            hook_score=7,
            clarity_score=7,
            payoff_score=7,
            overall_score=7.0,
            rationale="Shorter variant",
            self_contained=True,
            needs_visual_check=False,
            visual_cues=[],
            hashtags=["#quick"],
            description="Quick tip"
        ),
        ClipCandidate(
            clip_index=3,
            start_time=0.0,
            end_time=5.0,
            title="Too Short Intro",
            hook_text="Welcome to guide",
            hook_score=5,
            clarity_score=5,
            payoff_score=4,
            overall_score=4.6,
            rationale="Intro clip",
            self_contained=False,
            needs_visual_check=False,
            visual_cues=[],
            hashtags=[],
            description="Intro"
        )
    ]


class TestBoundarySnapping:
    """Tests for clip boundary snapping."""

    def test_snap_clip_boundaries(self, sample_transcript):
        """Test boundary snapping logic."""
        # 10.1 is close to 10.0 (word start) and 40.5 is close to 40.6 (word end)
        start, end = snap_clip_boundaries(sample_transcript, 10.1, 40.5, tolerance=0.5)
        assert start <= 10.4
        assert end >= 40.2


class TestDurationFiltering:
    """Tests for duration filtering."""

    def test_filter_clips_by_duration(self, sample_candidates, sample_transcript):
        """Filter out clips that are too short or too long."""
        validated = convert_to_validated(sample_candidates, sample_transcript)
        filtered = filter_clips_by_duration(validated, min_len=9, max_len=60)
        # Clip 1 (30.5s) and Clip 2 (20.0s) should pass, Clip 3 (5s) should fail
        assert len(filtered) == 2
        assert filtered[0].title == "The Secret Trick Revealed"
        assert filtered[1].title == "Overlap Trick"


class TestOverlapRemoval:
    """Tests for overlap removal (NMS)."""

    def test_remove_overlapping_clips(self, sample_candidates, sample_transcript):
        """Keep higher score clip when overlap occurs."""
        validated = convert_to_validated(sample_candidates, sample_transcript)
        filtered = filter_clips_by_duration(validated, min_len=9, max_len=60)
        # Clip 1 (10-40.5, score 8.8) and Clip 2 (15-35, score 7.0) overlap completely
        non_overlapping = remove_overlapping_clips(filtered, max_overlap_ratio=0.1)
        assert len(non_overlapping) == 1
        assert non_overlapping[0].title == "The Secret Trick Revealed"
        assert non_overlapping[0].clip_index == 1


class TestClaudePromptARetry:
    """Tests for Claude Prompt A call and retry mechanism."""

    @patch('src.analyze.get_claude_client')
    def test_call_claude_prompt_a_success(self, mock_client_getter):
        """Test successful Prompt A response parsing."""
        mock_client = Mock()
        mock_client_getter.return_value = mock_client

        json_response = {
            "mode": "transcript",
            "clips": [
                {
                    "clip_index": 1,
                    "start_time": 10.0,
                    "end_time": 30.0,
                    "title": "Mastering Python Generators",
                    "hook_text": "Did you know you can save 90% memory with one keyword?",
                    "hook_score": 9,
                    "clarity_score": 9,
                    "payoff_score": 9,
                    "overall_score": 9.0,
                    "rationale": "High value insight",
                    "self_contained": True,
                    "needs_visual_check": False,
                    "visual_cues": ["Code snippet"],
                    "hashtags": ["#python", "#coding"],
                    "description": "Save memory in Python."
                }
            ]
        }

        mock_msg = Mock()
        mock_msg.content = [Mock(text=json.dumps(json_response))]
        mock_client.messages.create.return_value = mock_msg

        result = call_claude_prompt_a("Some transcript text")
        assert isinstance(result, ClipExtractionResponse)
        assert len(result.clips) == 1
        assert result.clips[0].title == "Mastering Python Generators"

    @patch('src.analyze.get_claude_client')
    def test_call_claude_prompt_a_markdown_wrapped_json(self, mock_client_getter):
        """Test handling JSON wrapped in ```json ... ```."""
        mock_client = Mock()
        mock_client_getter.return_value = mock_client

        json_response = {
            "mode": "transcript",
            "clips": [
                {
                    "clip_index": 1,
                    "start_time": 5.0,
                    "end_time": 25.0,
                    "title": "Quick Fix",
                    "hook_text": "Stop making this beginner mistake!",
                    "hook_score": 8,
                    "clarity_score": 8,
                    "payoff_score": 8,
                    "overall_score": 8.0,
                    "rationale": "Clear mistake fix",
                    "self_contained": True,
                    "needs_visual_check": False,
                    "visual_cues": [],
                    "hashtags": ["#tips"],
                    "description": "Fix your mistake."
                }
            ]
        }

        wrapped_text = f"```json\n{json.dumps(json_response)}\n```"
        mock_msg = Mock()
        mock_msg.content = [Mock(text=wrapped_text)]
        mock_client.messages.create.return_value = mock_msg

        result = call_claude_prompt_a("Some transcript text")
        assert len(result.clips) == 1
        assert result.clips[0].title == "Quick Fix"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])