"""
Unit tests for narration script generation and word budget calculation.
"""

import pytest
from unittest.mock import Mock, patch
import json

from src.narrate import (
    calculate_word_budget,
    generate_narration_script,
    NarrationError
)
from src.schemas import NarrationScript, NarrationSegment
from src.db import init_db, create_clip, get_clip


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    import os
    db_file = tmp_path / "test_narrate.db"
    os.environ["SHORTSFORGE_DATABASE__PATH"] = str(db_file)
    init_db()


class TestWordBudget:
    """Tests for word budget formula."""

    def test_calculate_word_budget(self):
        # 30 seconds * 2.5 words/sec * 0.9 = 67.5 -> 67 words
        budget = calculate_word_budget(30.0)
        assert budget == 67

        # 10 seconds * 2.5 * 0.9 = 22.5 -> 22 words
        budget_10 = calculate_word_budget(10.0)
        assert budget_10 == 22


class TestNarrationScriptGeneration:
    """Tests for Prompt C narration generation."""

    @patch("src.narrate.get_claude_client")
    def test_generate_narration_script_success(self, mock_client_getter):
        clip_id = create_clip(
            video_id="test_vid",
            clip_index=1,
            start_time=0.0,
            end_time=20.0,
            title="Test Clip",
            hook_text="Check this out"
        )

        mock_client = Mock()
        mock_client_getter.return_value = mock_client

        response_payload = {
            "total_words": 35,
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Did you know this simple trick can double your speed?",
                    "max_words": 10
                },
                {
                    "start": 5.0,
                    "end": 20.0,
                    "text": "By optimizing your data structures, Python runs up to three times faster without any extra libraries.",
                    "max_words": 25
                }
            ]
        }

        mock_msg = Mock()
        mock_msg.content = [Mock(text=json.dumps(response_payload))]
        mock_client.messages.create.return_value = mock_msg

        script = generate_narration_script("test_vid", clip_id)
        assert isinstance(script, NarrationScript)
        assert script.total_words == 35
        assert len(script.segments) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])