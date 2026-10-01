"""
Unit tests for clip rendering and subtitle generation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import os

from src.utils.subtitles import (
    format_ass_time,
    generate_ass_header,
    create_word_highlight_ass
)
from src.clip import (
    escape_ffmpeg_path,
    build_vertical_filtergraph,
    render_clip,
    ClipRenderError
)


class TestSubtitleFormatting:
    """Tests for ASS subtitle formatting and timestamps."""

    def test_format_ass_time(self):
        assert format_ass_time(0.0) == "0:00:00.00"
        assert format_ass_time(5.25) == "0:00:05.25"
        assert format_ass_time(65.5) == "0:01:05.50"
        assert format_ass_time(3661.12) == "1:01:01.12"

    def test_generate_ass_header(self):
        header = generate_ass_header(1080, 1920)
        assert "PlayResX: 1080" in header
        assert "PlayResY: 1920" in header
        assert "Style: Default" in header
        assert "Style: Active" in header

    def test_create_word_highlight_ass(self, tmp_path):
        words = [
            {"word": "Welcome", "start": 0.0, "end": 0.5},
            {"word": "to", "start": 0.5, "end": 0.8},
            {"word": "ShortsForge", "start": 0.8, "end": 1.5},
        ]
        out_ass = tmp_path / "test.ass"
        create_word_highlight_ass(words, out_ass, words_per_line=3)

        assert out_ass.exists()
        content = out_ass.read_text(encoding="utf-8")
        assert "Dialogue: 0," in content
        assert "{\\c&H0000FFFF\\b1}" in content  # Active word color
        assert "Welcome" in content
        assert "ShortsForge" in content


class TestFiltergraphBuilder:
    """Tests for FFmpeg filtergraph construction."""

    def test_escape_ffmpeg_path_windows(self):
        p = Path("C:/work/test.ass")
        escaped = escape_ffmpeg_path(p)
        assert "C\\:" in escaped or "c\\:" in escaped or "/" in escaped

    def test_build_vertical_filtergraph_blur_split(self):
        vf = build_vertical_filtergraph(crop_mode="blur_split", width=1080, height=1920)
        assert "boxblur=20:5" in vf
        assert "scale=1080:1920" in vf
        assert "overlay=(W-w)/2:(H-h)/2" in vf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])