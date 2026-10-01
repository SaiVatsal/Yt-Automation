"""
ShortsForge Configuration Management

Loads settings from config.yaml with environment variable overrides.
Uses Pydantic Settings for validation and type safety.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class CaptionStyle(BaseSettings):
    font_name: str = "Arial-Bold"
    font_size: int = 28
    primary_color: str = "&H00FFFFFF"
    highlight_color: str = "&H0000FFFF"
    outline_color: str = "&H00000000"
    outline_width: int = 3
    margin_v: int = 180


class VideoFormat(BaseSettings):
    width: int = 1080
    height: int = 1920
    crop_mode: str = "blur_split"  # blur_split or face_track
    ducking_factor: float = 0.25
    mute_bg_music: bool = False


class VoiceboxConfig(BaseSettings):
    base_url: str = "http://127.0.0.1:17493"
    profile_id: str = "default_profile"
    language: str = "en"


class EdgeTTSConfig(BaseSettings):
    voice: str = "en-US-JennyNeural"


class TTSConfig(BaseSettings):
    provider: str = "voicebox"  # voicebox or edge_tts
    voicebox: VoiceboxConfig = VoiceboxConfig()
    edge_tts: EdgeTTSConfig = EdgeTTSConfig()


class DatabaseConfig(BaseSettings):
    path: str = "data/shortsforge.db"


class PathsConfig(BaseSettings):
    work_dir: str = "work"
    output_dir: str = "output"


class YouTubeConfig(BaseSettings):
    quota_daily_limit: int = 10000
    quota_per_upload: int = 1600


class Settings(BaseSettings):
    # Core clip settings
    clip_min_len: int = 9
    clip_max_len: int = 60
    top_n: int = 5

    # Transcription
    whisper_model: str = "small"
    frame_fps: int = 1
    speech_density_threshold: float = 0.35

    # Privacy & LLM
    privacy_default: str = "private"
    llm_model: str = "claude-sonnet-5-5"

    # Allowlist
    allowlist_channels: List[str] = ["MyOwnChannel", "ApprovedPartnerChannel"]

    # TTS
    tts: TTSConfig = TTSConfig()

    # Narration
    narration_enabled: bool = True
    narration_style: str = "friendly, energetic explainer"

    # Captions
    caption_style: CaptionStyle = CaptionStyle()

    # Video
    video_format: VideoFormat = VideoFormat()

    # Database & Paths
    database: DatabaseConfig = DatabaseConfig()
    paths: PathsConfig = PathsConfig()

    # YouTube
    youtube: YouTubeConfig = YouTubeConfig()

    model_config = SettingsConfigDict(
        env_prefix="SHORTSFORGE_",
        env_nested_delimiter="__",
        extra="ignore"
    )

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config.yaml") -> "Settings":
        """Load settings from YAML file, then apply environment variable overrides."""
        config_dict = {}
        if Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, loading from YAML if needed."""
    global _settings
    if _settings is None:
        _settings = Settings.load_from_yaml()
    return _settings


def reload_settings() -> Settings:
    """Force reload settings from YAML and environment."""
    global _settings
    _settings = Settings.load_from_yaml()
    return _settings