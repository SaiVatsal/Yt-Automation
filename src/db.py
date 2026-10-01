"""
ShortsForge Database Layer

SQLite schema initialization, models, and CRUD operations.
Uses sqlite3 directly with context managers for thread safety.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from src.config import get_settings


SCHEMA_SQL = """
-- Sources table: stores video source metadata and compliance status
CREATE TABLE IF NOT EXISTS sources (
    video_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    channel_title TEXT NOT NULL,
    duration REAL NOT NULL,
    license TEXT,
    is_creative_commons INTEGER NOT NULL DEFAULT 0,
    has_permission INTEGER NOT NULL DEFAULT 0,
    permission_notes TEXT,
    is_private_study INTEGER NOT NULL DEFAULT 0,
    compliance_status TEXT NOT NULL, -- 'passed', 'failed', 'private_study'
    speech_density REAL,
    mode TEXT, -- 'transcript', 'vision'
    local_source_path TEXT,
    transcript_path TEXT,
    raw_metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Compliance logs: audit trail for compliance checks
CREATE TABLE IF NOT EXISTS compliance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    check_type TEXT NOT NULL, -- 'allowlist', 'license', 'manual_override', 'private_study'
    passed INTEGER NOT NULL,
    reason TEXT NOT NULL,
    details_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(video_id) REFERENCES sources(video_id) ON DELETE CASCADE
);

-- Clips table: stores clip candidates and their processing status
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    clip_index INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    title TEXT NOT NULL,
    hook_text TEXT NOT NULL,
    rationale TEXT,
    viral_score REAL,
    visual_cues_json TEXT,
    narration_script TEXT,
    narration_audio_path TEXT,
    subtitles_ass_path TEXT,
    output_video_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending_review', -- 'pending_review', 'approved', 'rejected', 'rendered', 'uploaded'
    approval_notes TEXT,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(video_id) REFERENCES sources(video_id) ON DELETE CASCADE
);

-- Uploads table: tracks YouTube uploads
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id INTEGER NOT NULL,
    youtube_video_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    tags_json TEXT,
    privacy_status TEXT NOT NULL, -- 'public', 'unlisted', 'private'
    synthetic_media_disclaimer INTEGER NOT NULL DEFAULT 1,
    upload_status TEXT NOT NULL, -- 'pending', 'uploaded', 'failed', 'blocked_private_study'
    quota_units_used INTEGER NOT NULL DEFAULT 1600,
    uploaded_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(clip_id) REFERENCES clips(id) ON DELETE CASCADE
);

-- Quota tracker: daily YouTube API quota usage
CREATE TABLE IF NOT EXISTS quota_tracker (
    date_key TEXT PRIMARY KEY, -- 'YYYY-MM-DD'
    units_used INTEGER NOT NULL DEFAULT 0,
    daily_limit INTEGER NOT NULL DEFAULT 10000,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_clips_video_id ON clips(video_id);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
CREATE INDEX IF NOT EXISTS idx_uploads_clip_id ON uploads(clip_id);
CREATE INDEX IF NOT EXISTS idx_compliance_logs_video_id ON compliance_logs(video_id);
"""


@contextmanager
def get_connection():
    """Context manager for SQLite connection with row factory."""
    settings = get_settings()
    db_path = settings.database.path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize database schema."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        # Update daily limit from config
        settings = get_settings()
        conn.execute(
            "UPDATE quota_tracker SET daily_limit = ? WHERE date_key = ?",
            (settings.youtube.quota_daily_limit, datetime.now().strftime("%Y-%m-%d"))
        )


def execute(query: str, params: tuple = ()) -> sqlite3.Cursor:
    """Execute a query and return cursor."""
    with get_connection() as conn:
        return conn.execute(query, params)


def fetchone(query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    """Fetch a single row."""
    with get_connection() as conn:
        return conn.execute(query, params).fetchone()


def fetchall(query: str, params: tuple = ()) -> List[sqlite3.Row]:
    """Fetch all rows."""
    with get_connection() as conn:
        return conn.execute(query, params).fetchall()


# --- Sources CRUD ---

def create_source(
    video_id: str,
    url: str,
    title: str,
    channel_id: str,
    channel_title: str,
    duration: float,
    license: Optional[str] = None,
    is_creative_commons: bool = False,
    has_permission: bool = False,
    permission_notes: Optional[str] = None,
    is_private_study: bool = False,
    compliance_status: str = "pending",
    speech_density: Optional[float] = None,
    mode: Optional[str] = None,
    local_source_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    raw_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Insert or update a source record."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sources (
                video_id, url, title, channel_id, channel_title, duration,
                license, is_creative_commons, has_permission, permission_notes,
                is_private_study, compliance_status, speech_density, mode,
                local_source_path, transcript_path, raw_metadata_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                url=excluded.url,
                title=excluded.title,
                channel_id=excluded.channel_id,
                channel_title=excluded.channel_title,
                duration=excluded.duration,
                license=excluded.license,
                is_creative_commons=excluded.is_creative_commons,
                has_permission=excluded.has_permission,
                permission_notes=excluded.permission_notes,
                is_private_study=excluded.is_private_study,
                compliance_status=excluded.compliance_status,
                speech_density=excluded.speech_density,
                mode=excluded.mode,
                local_source_path=excluded.local_source_path,
                transcript_path=excluded.transcript_path,
                raw_metadata_json=excluded.raw_metadata_json,
                updated_at=excluded.updated_at
        """, (
            video_id, url, title, channel_id, channel_title, duration,
            license, int(is_creative_commons), int(has_permission), permission_notes,
            int(is_private_study), compliance_status, speech_density, mode,
            local_source_path, transcript_path,
            json.dumps(raw_metadata) if raw_metadata else None,
            datetime.now().isoformat()
        ))


def get_source(video_id: str) -> Optional[Dict[str, Any]]:
    """Get a source by video_id."""
    row = fetchone("SELECT * FROM sources WHERE video_id = ?", (video_id,))
    return dict(row) if row else None


def update_source(video_id: str, **kwargs) -> None:
    """Update source fields."""
    if not kwargs:
        return
    fields = ", ".join(f"{k}=?" for k in kwargs.keys())
    values = list(kwargs.values()) + [video_id]
    execute(f"UPDATE sources SET {fields}, updated_at=? WHERE video_id=?", values + [datetime.now().isoformat()])


# --- Compliance Logs ---

def log_compliance_check(
    video_id: str,
    check_type: str,
    passed: bool,
    reason: str,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log a compliance check result."""
    execute("""
        INSERT INTO compliance_logs (video_id, check_type, passed, reason, details_json)
        VALUES (?, ?, ?, ?, ?)
    """, (video_id, check_type, int(passed), reason, json.dumps(details) if details else None))


def get_compliance_logs(video_id: str) -> List[Dict[str, Any]]:
    """Get all compliance logs for a video."""
    rows = fetchall("SELECT * FROM compliance_logs WHERE video_id = ? ORDER BY created_at", (video_id,))
    return [dict(r) for r in rows]


# --- Clips CRUD ---

def create_clip(
    video_id: str,
    clip_index: int,
    start_time: float,
    end_time: float,
    title: str,
    hook_text: str,
    rationale: Optional[str] = None,
    viral_score: Optional[float] = None,
    visual_cues: Optional[List[str]] = None,
    narration_script: Optional[Dict[str, Any]] = None
) -> int:
    """Create a new clip record, return clip ID."""
    duration = end_time - start_time
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO clips (
                video_id, clip_index, start_time, end_time, duration,
                title, hook_text, rationale, viral_score,
                visual_cues_json, narration_script
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, clip_index, start_time, end_time, duration,
            title, hook_text, rationale, viral_score,
            json.dumps(visual_cues) if visual_cues else None,
            json.dumps(narration_script) if narration_script else None
        ))
        return cursor.lastrowid


def get_clips(video_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all clips for a video, optionally filtered by status."""
    if status:
        rows = fetchall(
            "SELECT * FROM clips WHERE video_id = ? AND status = ? ORDER BY clip_index",
            (video_id, status)
        )
    else:
        rows = fetchall(
            "SELECT * FROM clips WHERE video_id = ? ORDER BY clip_index",
            (video_id,)
        )
    return [dict(r) for r in rows]


def get_clip(clip_id: int) -> Optional[Dict[str, Any]]:
    """Get a clip by ID."""
    row = fetchone("SELECT * FROM clips WHERE id = ?", (clip_id,))
    return dict(row) if row else None


def update_clip(clip_id: int, **kwargs) -> None:
    """Update clip fields."""
    if not kwargs:
        return
    fields = ", ".join(f"{k}=?" for k in kwargs.keys())
    values = list(kwargs.values()) + [clip_id]
    execute(f"UPDATE clips SET {fields}, updated_at=? WHERE id=?", values + [datetime.now().isoformat()])


def approve_clip(clip_id: int, notes: Optional[str] = None) -> None:
    """Approve a clip for rendering/upload."""
    update_clip(clip_id, status="approved", approval_notes=notes, approved_at=datetime.now().isoformat())


def reject_clip(clip_id: int, notes: Optional[str] = None) -> None:
    """Reject a clip."""
    update_clip(clip_id, status="rejected", approval_notes=notes)


# --- Uploads CRUD ---

def create_upload(
    clip_id: int,
    title: str,
    description: str,
    tags: List[str],
    privacy_status: str,
    synthetic_media_disclaimer: bool = True
) -> int:
    """Create an upload record."""
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO uploads (
                clip_id, title, description, tags_json,
                privacy_status, synthetic_media_disclaimer,
                upload_status, quota_units_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            clip_id, title, description, json.dumps(tags),
            privacy_status, int(synthetic_media_disclaimer),
            "pending", get_settings().youtube.quota_per_upload
        ))
        return cursor.lastrowid


def get_upload(upload_id: int) -> Optional[Dict[str, Any]]:
    """Get upload by ID."""
    row = fetchone("SELECT * FROM uploads WHERE id = ?", (upload_id,))
    return dict(row) if row else None


def update_upload(upload_id: int, **kwargs) -> None:
    """Update upload fields."""
    if not kwargs:
        return
    fields = ", ".join(f"{k}=?" for k in kwargs.keys())
    values = list(kwargs.values()) + [upload_id]
    execute(f"UPDATE uploads SET {fields} WHERE id=?", values)


# --- Quota Tracking ---

def get_daily_quota_used() -> int:
    """Get quota used today."""
    today = datetime.now().strftime("%Y-%m-%d")
    row = fetchone("SELECT units_used FROM quota_tracker WHERE date_key = ?", (today,))
    return row["units_used"] if row else 0


def increment_quota(units: int) -> None:
    """Increment daily quota usage."""
    today = datetime.now().strftime("%Y-%m-%d")
    settings = get_settings()
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO quota_tracker (date_key, units_used, daily_limit, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date_key) DO UPDATE SET
                units_used = units_used + excluded.units_used,
                daily_limit = excluded.daily_limit,
                updated_at = excluded.updated_at
        """, (today, units, settings.youtube.quota_daily_limit, datetime.now().isoformat()))


def check_quota_available(units_needed: int = 1600) -> bool:
    """Check if enough quota remains for an upload."""
    settings = get_settings()
    used = get_daily_quota_used()
    return (used + units_needed) <= settings.youtube.quota_daily_limit