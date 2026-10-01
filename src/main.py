"""
ShortsForge CLI Entry Point

Typer-based command line interface for the ShortsForge pipeline.
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.config import get_settings, reload_settings
from src.db import init_db, get_source, get_clips, get_clip, approve_clip
from src.compliance import check_compliance, ComplianceRejectionError, is_private_study, can_upload
from src.download import download_video
from src.transcribe import transcribe_video
from src.analyze import analyze_video, get_clip_transcript
from src.narrate import generate_narration_script, synthesize_clip_narration
from src.clip import render_clip
from src.tts import synthesize_speech
from src.upload import upload_clip_to_youtube, YouTubeUploadError

app = typer.Typer(
    name="shortsforge",
    help="ShortsForge: Transform long videos into viral YouTube Shorts",
    add_completion=False
)

console = Console()


def setup_environment():
    """Load .env file if exists."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


@app.callback()
def callback():
    """ShortsForge: Transform long videos into viral YouTube Shorts"""
    setup_environment()
    init_db()


@app.command()
def analyze(
    url: str = typer.Argument(..., help="YouTube video URL"),
    has_permission: bool = typer.Option(False, "--has-permission", help="I have permission to use this content"),
    permission_notes: Optional[str] = typer.Option(None, "--permission-notes", help="Notes about permission"),
    private_study: bool = typer.Option(False, "--private-study", help="Process for private study only (no upload)"),
    force: bool = typer.Option(False, "--force", help="Force re-processing even if cached"),
):
    """
    Run full analysis pipeline: compliance -> download -> transcribe -> extract clips.
    Outputs clip suggestions as JSON.
    """
    rprint(Panel.fit("[bold cyan]ShortsForge Analysis Pipeline[/bold cyan]", border_style="cyan"))

    try:
        # Stage 1: Compliance
        rprint("[yellow]Stage 1:[/yellow] Checking compliance...")
        metadata = check_compliance(
            url,
            has_permission=has_permission,
            permission_notes=permission_notes,
            private_study=private_study
        )
        rprint(f"  ✓ Video: {metadata.title}")
        rprint(f"  ✓ Channel: {metadata.channel_title}")
        rprint(f"  ✓ Duration: {metadata.duration:.1f}s")
        rprint(f"  ✓ Compliance: PASSED")

        video_id = metadata.video_id

        # Stage 2: Download
        rprint("[yellow]Stage 2:[/yellow] Downloading video...")
        video_path = download_video(url, video_id)
        rprint(f"  ✓ Saved to: {video_path}")

        # Stage 3: Transcribe
        rprint("[yellow]Stage 3:[/yellow] Transcribing audio...")
        transcript = transcribe_video(video_id, video_path, force=force)
        rprint(f"  ✓ Language: {transcript.language}")
        rprint(f"  ✓ Duration: {transcript.duration:.1f}s")
        rprint(f"  ✓ Speech density: {transcript.speech_density:.1%}")

        # Stage 4: Analyze
        rprint("[yellow]Stage 4:[/yellow] Extracting clip candidates...")
        clips = analyze_video(video_id)

        # Display results
        if not clips:
            rprint("[red]No suitable clips found.[/red]")
            return

        table = Table(title=f"Clip Candidates for {metadata.title}")
        table.add_column("#", style="cyan", width=3)
        table.add_column("Title", style="white", max_width=40)
        table.add_column("Time", style="green", width=15)
        table.add_column("Duration", style="yellow", width=10)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("Hook", style="white", max_width=50)

        for clip in clips:
            duration = clip.end_time - clip.start_time
            time_str = f"{clip.start_time:.1f}-{clip.end_time:.1f}s"
            table.add_row(
                str(clip.clip_index),
                clip.title[:38] + ".." if len(clip.title) > 40 else clip.title,
                time_str,
                f"{duration:.1f}s",
                f"{clip.overall_score:.1f}/10",
                clip.hook_text[:48] + ".." if len(clip.hook_text) > 50 else clip.hook_text
            )

        console.print(table)

        # Output JSON for programmatic use
        output = {
            "video_id": video_id,
            "title": metadata.title,
            "clips": [
                {
                    "clip_index": c.clip_index,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "duration": c.end_time - c.start_time,
                    "title": c.title,
                    "hook_text": c.hook_text,
                    "hook_score": c.hook_score,
                    "clarity_score": c.clarity_score,
                    "payoff_score": c.payoff_score,
                    "overall_score": c.overall_score,
                    "rationale": c.rationale,
                    "self_contained": c.self_contained,
                    "needs_visual_check": c.needs_visual_check,
                    "visual_cues": c.visual_cues,
                    "hashtags": c.hashtags,
                    "description": c.description
                }
                for c in clips
            ]
        }
        rprint("\n[bold]JSON Output:[/bold]")
        rprint(json.dumps(output, indent=2))

    except ComplianceRejectionError as e:
        rprint(f"[red]Compliance failed:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        rprint(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command()
def narrate(
    video_id: str = typer.Argument(..., help="Video ID"),
    clip_id: int = typer.Argument(..., help="Clip ID from database"),
    style: Optional[str] = typer.Option(None, "--style", help="Narration tone style note")
):
    """Generate Claude Prompt C narration script and synthesize audio."""
    rprint(f"[yellow]Generating narration for clip {clip_id}...[/yellow]")
    script = generate_narration_script(video_id, clip_id, style_note=style)
    rprint(f"✓ Script generated ({script.total_words} words)")

    rprint("[yellow]Synthesizing audio with Voicebox / Edge-TTS...[/yellow]")
    wav_path = synthesize_clip_narration(video_id, clip_id, script=script)
    rprint(f"✓ Saved voiceover audio to: {wav_path}")


@app.command()
def render(
    video_id: str = typer.Argument(..., help="Video ID"),
    clip_id: int = typer.Argument(..., help="Clip ID from database")
):
    """Render vertical 1080x1920 MP4 Short with audio ducking and subtitles."""
    rprint(f"[yellow]Rendering vertical Short clip {clip_id}...[/yellow]")
    clip = get_clip(clip_id)
    if not clip:
        rprint(f"[red]Clip {clip_id} not found[/red]")
        sys.exit(1)

    narr_audio = None
    if clip.get("narration_audio_path") and Path(clip["narration_audio_path"]).exists():
        narr_audio = Path(clip["narration_audio_path"])

    out_mp4 = render_clip(video_id, clip_id, narration_audio_path=narr_audio)
    rprint(f"✓ [green]Successfully rendered:[/green] {out_mp4}")


@app.command()
def approve(
    clip_id: int = typer.Argument(..., help="Clip ID to approve")
):
    """Mark a clip as approved for publishing."""
    approve_clip(clip_id)
    rprint(f"✓ [green]Clip {clip_id} marked as APPROVED[/green]")


@app.command()
def upload(
    clip_id: int = typer.Argument(..., help="Clip ID from database"),
    privacy: Optional[str] = typer.Option(None, "--privacy", help="public, unlisted, or private")
):
    """Upload an approved clip to YouTube Shorts."""
    rprint(f"[yellow]Uploading clip {clip_id} to YouTube Data API v3...[/yellow]")
    try:
        res = upload_clip_to_youtube(clip_id, privacy_status=privacy)
        rprint(f"✓ [bold green]Upload Complete![/bold green] URL: {res['url']}")
    except YouTubeUploadError as e:
        rprint(f"[red]Upload Failed:[/red] {e}")
        sys.exit(1)


@app.command()
def transcribe(
    video_id: str = typer.Argument(..., help="Video ID to transcribe"),
    force: bool = typer.Option(False, "--force", help="Force re-transcription"),
):
    """Transcribe a previously downloaded video."""
    rprint(f"[yellow]Transcribing video {video_id}...[/yellow]")
    transcript = transcribe_video(video_id, force=force)
    rprint(f"✓ Transcription complete. Speech density: {transcript.speech_density:.1%}")


@app.command()
def clips(
    video_id: str = typer.Argument(..., help="Video ID"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
):
    """List clips for a video."""
    source = get_source(video_id)
    if not source:
        rprint(f"[red]Video {video_id} not found[/red]")
        return

    clips_list = get_clips(video_id, status)

    if not clips_list:
        rprint("[yellow]No clips found[/yellow]")
        return

    table = Table(title=f"Clips for {source['title']}")
    table.add_column("ID", style="cyan")
    table.add_column("Index", style="cyan")
    table.add_column("Title", style="white", max_width=40)
    table.add_column("Time", style="green")
    table.add_column("Duration", style="yellow")
    table.add_column("Status", style="magenta")
    table.add_column("Score", style="blue")

    for clip in clips_list:
        duration = clip['end_time'] - clip['start_time']
        table.add_row(
            str(clip['id']),
            str(clip['clip_index']),
            clip['title'][:38] + ".." if len(clip['title']) > 40 else clip['title'],
            f"{clip['start_time']:.1f}-{clip['end_time']:.1f}s",
            f"{duration:.1f}s",
            clip['status'],
            f"{clip['viral_score']:.1f}" if clip['viral_score'] else "N/A"
        )

    console.print(table)


@app.command()
def clip_transcript(
    video_id: str = typer.Argument(..., help="Video ID"),
    clip_id: int = typer.Argument(..., help="Clip ID"),
):
    """Get transcript for a specific clip."""
    clip = get_clips(video_id)
    clip_data = next((c for c in clip if c['id'] == clip_id), None)
    if not clip_data:
        rprint(f"[red]Clip {clip_id} not found[/red]")
        return

    transcript = get_clip_transcript(video_id, clip_data['start_time'], clip_data['end_time'])
    rprint(Panel(transcript, title=f"Transcript for Clip {clip_id}", border_style="green"))


@app.command()
def test_tts(
    text: str = typer.Argument(..., help="Text to synthesize"),
    voice: Optional[str] = typer.Option(None, "--voice", help="Voice ID / Profile"),
    speed: float = typer.Option(1.0, "--speed", help="Speech speed multiplier"),
    out: Optional[str] = typer.Option(None, "--out", help="Output WAV path")
):
    """Synthesize text using Voicebox or Edge-TTS fallback and save to WAV."""
    rprint(f"[yellow]Synthesizing speech...[/yellow]")
    rprint(f"Text: \"{text}\"")
    dest = Path(out) if out else Path("output/test_tts.wav")
    dest.parent.mkdir(parents=True, exist_ok=True)
    res_path = synthesize_speech(text=text, voice=voice, speed=speed, output_path=dest)
    rprint(f"✓ [green]Saved audio to:[/green] {res_path}")


@app.command()
def config_show():
    """Show current configuration."""
    settings = get_settings()
    import yaml
    rprint(Panel(yaml.dump(settings.model_dump(), sort_keys=False), title="Configuration", border_style="cyan"))


@app.command()
def db_init():
    """Initialize database schema."""
    init_db()
    rprint("[green]Database initialized[/green]")


if __name__ == "__main__":
    app()