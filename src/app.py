"""
ShortsForge Streamlit Review Portal

Stage 8: Interactive Human-in-the-Loop Review UI.
Review clips, edit metadata and scripts, re-render, approve/reject, and trigger YouTube upload.
"""

import os
import json
from pathlib import Path
import streamlit as st

from src.config import get_settings
from src.db import (
    init_db,
    get_source,
    get_clips,
    get_clip,
    update_clip,
    approve_clip,
    reject_clip,
    fetchall,
    get_daily_quota_used,
    check_quota_available
)
from src.compliance import check_compliance, is_private_study, can_upload
from src.download import download_video
from src.transcribe import transcribe_video
from src.analyze import analyze_video
from src.narrate import generate_narration_script, synthesize_clip_narration
from src.clip import render_clip


def init_page():
    st.set_page_config(
        page_title="ShortsForge | Studio Review",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    init_db()


def render_sidebar():
    st.sidebar.title("🎬 ShortsForge")
    st.sidebar.markdown("---")

    # Daily Quota Widget
    settings = get_settings()
    used_quota = get_daily_quota_used()
    max_quota = settings.youtube.quota_daily_limit
    pct = min(1.0, used_quota / max_quota)

    st.sidebar.metric(
        label="YouTube Quota Today",
        value=f"{used_quota} / {max_quota}",
        delta=f"{int(max_quota - used_quota)} remaining"
    )
    st.sidebar.progress(pct)

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Active Configuration")
    st.sidebar.text(f"LLM: {settings.llm_model}")
    st.sidebar.text(f"Whisper: {settings.whisper_model}")
    st.sidebar.text(f"TTS Provider: {settings.tts.provider}")
    st.sidebar.text(f"Default Privacy: {settings.privacy_default}")


def render_ingestion_tab():
    st.header("📥 Ingest New Video")
    url = st.text_input("Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=...")

    col1, col2 = st.columns(2)
    with col1:
        has_perm = st.checkbox("I own this content / I have creator permission")
        perm_note = st.text_input("Permission Note / Proof Reference (if ticked):", placeholder="Agreement #123")
    with col2:
        private_mode = st.checkbox("🔒 Private Study Mode (Local evaluation only, permanently disables uploads)")

    if st.button("🚀 Process & Generate Clips", type="primary"):
        if not url:
            st.error("Please enter a valid YouTube URL.")
            return

        with st.status("Processing video pipeline...", expanded=True) as status:
            try:
                # Stage 1: Compliance
                st.write("Checking copyright & compliance rules...")
                metadata = check_compliance(
                    url=url,
                    has_permission=has_perm,
                    permission_notes=perm_note if has_perm else None,
                    private_study=private_mode
                )
                st.write(f"✓ Video: **{metadata.title}** ({metadata.channel_title})")

                # Stage 2: Download
                st.write("Ingesting source video...")
                video_path = download_video(url, metadata.video_id)
                st.write(f"✓ Downloaded to `{video_path}`")

                # Stage 3: Transcribe
                st.write("Transcribing audio with faster-whisper...")
                transcript = transcribe_video(metadata.video_id, video_path)
                st.write(f"✓ Speech Density: **{transcript.speech_density:.1%}**")

                # Stage 4: Analyze
                st.write("Analyzing and extracting viral moment candidates with Claude...")
                clips = analyze_video(metadata.video_id)
                st.write(f"✓ Extracted {len(clips)} candidates!")

                status.update(label="Ingestion complete! Head over to the Review Tab.", state="complete")
                st.success(f"Generated {len(clips)} clips for review!")
                st.rerun()

            except Exception as e:
                status.update(label=f"Pipeline error: {e}", state="error")
                st.error(str(e))


def render_review_tab():
    st.header("👁️ Clip Review & Approval Dashboard")

    # Fetch all processed sources
    sources_rows = fetchall("SELECT * FROM sources ORDER BY created_at DESC")
    if not sources_rows:
        st.info("No videos ingested yet. Start in the Ingest tab!")
        return

    source_options = {f"{s['title']} ({s['video_id']})": s['video_id'] for s in sources_rows}
    selected_label = st.selectbox("Select Video Source:", list(source_options.keys()))
    video_id = source_options[selected_label]
    source = get_source(video_id)

    # Compliance Status Pill
    if source["is_private_study"] == 1:
        st.warning("🔒 **Private Study Mode Enabled**: YouTube upload is disabled for clips from this source.")
    else:
        st.success(f"✅ **Licensed Source**: {source.get('license') or 'Approved via channel/manual permission'}")

    # Fetch Clips for this video
    clips = get_clips(video_id)
    if not clips:
        st.info("No clips extracted for this video.")
        return

    st.markdown("---")

    for clip in clips:
        with st.expander(f"Clip #{clip['clip_index']}: {clip['title']} [{clip['status'].upper()}] - {clip['duration']:.1f}s", expanded=(clip['status'] == 'pending_review')):
            c1, c2 = st.columns([1, 1])

            with c1:
                st.subheader("🎥 Video Preview")
                video_file = clip.get("output_video_path")
                if video_file and Path(video_file).exists():
                    st.video(str(video_file))
                else:
                    st.info("Clip has not been rendered yet. Click 'Render Clip' below.")

                st.markdown(f"**Timing**: `{clip['start_time']:.1f}s` to `{clip['end_time']:.1f}s` (`{clip['duration']:.1f}s`)")
                st.markdown(f"**Viral Score**: `{clip.get('viral_score', 'N/A')}/10`")
                st.markdown(f"**Hook**: *\"{clip.get('hook_text', '')}\"*")
                st.markdown(f"**Rationale**: {clip.get('rationale', '')}")

            with c2:
                st.subheader("📝 Metadata & Narration Script")

                # Editable fields
                new_title = st.text_input("Title (Max 70 chars):", value=clip["title"], key=f"t_{clip['id']}")
                new_hook = st.text_input("Hook:", value=clip.get("hook_text", ""), key=f"h_{clip['id']}")

                # Narration Script
                script_val = ""
                if clip.get("narration_script"):
                    if isinstance(clip["narration_script"], str):
                        try:
                            parsed_s = json.loads(clip["narration_script"])
                            script_val = " ".join(seg.get("text", "") for seg in parsed_s.get("segments", []))
                        except Exception:
                            script_val = clip["narration_script"]
                    elif isinstance(clip["narration_script"], dict):
                        script_val = " ".join(seg.get("text", "") for seg in clip["narration_script"].get("segments", []))

                edited_script = st.text_area("Narration Voiceover Script:", value=script_val, height=120, key=f"s_{clip['id']}")

                # Action Button Row
                btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

                with btn_c1:
                    if st.button("💾 Save", key=f"save_{clip['id']}"):
                        update_clip(clip["id"], title=new_title, hook_text=new_hook)
                        st.success("Saved metadata!")

                with btn_c2:
                    if st.button("🎙️ Gen Narration", key=f"gn_{clip['id']}"):
                        with st.spinner("Generating narration script with Claude..."):
                            generate_narration_script(video_id, clip["id"])
                            st.success("Generated script!")
                            st.rerun()

                with btn_c3:
                    if st.button("🎬 Render", key=f"rnd_{clip['id']}", type="secondary"):
                        with st.spinner("Rendering vertical 9:16 short with ffmpeg..."):
                            # Synthesize TTS audio if script exists
                            narr_audio = None
                            if clip.get("narration_script"):
                                narr_audio = synthesize_clip_narration(video_id, clip["id"])

                            out_path = render_clip(video_id, clip["id"], narration_audio_path=narr_audio)
                            st.success(f"Rendered: {out_path.name}")
                            st.rerun()

                with btn_c4:
                    if clip["status"] != "approved":
                        if st.button("✅ Approve", key=f"app_{clip['id']}", type="primary"):
                            approve_clip(clip["id"])
                            st.success("Approved!")
                            st.rerun()
                    else:
                        if st.button("❌ Reject", key=f"rej_{clip['id']}"):
                            reject_clip(clip["id"])
                            st.warning("Rejected")
                            st.rerun()

                # Upload Section
                if clip["status"] == "approved":
                    st.markdown("---")
                    st.markdown("### 📤 YouTube Upload")
                    if source["is_private_study"] == 1:
                        st.error("Upload disabled: This video is flagged for Private Study only.")
                    else:
                        if st.button("🚀 Publish to YouTube", key=f"up_{clip['id']}", type="primary"):
                            from src.upload import upload_clip_to_youtube
                            with st.spinner("Uploading to YouTube via Data API v3..."):
                                try:
                                    res = upload_clip_to_youtube(clip["id"])
                                    st.success(f"Uploaded successfully! Video ID: {res['youtube_video_id']}")
                                except Exception as e:
                                    st.error(f"Upload failed: {e}")


def main():
    init_page()
    render_sidebar()

    tab1, tab2 = st.tabs(["👁️ Review Studio", "📥 Ingest Source"])
    with tab1:
        render_review_tab()
    with tab2:
        render_ingestion_tab()


if __name__ == "__main__":
    main()