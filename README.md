# 🎬 ShortsForge

**ShortsForge** is a personal, automated AI pipeline and human-in-the-loop studio that converts long-form YouTube videos into high-retention, vertically formatted (9:16, 1080x1920) YouTube Shorts with automated moment mining, AI narration, local Voicebox TTS synthesis, dynamic karaoke captions, and compliance-checked YouTube Data API v3 uploading.

---

## ⚡ Workflow in One Line

```
YouTube URL ──► Compliance Guard ──► yt-dlp Download ──► faster-whisper (Word Timestamps)
                  │
                  ▼
   [Speech Density >= 0.35: Transcript Mode (Prompt A)]
   [Speech Density < 0.35:  Vision Mode (Scenedetect + Librosa + Prompt B -> Prompt A)]
                  │
                  ▼
   Claude Narration Script (Prompt C: duration * 2.5 * 0.9)
                  │
                  ▼
   Local Voicebox TTS Synthesis (with automatic Edge-TTS fallback)
                  │
                  ▼
   FFmpeg 9:16 Vertical Crop (Blur Background) + Audio Ducking (25%) + ASS Captions
                  │
                  ▼
   Streamlit Review UI (Preview, Edit, Regenerate, Approve)
                  │
                  ▼ (On Explicit Approval & Not Private Study)
   YouTube Data API v3 Resumable Upload
```

---

## 🛠️ Prerequisites & Setup

### 1. System Dependencies
- **Python 3.11+**
- **FFmpeg**: Ensure `ffmpeg` and `ffprobe` are installed and available in your system `PATH`.
  - Windows (via Scoop/Winget): `winget install Gyan.FFmpeg` or `scoop install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`

### 2. Python Environment & Installation
```bash
git clone https://github.com/your-username/ShortsForge.git
cd ShortsForge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Voicebox Local TTS Setup (Optional but Recommended)
ShortsForge natively integrates with local Voicebox REST API for zero-latency, private, studio-quality voiceover generation:
1. Launch Voicebox on `http://127.0.0.1:17493` (or the URL set in `config.yaml`).
2. Download a voice profile / engine in Voicebox and note the `profile_id` (e.g. `default_profile`).
3. If Voicebox is not running or unreachable, ShortsForge **automatically falls back to Edge-TTS** (`en-US-JennyNeural`).

### 4. Google Cloud & YouTube Data API v3 Setup
To enable YouTube Shorts uploads:
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project and enable **YouTube Data API v3**.
3. Configure OAuth Consent Screen (set User Type to External, add yourself as a Test User).
4. Create Credentials -> **OAuth client ID** -> Application type: **Desktop app**.
5. Download the credentials JSON, rename it to `client_secrets.json`, and place it in the `data/` directory:
   ```
   data/client_secrets.json
   ```
6. Alternatively, set `YOUTUBE_CLIENT_ID` and `YOUTUBE_CLIENT_SECRET` in your `.env` file.

### 5. Configuration & Environment Variables
Copy `.env.example` to `.env` and fill in your Anthropic API key:
```bash
cp .env.example .env
```

Edit `.env`:
```env
ANTHROPIC_API_KEY=sk-ant-api03-...
```

Review and adjust pipeline settings in `config.yaml`:
```yaml
clip_min_len: 9
clip_max_len: 60
top_n: 5
whisper_model: "small"
frame_fps: 1
speech_density_threshold: 0.35
privacy_default: "private"
llm_model: "claude-sonnet-5-5"

allowlist_channels:
  - "MyOwnChannel"
  - "ApprovedPartnerChannel"

tts:
  provider: "voicebox"
  voicebox:
    base_url: "http://127.0.0.1:17493"
    profile_id: "default_profile"
    language: "en"
  edge_tts:
    voice: "en-US-JennyNeural"
```

---

## 🚀 Running ShortsForge

### A. Streamlit Review UI (Recommended)
Launch the interactive web portal:
```bash
streamlit run src/app.py
```
- **Ingest Tab**: Paste any YouTube URL, specify permission/compliance flags (or enable Private Study Mode), and trigger automatic transcription and moment mining.
- **Review Tab**: Watch rendered 9:16 vertical video previews, edit titles, modify narration scripts, regenerate voiceovers with Voicebox/Edge-TTS, and approve clips.
- **Upload**: One-click publish to YouTube Shorts with automatic `#Shorts` tagging and AI synthetic media disclosure.

### B. CLI Usage (Typer)

#### 1. Ingest & Analyze Video (Stages 1-4)
```bash
# Standard licensed / allowlisted video
python -m src.main analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# With manual permission note
python -m src.main analyze "https://www.youtube.com/watch?v=VIDEO_ID" --has-permission --permission-notes "Agreement #123"

# Private study mode (local only, disables YouTube upload)
python -m src.main analyze "https://www.youtube.com/watch?v=VIDEO_ID" --private-study
```

#### 2. List Extracted Clips for a Video
```bash
python -m src.main clips VIDEO_ID
```

#### 3. Test TTS Voice Synthesis
```bash
python -m src.main test-tts "ShortsForge is converting this video into a vertical short."
```

#### 4. View Configuration & Database
```bash
python -m src.main config-show
python -m src.main db-init
```

---

## 🧪 Running Unit Tests

Run the complete test suite with `pytest`:
```bash
pytest -v
```

Test coverage includes:
- `tests/test_compliance.py`: Allowlist, CC licenses, manual overrides, and Private Study gating.
- `tests/test_transcribe.py`: Word timestamps, speech density metric, and boundary snapping.
- `tests/test_analyze.py`: Claude Prompt A parsing, Pydantic validation, and non-max overlap removal.
- `tests/test_tts.py`: Voicebox REST client, polling loop, and Edge-TTS fallback.
- `tests/test_clip.py`: ASS subtitle formatting and FFmpeg 1080x1920 filtergraph generation.
- `tests/test_narrate.py`: Word budget formulas (`duration * 2.5 * 0.9`) and Prompt C.
- `tests/test_upload.py`: YouTube OAuth credentials, quota tracking (1600 units/upload), and Private Study protection.

---

## 📂 Project Structure

```
ShortsForge/
├── config.yaml               # Settings (clip length, TTS, captions, video format)
├── .env.example              # Template for API keys & secrets
├── requirements.txt          # Pinned Python package dependencies
├── README.md                 # Complete documentation
├── data/
│   ├── shortsforge.db        # SQLite database
│   ├── client_secrets.json   # Google OAuth client secrets
│   └── youtube_token.json    # Cached Google OAuth token
├── work/                     # Working cache directory for downloaded videos & transcripts
├── output/                   # Rendered vertical 9:16 Shorts MP4 files
├── src/
│   ├── __init__.py
│   ├── config.py             # Pydantic BaseSettings & YAML loader
│   ├── db.py                 # SQLite database models & queries
│   ├── compliance.py         # Stage 1: Copyright & compliance verification
│   ├── download.py           # Stage 2: yt-dlp downloader with caching
│   ├── transcribe.py         # Stage 3: faster-whisper transcription & speech density
│   ├── schemas.py            # Pydantic models & LLM prompt definitions
│   ├── analyze.py            # Stage 4: Claude Prompt A clip candidate extraction
│   ├── vision.py             # Stage 4B: Scene cuts (scenedetect) & audio energy (librosa)
│   ├── narrate.py            # Stage 5: Claude Prompt C voiceover scripting
│   ├── tts.py                # Stage 6: Voicebox REST API provider & Edge-TTS fallback
│   ├── clip.py               # Stage 7: FFmpeg vertical 9:16 layout, audio ducking, ASS captions
│   ├── upload.py             # Stage 9: YouTube Data API v3 resumable upload
│   ├── app.py                # Stage 8: Streamlit Review Portal
│   ├── main.py               # Typer CLI application
│   └── utils/
│       ├── __init__.py
│       └── subtitles.py      # Styled ASS subtitle generator
└── tests/
    ├── test_compliance.py
    ├── test_transcribe.py
    ├── test_analyze.py
    ├── test_tts.py
    ├── test_clip.py
    ├── test_narrate.py
    └── test_upload.py
```

---

## ⚖️ Compliance & Ethical AI Principles

1. **Permission First**: Ingestion is gated by default. Non-licensed videos require manual permission notes or must use **Private Study Mode**, which permanently prevents accidental uploads.
2. **Attribution & Transparency**: Generated descriptions credit the original source video and creator.
3. **Synthetic Media Disclosure**: Automatically flags AI voiceover narration in YouTube metadata for compliance with platform guidelines.
4. **Voicebox Safety**: Uses stock presets or Voice Design voices; never clones unauthorized third-party voices.
5. **Human in the Loop**: Videos are never published without explicit review and approval in the Streamlit studio.