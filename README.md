# Meeting Intelligence Agent

Turn pre-recorded meeting audio into structured knowledge: speaker-attributed transcripts, decisions, action items, summaries, and interactive Q&A.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Component Reference](#component-reference)
  - [Audio Loading](#1-audio-loading)
  - [Voice Activity Detection](#2-voice-activity-detection-vad)
  - [Speaker Diarization](#3-speaker-diarization)
  - [Speech-to-Text (ASR)](#4-speech-to-text-asr)
  - [Transcript Assembly](#5-transcript-assembly)
  - [Post-Processing](#6-post-processing)
  - [Topic Segmentation](#7-topic-segmentation)
  - [Decision & Action Extraction](#8-decision--action-item-extraction)
  - [Meeting Summarization](#9-meeting-summarization)
  - [Q&A](#10-qa-over-transcript)
  - [Text-to-Speech](#11-text-to-speech)
  - [Output Writers](#12-output-writers)
- [Data Models](#data-models)
- [Configuration](#configuration)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Testing](#testing)
- [Dependencies](#dependencies)

---

## Architecture Overview

The system processes audio through a 12-stage pipeline, split into two independent halves:

- **Audio Pipeline** (stages 1-6) — runs locally using ML models, no API calls needed
- **Understanding Pipeline** (stages 7-12) — uses OpenAI GPT-4 for NLP analysis

This separation means you can transcribe once and re-analyze many times without reprocessing audio.

```
Audio File (.wav, .mp3, .flac, .ogg, .m4a, .wma, .webm)
   │
   ├──[1] Audio Loader ─────────── pydub: decode + normalize to mono 16kHz float32
   ├──[2] VAD Processor ─────────── Silero VAD: detect speech vs silence
   ├──[3] Speaker Diarizer ──────── pyannote.audio: identify who spoke when
   ├──[4] ASR Engine ────────────── faster-whisper: speech → text with word timestamps
   ├──[5] Transcript Assembler ──── temporal overlap merge of ASR + diarization
   ├──[6] Transcript Cleaner ────── regex: disfluency removal, normalization
   │
   │   ── skip_llm boundary (--no-llm stops here) ──
   │
   ├──[7] Topic Segmenter ──────── GPT-4: identify topic boundaries
   ├──[8] Meeting Extractor ─────── GPT-4: extract decisions + action items
   ├──[9] Meeting Summarizer ────── GPT-4: structured summary (map-reduce for long meetings)
   ├──[10] Meeting Q&A ──────────── GPT-4: answer questions about the meeting
   ├──[11] Output Writers ──────── JSON + Markdown report generation
   └──[12] TTS Narrator ────────── edge-tts: optional audio summary playback
```

---

## Pipeline Stages

| Stage | Module | Library | What It Does |
|-------|--------|---------|-------------|
| 1. Load | `audio/loader.py` | pydub | Decodes any audio format → mono 16kHz float32 numpy array |
| 2. VAD | `audio/preprocessor.py` | Silero VAD | Detects speech regions, filters silence |
| 3. Diarize | `diarization/diarizer.py` | pyannote.audio | Labels speaker turns (who spoke when) |
| 4. Transcribe | `transcription/asr.py` | faster-whisper | Speech → text with word-level timestamps |
| 5. Assemble | `transcription/assembler.py` | — | Merges ASR text with speaker labels via temporal overlap |
| 6. Clean | `postprocessing/cleaner.py` | — | Removes "uh", "um", repeated words; normalizes text |
| 7. Segment | `postprocessing/segmenter.py` | GPT-4 | Breaks transcript into topic blocks |
| 8. Extract | `understanding/extractor.py` | GPT-4 | Finds decisions and action items |
| 9. Summarize | `understanding/summarizer.py` | GPT-4 | Generates structured meeting summary |
| 10. Q&A | `understanding/qa.py` | GPT-4 | Answers questions grounded in the transcript |
| 11. Output | `output/` | — | Writes JSON and Markdown reports |
| 12. TTS | `tts/speaker.py` | edge-tts | Narrates summary as MP3 audio |

---

## Project Structure

```
meeting_intelligence_agent/
├── pyproject.toml                          # Package metadata and dependencies
├── config.yaml                             # Default pipeline configuration
├── .env.example                            # Template for API keys
│
├── src/meeting_intelligence/
│   ├── __init__.py                         # Package version
│   ├── cli.py                              # Click CLI: process, ask, summarize
│   ├── pipeline.py                         # Pipeline orchestrator (lazy init, progress)
│   ├── config.py                           # YAML + env config loading
│   │
│   ├── models/                             # Pydantic data models (shared contracts)
│   │   ├── audio.py                        #   AudioData, AudioMetadata, SpeechSegment
│   │   ├── transcript.py                   #   Word, TranscriptSegment, Transcript, TopicBlock
│   │   └── meeting.py                      #   Decision, ActionItem, MeetingSummary, MeetingResult
│   │
│   ├── audio/                              # Audio ingestion
│   │   ├── loader.py                       #   Load + normalize any audio format
│   │   └── preprocessor.py                 #   Silero VAD speech detection
│   │
│   ├── diarization/                        # Speaker identification
│   │   └── diarizer.py                     #   pyannote.audio speaker diarization
│   │
│   ├── transcription/                      # Speech-to-text
│   │   ├── asr.py                          #   faster-whisper ASR engine
│   │   └── assembler.py                    #   Merge diarization + ASR (temporal overlap)
│   │
│   ├── postprocessing/                     # Text cleanup
│   │   ├── cleaner.py                      #   Disfluency removal, punctuation normalization
│   │   └── segmenter.py                    #   LLM-based topic segmentation
│   │
│   ├── understanding/                      # LLM-powered analysis
│   │   ├── llm_client.py                   #   OpenAI GPT-4 wrapper (structured output, chunking)
│   │   ├── extractor.py                    #   Decision + action item extraction
│   │   ├── summarizer.py                   #   Meeting summary generation
│   │   └── qa.py                           #   Q&A over transcript
│   │
│   ├── tts/                                # Audio output
│   │   └── speaker.py                      #   edge-tts summary narration
│   │
│   ├── output/                             # Report generation
│   │   ├── json_writer.py                  #   Full JSON output
│   │   └── markdown_writer.py              #   Markdown report with tables
│   │
│   └── utils/                              # Shared utilities
│       ├── logging.py                      #   Rich-powered structured logging
│       ├── timing.py                       #   Performance timing decorators
│       └── progress.py                     #   Pipeline stage progress callbacks
│
└── tests/
    ├── conftest.py                         # Shared fixtures (sample audio, transcripts)
    ├── fixtures/                            # Test audio files
    ├── test_assembler.py                   # Transcript assembly tests (6 tests)
    ├── test_cleaner.py                     # Text cleaning tests (6 tests)
    ├── test_config.py                      # Configuration loading tests (3 tests)
    └── test_output.py                      # JSON/Markdown writer tests (4 tests)
```

---

## Component Reference

### 1. Audio Loading

**File**: `src/meeting_intelligence/audio/loader.py`

Loads audio in any common format and normalizes it for downstream ML models.

| Function | Description |
|----------|-------------|
| `load_audio(file_path, target_sample_rate=16000)` | Main entry point. Returns `AudioData` (mono, 16kHz, float32) |
| `validate_audio_file(file_path)` | Checks file exists, non-empty, format supported |
| `get_audio_metadata(file_path)` | Extracts metadata without loading samples into memory |

**Supported formats**: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.wma`, `.webm`

**How it works**: Uses `pydub.AudioSegment.from_file()` to decode any format (requires `ffmpeg`), then converts to mono, resamples to 16kHz, and extracts samples as a numpy float32 array normalized to [-1, 1].

---

### 2. Voice Activity Detection (VAD)

**File**: `src/meeting_intelligence/audio/preprocessor.py`
**Class**: `VADProcessor`

Identifies which parts of the audio contain speech vs. silence.

| Method | Description |
|--------|-------------|
| `detect_speech(audio)` | Returns list of `SpeechSegment` (start/end in seconds) |
| `get_speech_ratio(segments, duration)` | Fraction of audio containing speech |

**How it works**: Loads [Silero VAD](https://github.com/snakers4/silero-vad) from torch.hub on first call (lazy loading). Converts numpy samples to a PyTorch tensor and calls `get_speech_timestamps()` with configurable threshold and duration parameters.

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.5 | Speech detection confidence threshold |
| `min_speech_duration_ms` | 250 | Ignore speech shorter than this |
| `min_silence_duration_ms` | 100 | Minimum silence gap to split segments |

---

### 3. Speaker Diarization

**File**: `src/meeting_intelligence/diarization/diarizer.py`
**Class**: `SpeakerDiarizer`

Determines *who* spoke *when* — assigns speaker labels (SPEAKER_00, SPEAKER_01, ...) to time ranges.

| Method | Description |
|--------|-------------|
| `diarize(audio, num_speakers, min_speakers, max_speakers)` | Returns list of `DiarizationSegment` |
| `get_speaker_stats(segments)` | Total speaking time per speaker |

**How it works**: Uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) `Pipeline.from_pretrained()`. Audio is passed directly as a waveform tensor (no temp file needed). Supports optional speaker count hints.

**Requirements**:
- A HuggingFace account with accepted terms for `pyannote/speaker-diarization-3.1`
- `HF_TOKEN` environment variable set

---

### 4. Speech-to-Text (ASR)

**File**: `src/meeting_intelligence/transcription/asr.py`
**Class**: `ASREngine`

Transcribes audio to text with word-level timestamps.

| Method | Description |
|--------|-------------|
| `transcribe(audio, language)` | Full audio transcription → `(segments, language_code)` |
| `transcribe_segment(audio, start, end)` | Transcribe a specific time slice |

**How it works**: Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2-optimized OpenAI Whisper). Produces text segments with per-word timing and confidence scores. Auto-detects language if not specified.

**Configuration**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_size` | `large-v2` | Whisper model size (tiny/base/small/medium/large-v2/large-v3) |
| `compute_type` | `int8` | Quantization type. Use `float16` for GPU |
| `beam_size` | 5 | Beam search width (higher = more accurate, slower) |
| `word_timestamps` | true | Enable word-level timing (needed for accurate speaker assignment) |

---

### 5. Transcript Assembly

**File**: `src/meeting_intelligence/transcription/assembler.py`
**Class**: `TranscriptAssembler`

The core algorithm: merges "what was said" (ASR) with "who said it" (diarization) into a unified, speaker-attributed transcript.

| Method | Description |
|--------|-------------|
| `assemble(asr_segments, diarization_segments, ...)` | Main merge → `Transcript` |

**Algorithm**:
1. For each ASR segment, if word-level timestamps exist, assign each *word* to the speaker with maximum temporal overlap, then group consecutive words by speaker — this correctly handles mid-sentence speaker changes.
2. If no word timestamps, assign the entire segment to the speaker with the greatest overlap.
3. Merge consecutive segments from the same speaker into single turns.

This two-pass approach (full-audio ASR first, then merge with diarization) is the industry standard pattern used by [WhisperX](https://github.com/m-bain/whisperX).

---

### 6. Post-Processing

**File**: `src/meeting_intelligence/postprocessing/cleaner.py`
**Class**: `TranscriptCleaner`

Cleans up raw ASR output for readability.

| Method | Description |
|--------|-------------|
| `clean(transcript)` | Apply all cleaning steps, return new `Transcript` |
| `remove_disfluencies(text)` | Remove "uh", "um", "er", "like", "you know", etc. |
| `remove_repeated_words(text)` | "the the" → "the" |
| `normalize_whitespace(text)` | Collapse multiple spaces |
| `capitalize_sentences(text)` | Capitalize after sentence-ending punctuation |

---

### 7. Topic Segmentation

**File**: `src/meeting_intelligence/postprocessing/segmenter.py`
**Class**: `TopicSegmenter`

Groups transcript segments into coherent topic blocks.

| Method | Description |
|--------|-------------|
| `segment(transcript)` | Returns list of `TopicBlock` |

**How it works**: Sends the numbered transcript to GPT-4 and asks it to identify topic boundaries with names and summaries. Falls back to time-based segmentation (5-minute intervals) if the LLM is unavailable.

---

### 8. Decision & Action Item Extraction

**File**: `src/meeting_intelligence/understanding/extractor.py`
**Class**: `MeetingExtractor`

Uses GPT-4 to identify decisions and tasks from the transcript.

| Method | Returns |
|--------|---------|
| `extract_decisions(transcript)` | List of `Decision` — agreements, commitments, resolutions |
| `extract_action_items(transcript)` | List of `ActionItem` — tasks with assignee, deadline, priority |

**Examples**:
- *"We'll deploy Friday"* → **Decision**: Deploy on Friday
- *"John, send the report by EOD"* → **ActionItem**: task=send report, assignee=John, deadline=EOD, priority=high

---

### 9. Meeting Summarization

**File**: `src/meeting_intelligence/understanding/summarizer.py`
**Class**: `MeetingSummarizer`

Generates a structured meeting summary.

| Method | Description |
|--------|-------------|
| `summarize(transcript, decisions, action_items)` | Returns `MeetingSummary` |

**Strategies**:
- **Single-pass**: For transcripts under 6000 tokens — one GPT-4 call
- **Map-reduce**: For longer meetings — summarize each chunk independently, then combine chunk summaries into a final summary

**Output fields**: title, executive summary, key points (3-7), topics discussed, participant count, duration.

---

### 10. Q&A Over Transcript

**File**: `src/meeting_intelligence/understanding/qa.py`
**Class**: `MeetingQA`

Ask natural language questions about the meeting and get answers grounded in the transcript.

| Method | Description |
|--------|-------------|
| `load_transcript(transcript)` | Set context for subsequent questions |
| `ask(question, transcript?)` | Returns `QAResponse` with answer, citations, confidence |

**Example**: *"What did we agree about deployment?"* → *"The team agreed to deploy on Friday, as stated by SPEAKER_01 at 4.9s."*

---

### 11. Text-to-Speech

**File**: `src/meeting_intelligence/tts/speaker.py`
**Class**: `SummaryNarrator`

Converts the meeting summary to spoken audio using Microsoft's neural TTS voices.

| Method | Description |
|--------|-------------|
| `narrate(summary, output_path)` | Async — generates MP3 audio of the summary |
| `narrate_sync(summary, output_path)` | Sync wrapper for CLI use |

**Default voice**: `en-US-AndrewNeural`
**Output format**: MP3
**Requires**: Network access (calls Microsoft Edge TTS service)

---

### 12. Output Writers

**Files**: `src/meeting_intelligence/output/json_writer.py`, `markdown_writer.py`

| Class | Output | Filename |
|-------|--------|----------|
| `JSONWriter` | Full structured data (parseable) | `meeting_YYYYMMDD_HHMMSS.json` |
| `MarkdownWriter` | Human-readable report with tables | `meeting_YYYYMMDD_HHMMSS.md` |

**Markdown report sections**: Title & metadata, Executive Summary, Key Points, Topics Discussed, Decisions (table), Action Items (table), Full Transcript with timestamps.

---

## Data Models

All models live in `src/meeting_intelligence/models/` and use Pydantic v2.

### Audio Models (`models/audio.py`)

```
AudioMetadata    — file_path, duration_seconds, sample_rate, channels, format, file_size_bytes
AudioData        — samples (float32 array), sample_rate, duration_seconds, metadata
SpeechSegment    — start, end, confidence  (+.duration property)
```

### Transcript Models (`models/transcript.py`)

```
Word               — text, start, end, confidence
DiarizationSegment — speaker, start, end
TranscriptSegment  — speaker, start, end, text, words[], confidence
TopicBlock         — topic, summary, segments[], start, end
Transcript         — segments[], speakers[], language, duration_seconds, topic_blocks[]
                     (+.full_text, .speaker_text properties)
```

### Meeting Models (`models/meeting.py`)

```
Decision       — id, text, context, timestamp, participants[], confidence
ActionItem     — id, task, assignee, deadline, priority (high/medium/low), status, confidence
MeetingSummary — title, executive_summary, key_points[], topics_discussed[],
                 participant_count, duration_minutes
QAResponse     — question, answer, relevant_segments[], confidence
MeetingResult  — audio_file, processed_at, transcript, summary, decisions[],
                 action_items[], qa_history[]
```

---

## Configuration

### config.yaml

```yaml
pipeline:
  device: "cpu"           # "cpu" or "cuda"
  language: "en"          # null for auto-detect

audio:
  target_sample_rate: 16000
  target_channels: 1

vad:
  threshold: 0.5
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100

diarization:
  model: "pyannote/speaker-diarization-3.1"
  min_speakers: null      # set to constrain speaker count
  max_speakers: null

asr:
  model_size: "large-v2"  # tiny/base/small/medium/large-v2/large-v3
  compute_type: "int8"    # "float16" for GPU
  beam_size: 5
  word_timestamps: true

llm:
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 4096

tts:
  enabled: false
  voice: "en-US-AndrewNeural"
  rate: "+0%"

output:
  directory: "./output"
  formats:
    - json
    - markdown
```

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

| Variable | Required For | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | LLM features (stages 7-10) | OpenAI API key for GPT-4 |
| `HF_TOKEN` | Speaker diarization (stage 3) | HuggingFace token with access to pyannote models |

The config supports `${VAR_NAME}` substitution anywhere in YAML values, so you can reference env vars directly in `config.yaml` if preferred.

---

## Setup & Installation

### Prerequisites

1. **Python 3.10+**

2. **ffmpeg** (required for audio format decoding):
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   sudo apt install ffmpeg

   # Conda
   conda install ffmpeg
   ```

3. **HuggingFace access** for pyannote diarization:
   - Create an account at https://huggingface.co
   - Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Generate an access token at https://huggingface.co/settings/tokens

4. **OpenAI API key** (for LLM features):
   - Get one at https://platform.openai.com/api-keys

### Installation

```bash
# Clone the project
cd meeting_intelligence_agent

# Install in development mode
pip install -e ".[dev]"

# Set up API keys
cp .env.example .env
# Edit .env with your keys:
#   OPENAI_API_KEY=sk-...
#   HF_TOKEN=hf_...
```

### First Run Note

The first time you run the pipeline, it will automatically download:
- **Silero VAD model** (~2 MB)
- **Whisper large-v2** (~3 GB) — use `model_size: "small"` in config for faster download (~500 MB)
- **pyannote diarization model** (~100 MB)

These are cached locally and reused on subsequent runs.

---

## Usage

### Process a Meeting Recording

```bash
# Full pipeline: transcript + GPT-4 analysis
meeting-intel process recording.mp3

# Specify output directory and expected speaker count
meeting-intel process recording.mp3 -o ./results -s 4

# Auto-detect language (omit -l flag)
meeting-intel process recording.mp3

# Specify language explicitly
meeting-intel process recording.mp3 -l fr
```

### Transcript Only (No API Cost)

```bash
# Skip GPT-4 analysis — only runs audio pipeline (stages 1-6)
meeting-intel process recording.mp3 --no-llm
```

### Ask Questions About a Meeting

```bash
# After processing, ask questions about the meeting
meeting-intel ask output/meeting_20260218_143000.json -q "What did we agree about deployment?"
meeting-intel ask output/meeting_20260218_143000.json -q "Who is responsible for the report?"
```

### Re-Generate Summary

```bash
# Re-run LLM analysis on an existing transcript (no audio reprocessing)
meeting-intel summarize output/meeting_20260218_143000.json
```

### Ask Questions During Processing

```bash
# Process and ask questions in one command
meeting-intel process recording.mp3 \
  -q "What were the main disagreements?" \
  -q "What is the timeline for the project?"
```

### Generate Audio Summary

```bash
# Generate an MP3 narration of the meeting summary
meeting-intel process recording.mp3 --tts
```

### Verbose Mode

```bash
# Enable debug logging to see all pipeline details
meeting-intel -v process recording.mp3
```

### Custom Config

```bash
# Use a custom configuration file
meeting-intel --config my_config.yaml process recording.mp3
```

### Example Output

After processing, the `output/` directory will contain:

```
output/
├── meeting_20260218_143000.json     # Full structured data (machine-readable)
├── meeting_20260218_143000.md       # Human-readable report
└── summary_20260218_143000.mp3      # Audio summary (if --tts used)
```

The **Markdown report** includes:
- Meeting title, date, duration, speaker count
- Executive summary
- Key points (bulleted list)
- Topics discussed
- Decisions table (decision, participants)
- Action items table (task, assignee, deadline, priority)
- Full transcript with timestamps and speaker labels

---

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_assembler.py -v

# Run with coverage
pytest --cov=meeting_intelligence

# Skip integration tests (those requiring models/API keys)
pytest -m "not integration"
```

**Test coverage** (19 tests):
| Test File | Tests | What It Covers |
|-----------|-------|---------------|
| `test_assembler.py` | 6 | Temporal overlap merge, word-level splitting, edge cases |
| `test_cleaner.py` | 6 | Disfluency removal, repeated words, normalization |
| `test_config.py` | 3 | Default config, YAML loading, env var override |
| `test_output.py` | 4 | JSON and Markdown output structure and content |

---

## Dependencies

### Runtime

| Package | Version | Purpose |
|---------|---------|---------|
| `pydub` | >= 0.25.1 | Audio format loading and conversion |
| `numpy` | >= 1.24 | Audio sample array operations |
| `silero-vad` | >= 5.1 | Voice Activity Detection model |
| `torch` | >= 2.0 | PyTorch backend for ML models |
| `torchaudio` | >= 2.0 | Audio utilities for PyTorch |
| `pyannote.audio` | >= 3.1 | Speaker diarization pipeline |
| `faster-whisper` | >= 1.0 | CTranslate2-optimized Whisper ASR |
| `openai` | >= 1.40 | GPT-4 API client |
| `tiktoken` | >= 0.7 | Token counting for context management |
| `pydantic` | >= 2.8 | Data model validation and serialization |
| `pydantic-settings` | >= 2.0 | Settings management |
| `pyyaml` | >= 6.0 | YAML configuration loading |
| `python-dotenv` | >= 1.0 | `.env` file loading |
| `click` | >= 8.1 | CLI framework |
| `rich` | >= 13.0 | Terminal formatting and progress display |
| `edge-tts` | >= 6.1 | Microsoft Neural text-to-speech |

### System

| Dependency | Installation |
|-----------|-------------|
| `ffmpeg` | `brew install ffmpeg` (macOS) / `apt install ffmpeg` (Linux) |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-mock` | Mocking utilities |
| `ruff` | Linting and formatting |
| `mypy` | Static type checking |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Two-pass ASR + diarization merge** | Full-audio ASR first, then merge with diarization. Better accuracy than transcribing each speaker segment independently, since cross-segment context improves Whisper output. Industry standard (WhisperX). |
| **Word-level speaker assignment** | When word timestamps are available, each word is assigned to its overlapping speaker independently. This correctly handles mid-sentence speaker changes and interruptions. |
| **Lazy model loading** | Whisper (~3 GB), pyannote, and Silero VAD models load only when first needed. This keeps `import` fast and means `meeting-intel ask` never loads audio models. |
| **Structured output with fallback** | Tries OpenAI's native structured output API first (`gpt-4o`+), falls back to JSON mode with Pydantic schema injection for `gpt-4`. Works with any OpenAI model. |
| **Map-reduce summarization** | Long meetings exceeding the token limit are chunked, summarized independently (map), then combined into a final summary (reduce). |
| **Separate audio/LLM pipelines** | `--no-llm` runs only stages 1-6 (free, no API cost). `summarize` and `ask` commands re-run only the LLM stages on saved transcripts. |
