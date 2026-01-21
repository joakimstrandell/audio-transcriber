# Audio Transcriber - Project Context

## Overview

A macOS CLI tool that captures system audio using ScreenCaptureKit and transcribes it locally using OpenAI's Whisper model. No external services required - everything runs locally.

## Quick Reference

```bash
# Activate virtual environment
source env/bin/activate

# Run commands
transcriber record              # Record system audio, Ctrl+C to stop
transcriber record --mic        # Also capture microphone
transcriber record -m small     # Use 'small' Whisper model
transcriber stream              # Real-time streaming transcription
transcriber stream -c 3.0       # Stream with 3-second chunks (default: 5.0)
transcriber transcribe <file>   # Transcribe existing audio file
transcriber list                # List saved transcriptions
transcriber show <id>           # View specific transcription
transcriber check               # Verify permissions
transcriber path                # Show storage directories
```

## Project Structure

```
audio_transcriber/
├── __init__.py
├── cli.py           # Click CLI commands (record, transcribe, list, show, check, path)
├── recorder.py      # AudioRecorder - ScreenCaptureKit audio capture
├── streaming.py     # StreamingTranscriber - Real-time transcription with faster-whisper (WIP)
└── transcriber.py   # Transcriber - Batch transcription with openai-whisper
```

## Key Components

### AudioRecorder (recorder.py)
- Uses ScreenCaptureKit to capture system audio (macOS 13+)
- Creates minimal video capture (2x2px @ 1fps) to access audio stream
- Outputs 16kHz mono WAV files (optimal for Whisper)
- Optional microphone capture with `--mic` flag
- Uses NSRunLoop pumping for async Objective-C operations

### Transcriber (transcriber.py)
- Uses openai-whisper for batch transcription
- Lazy model loading (downloads on first use)
- Loads WAV via scipy (no ffmpeg dependency)
- Stores results in `~/.audio-transcriber/transcriptions/`

### StreamingTranscriber (streaming.py)
- Uses faster-whisper for real-time transcription
- Processes audio in configurable chunks (default 5s) with 1s overlap
- Background thread for non-blocking transcription
- Integrated via `transcriber stream` command
- VAD (Voice Activity Detection) filtering for cleaner output

## Data Storage

```
~/.audio-transcriber/
├── recordings/       # WAV files
└── transcriptions/   # JSON + TXT files
```

## Dependencies

- `pyobjc-*` - Python/Objective-C bridge for ScreenCaptureKit
- `openai-whisper` - Batch transcription (used by `record` command)
- `faster-whisper` - Streaming transcription (used by `stream` command, CTranslate2-based, faster)
- `numpy`, `scipy` - Audio processing
- `click`, `rich` - CLI

## Requirements

- macOS 13.0+ (ScreenCaptureKit)
- Python 3.9+
- Screen Recording permission (System Settings > Privacy & Security)
- Microphone permission (only if using `--mic`)

## Development Notes

- Virtual environment at `env/`
- Install with: `pip install -e .`
- Entry point: `transcriber` command

## Current State

- ✅ System audio capture working
- ✅ Microphone capture working
- ✅ Batch transcription working (`transcriber record`)
- ✅ Streaming transcription working (`transcriber stream`)

## More Info

See ARCHITECTURE.md for detailed technical documentation.
