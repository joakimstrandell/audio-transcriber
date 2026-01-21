# Audio Transcriber - Architecture & Implementation

## Overview

A CLI tool that captures system audio on macOS using ScreenCaptureKit and transcribes it locally using OpenAI's Whisper model.

## System Requirements

- macOS 13.0+ (ScreenCaptureKit audio capture)
- Python 3.9+
- Screen Recording permission granted to terminal app
- No ffmpeg required (audio loaded directly via scipy)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (cli.py)                         │
│  Commands: record, transcribe, list, show, check, path      │
└─────────────────────┬───────────────────┬───────────────────┘
                      │                   │
                      ▼                   ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│     AudioRecorder           │ │     Transcriber             │
│     (recorder.py)           │ │     (transcriber.py)        │
├─────────────────────────────┤ ├─────────────────────────────┤
│ - ScreenCaptureKit API      │ │ - OpenAI Whisper (local)    │
│ - System audio capture      │ │ - Multi-language support    │
│ - WAV file output           │ │ - JSON + TXT output         │
└─────────────────────────────┘ └─────────────────────────────┘
                      │                   │
                      ▼                   ▼
              ~/.audio-transcriber/
              ├── recordings/       # WAV files
              └── transcriptions/   # JSON + TXT files
```

## Key Components

### 1. AudioRecorder (recorder.py)

Uses macOS ScreenCaptureKit to capture system audio without requiring virtual audio drivers.

**Key classes:**
- `AudioRecorder` - Main recording class
- `StreamOutput` - Objective-C delegate for receiving audio samples

**How it works:**
1. Gets shareable content (displays) via `SCShareableContent`
2. Creates `SCContentFilter` using `initWithDisplay_excludingApplications_exceptingWindows_`
3. Configures `SCStream` for audio-only capture (video minimized to 2x2 @ 1fps)
4. Creates dispatch queue for audio sample handling
5. Registers `StreamOutput` delegate to receive `CMSampleBuffer` audio data
6. Extracts audio bytes via `CMBlockBufferCopyDataBytes`
7. Pumps NSRunLoop during async start/stop operations
8. Converts audio to numpy arrays and saves as 16-bit PCM WAV

**Configuration:**
- Sample rate: 16kHz (optimal for Whisper)
- Channels: 1 (mono)
- Format: 16-bit PCM WAV
- Optional microphone capture (--mic flag)

### 2. Transcriber (transcriber.py)

Uses OpenAI Whisper for local speech-to-text transcription.

**Features:**
- Loads WAV files directly via scipy (no ffmpeg dependency)
- Lazy model loading (downloads on first use)
- Multiple model sizes: tiny, base, small, medium, large
- Auto language detection or manual specification
- Outputs both plain text and JSON with segments

**Model sizes:**
| Model  | Size   | Speed    | Accuracy |
|--------|--------|----------|----------|
| tiny   | 39 MB  | Fastest  | Lower    |
| base   | 74 MB  | Fast     | Good     |
| small  | 244 MB | Medium   | Better   |
| medium | 769 MB | Slow     | High     |
| large  | 1.5 GB | Slowest  | Highest  |

### 3. CLI (cli.py)

Built with Click and Rich for a nice terminal experience.

**Commands:**
- `transcriber record` - Start recording, Ctrl+C to stop and transcribe
  - `--mic` - Also capture microphone input
  - `-m/--model` - Whisper model size (tiny, base, small, medium, large)
  - `-l/--language` - Language code (auto-detects if not specified)
- `transcriber transcribe <file>` - Transcribe existing audio file
- `transcriber list` - List all saved transcriptions
- `transcriber show <id>` - View specific transcription
- `transcriber check` - Verify ScreenCaptureKit and permissions
- `transcriber path` - Show storage directories

## Data Flow

```
1. User runs: transcriber record
                    │
                    ▼
2. ScreenCaptureKit captures system audio
   (all audio playing through speakers)
                    │
                    ▼
3. Audio samples collected in memory
   (float32 numpy arrays)
                    │
                    ▼
4. User presses Ctrl+C
                    │
                    ▼
5. Audio saved to WAV file
   ~/.audio-transcriber/recordings/recording_YYYYMMDD_HHMMSS.wav
                    │
                    ▼
6. Whisper model loaded (if not cached)
                    │
                    ▼
7. Transcription generated
                    │
                    ▼
8. Results saved to:
   ~/.audio-transcriber/transcriptions/transcription_YYYYMMDD_HHMMSS.txt
   ~/.audio-transcriber/transcriptions/transcription_YYYYMMDD_HHMMSS.json
```

## Dependencies

```
pyobjc-core              - Python/Objective-C bridge
pyobjc-framework-ScreenCaptureKit - ScreenCaptureKit bindings
pyobjc-framework-AVFoundation     - Audio format handling
pyobjc-framework-CoreMedia        - Media buffer handling
numpy                    - Audio data processing
scipy                    - Signal processing
openai-whisper           - Speech-to-text model
click                    - CLI framework
rich                     - Terminal formatting
```

## Why ScreenCaptureKit?

Before macOS 12.3, capturing system audio required:
- Virtual audio drivers (BlackHole, Loopback)
- Complex Audio MIDI Setup configuration
- Routing audio through virtual devices

ScreenCaptureKit provides:
- Native API - no third-party drivers
- Direct system audio access
- Permission-based security model
- Low latency capture

## Permissions

The app requires **Screen Recording** permission because ScreenCaptureKit's audio capture is tied to screen capture APIs (even though we minimize video capture to 2x2 pixels at 1fps).

When using `--mic`, **Microphone** permission is also required.

Grant permissions via:
- System Settings → Privacy & Security → Screen Recording → [Your Terminal App]
- System Settings → Privacy & Security → Microphone → [Your Terminal App] (if using --mic)

## Future Improvements

Potential enhancements:
- [ ] Real-time transcription (stream to Whisper)
- [ ] Audio-only mode without display filter (if API supports)
- [ ] Speaker diarization (who said what)
- [ ] Export to SRT/VTT subtitles
- [ ] Web UI for playback with synced transcript
- [ ] Configurable audio quality settings
