# Audio Transcriber

A macOS CLI tool that captures system audio and transcribes it locally using OpenAI's Whisper. No cloud services, no API keys—everything runs on your machine.

## Features

- **System Audio Capture** - Records all audio playing on your Mac using ScreenCaptureKit
- **Microphone Support** - Optionally capture microphone input alongside system audio
- **Local Transcription** - Uses Whisper models running entirely on your machine
- **Real-time Streaming** - See transcription results as you record with `stream` command
- **Batch Processing** - Transcribe existing audio files
- **Multi-language** - Auto-detects language or manually specify

## Requirements

- macOS 13.0+ (Ventura or later)
- Python 3.9+
- Screen Recording permission

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install
pip install -e .
```

On first run, you'll be prompted to grant Screen Recording permission:
**System Settings → Privacy & Security → Screen Recording → [Your Terminal App]**

## Usage

### Record and Transcribe

```bash
# Record system audio, press Ctrl+C to stop and transcribe
transcriber record

# Also capture microphone
transcriber record --mic

# Use a larger model for better accuracy
transcriber record -m medium
```

### Real-time Streaming

```bash
# See transcription as you record
transcriber stream

# Faster updates with smaller chunks
transcriber stream -c 3.0
```

### Other Commands

```bash
# Transcribe an existing audio file
transcriber transcribe recording.wav

# List saved transcriptions
transcriber list

# View a specific transcription
transcriber show <id>

# Check permissions and setup
transcriber check

# Show storage directories
transcriber path
```

## Model Sizes

| Model  | Size   | Speed    | Accuracy |
|--------|--------|----------|----------|
| tiny   | 39 MB  | Fastest  | Lower    |
| base   | 74 MB  | Fast     | Good     |
| small  | 244 MB | Medium   | Better   |
| medium | 769 MB | Slow     | High     |
| large  | 1.5 GB | Slowest  | Highest  |

Models are downloaded automatically on first use.

## How It Works

1. **ScreenCaptureKit** captures system audio (the native macOS API for screen/audio recording)
2. Audio is saved as 16kHz mono WAV (optimal for Whisper)
3. **Whisper** (batch) or **faster-whisper** (streaming) transcribes locally
4. Results saved to `~/.audio-transcriber/`

## Storage

```
~/.audio-transcriber/
├── recordings/       # WAV audio files
└── transcriptions/   # JSON and TXT transcripts
```

## License

MIT
