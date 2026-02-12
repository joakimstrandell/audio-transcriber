"""Transcription module using OpenAI Whisper."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import wavfile
import whisper


class Transcriber:
    """Transcribes audio files using OpenAI Whisper."""

    def __init__(
        self,
        model_name: str = "base",
        transcriptions_dir: Optional[Path] = None,
    ):
        """Initialize the transcriber.

        Args:
            model_name: Whisper model to use. Options: tiny, base, small, medium, large
                       Larger models are more accurate but slower and require more memory.
            transcriptions_dir: Directory to save transcription files.
        """
        self.model_name = model_name
        self.model = None  # Lazy load
        self.transcriptions_dir = transcriptions_dir or Path.home() / ".audio-transcriber" / "transcriptions"
        self.transcriptions_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self):
        """Load the Whisper model (lazy loading)."""
        if self.model is None:
            self.model = whisper.load_model(self.model_name)

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> dict:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Optional language code (e.g., 'en', 'sv'). If None, auto-detects.

        Returns:
            Dictionary containing transcription results and metadata.
        """
        self._load_model()

        # Load audio directly from WAV file (avoids ffmpeg dependency)
        sample_rate, audio_data = wavfile.read(audio_path)

        # Convert to float32 in range [-1, 1] as Whisper expects
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)

        # Transcribe the audio
        options = {"fp16": False}  # Avoid FP16 warning on CPU
        if language:
            options["language"] = language

        result = self.model.transcribe(audio_data, **options)

        # Create transcription record
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcription_data = {
            "id": timestamp,
            "timestamp": datetime.now().isoformat(),
            "audio_file": str(audio_path),
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
            "model": self.model_name,
        }

        # Save transcription
        output_path = self._save_transcription(transcription_data, timestamp)
        transcription_data["output_file"] = str(output_path)

        return transcription_data

    def _save_transcription(self, data: dict, timestamp: str) -> Path:
        """Save transcription to file.

        Creates both a plain text file and a JSON file with full metadata.
        """
        # Save plain text file
        text_path = self.transcriptions_dir / f"transcription_{timestamp}.txt"
        with open(text_path, "w") as f:
            f.write(f"Transcription - {data['timestamp']}\n")
            f.write(f"Audio file: {data['audio_file']}\n")
            f.write(f"Language: {data['language']}\n")
            f.write(f"Model: {data['model']}\n")
            f.write("-" * 50 + "\n\n")
            f.write(data["text"])
            f.write("\n")

        # Save JSON file with full metadata
        json_path = self.transcriptions_dir / f"transcription_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return text_path

    def list_transcriptions(self) -> list:
        """List all saved transcriptions.

        Returns:
            List of transcription metadata dictionaries.
        """
        transcriptions = []
        for json_file in sorted(self.transcriptions_dir.glob("transcription_*.json"), reverse=True):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    transcriptions.append({
                        "id": data.get("id", json_file.stem),
                        "timestamp": data.get("timestamp", "unknown"),
                        "language": data.get("language", "unknown"),
                        "text_preview": data.get("text", "")[:100] + "..." if len(data.get("text", "")) > 100 else data.get("text", ""),
                        "text_file": str(json_file.with_suffix(".txt")),
                        "json_file": str(json_file),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

        return transcriptions

    def get_transcription(self, transcription_id: str) -> Optional[dict]:
        """Get a specific transcription by ID.

        Args:
            transcription_id: The transcription ID (timestamp).

        Returns:
            Transcription data dictionary, or None if not found.
        """
        json_path = self.transcriptions_dir / f"transcription_{transcription_id}.json"
        if json_path.exists():
            with open(json_path) as f:
                return json.load(f)
        return None
