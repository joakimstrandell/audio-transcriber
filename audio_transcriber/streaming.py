"""Streaming transcription module using faster-whisper."""

import threading
import queue
import time
from typing import Optional, Callable, Tuple
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel


class StreamingTranscriber:
    """Real-time streaming transcriber using faster-whisper.

    Processes audio in chunks and provides partial results as they become available.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "float32",  # Avoid float16 warning on CPU/macOS
        language: Optional[str] = None,
        vad_filter: bool = True,
        vad_threshold: float = 0.3,  # Lower = more sensitive (default 0.5 is too aggressive for mic)
    ):
        """Initialize the streaming transcriber.

        Args:
            model_name: Whisper model to use. Options: tiny, base, small, medium, large-v2, large-v3
            device: Device to use for inference. Options: auto, cpu, cuda
            compute_type: Compute type for inference. Options: default, float16, int8
            language: Language code for transcription. If None, auto-detects.
            vad_filter: Whether to use Voice Activity Detection to filter non-speech.
            vad_threshold: VAD threshold (0.0-1.0). Lower = more sensitive to quiet speech.
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.vad_threshold = vad_threshold
        self.model: Optional[WhisperModel] = None

        # Streaming state
        self._audio_buffer: list = []
        self._buffer_lock = threading.Lock()
        self._transcription_thread: Optional[threading.Thread] = None
        self._running = False
        self._result_queue: queue.Queue = queue.Queue()
        self._full_transcript: list = []
        self._on_transcript: Optional[Callable[[str, bool], None]] = None

        # Configuration
        self.chunk_duration = 5.0  # Process audio in 5-second chunks
        self.sample_rate = 16000
        self.min_chunk_samples = int(self.chunk_duration * self.sample_rate)

        # Overlap for better continuity (1 second overlap)
        self.overlap_duration = 1.0
        self.overlap_samples = int(self.overlap_duration * self.sample_rate)

        # Track processed samples to avoid repeating
        self._processed_samples = 0

    def _load_model(self):
        """Load the faster-whisper model (lazy loading)."""
        if self.model is None:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

    def load_model(self):
        """Public method to pre-load the model."""
        self._load_model()

    def start(self, on_transcript: Optional[Callable[[str, bool], None]] = None):
        """Start the streaming transcription.

        Args:
            on_transcript: Callback function called with (text, is_partial).
                          is_partial is True for intermediate results, False for final.
        """
        self._load_model()
        self._running = True
        self._audio_buffer = []
        self._full_transcript = []
        self._processed_samples = 0
        self._on_transcript = on_transcript

        # Start background transcription thread
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
        )
        self._transcription_thread.start()

    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to the buffer for transcription.

        Args:
            audio_data: Audio samples as float32 numpy array, expected at 16kHz mono.
        """
        if not self._running:
            return

        with self._buffer_lock:
            self._audio_buffer.append(audio_data.copy())

    def _get_audio_chunk(self) -> Optional[np.ndarray]:
        """Get audio chunk from buffer if enough samples are available."""
        with self._buffer_lock:
            if not self._audio_buffer:
                return None

            # Concatenate all buffered audio
            total_audio = np.concatenate(self._audio_buffer)
            total_samples = len(total_audio)

            # Check if we have enough for a chunk
            if total_samples < self.min_chunk_samples:
                return None

            # Take a chunk for processing
            chunk = total_audio[:self.min_chunk_samples].copy()

            # Keep overlap for next chunk (better continuity)
            remaining_start = self.min_chunk_samples - self.overlap_samples
            if remaining_start > 0:
                remaining = total_audio[remaining_start:]
                self._audio_buffer = [remaining] if len(remaining) > 0 else []
            else:
                self._audio_buffer = []

            return chunk

    def _transcription_loop(self):
        """Background loop that processes audio chunks."""
        while self._running:
            chunk = self._get_audio_chunk()

            if chunk is not None:
                # Transcribe the chunk
                text = self._transcribe_chunk(chunk)
                if text.strip():
                    self._full_transcript.append(text.strip())
                    if self._on_transcript:
                        self._on_transcript(text.strip(), True)
                    self._result_queue.put(("partial", text.strip()))
            else:
                # No chunk available, wait a bit
                time.sleep(0.1)

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe a single audio chunk.

        Args:
            audio: Audio samples as float32 numpy array at 16kHz.

        Returns:
            Transcribed text.
        """
        if self.model is None:
            return ""

        try:
            # Transcribe with faster-whisper
            # VAD parameters tuned for microphone input (lower threshold = more sensitive)
            vad_params = dict(
                threshold=self.vad_threshold,
                min_silence_duration_ms=300,  # Shorter silence detection
                speech_pad_ms=200,  # Padding around speech
            ) if self.vad_filter else None

            segments, info = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=self.vad_filter,
                vad_parameters=vad_params,
            )

            # Collect text from segments
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())

            return " ".join(text_parts)

        except Exception as e:
            return ""

    def stop(self) -> str:
        """Stop streaming and return the full transcript.

        Returns:
            The complete transcription text.
        """
        self._running = False

        # Wait for transcription thread to finish
        if self._transcription_thread and self._transcription_thread.is_alive():
            self._transcription_thread.join(timeout=2.0)

        # Process any remaining audio in the buffer
        with self._buffer_lock:
            if self._audio_buffer:
                remaining_audio = np.concatenate(self._audio_buffer)
                if len(remaining_audio) > self.sample_rate * 0.5:  # At least 0.5 seconds
                    text = self._transcribe_chunk(remaining_audio)
                    if text.strip():
                        self._full_transcript.append(text.strip())
                        if self._on_transcript:
                            self._on_transcript(text.strip(), False)
                self._audio_buffer = []

        return " ".join(self._full_transcript)

    def get_results(self, block: bool = False, timeout: float = 0.1):
        """Get transcription results from the queue.

        Args:
            block: Whether to block waiting for results.
            timeout: Timeout in seconds when blocking.

        Yields:
            Tuples of (result_type, text) where result_type is "partial" or "final".
        """
        while True:
            try:
                result = self._result_queue.get(block=block, timeout=timeout)
                yield result
            except queue.Empty:
                break

    def get_full_transcript(self) -> str:
        """Get the current full transcript.

        Returns:
            All transcribed text so far.
        """
        return " ".join(self._full_transcript)


class StreamingRecorderTranscriber:
    """Combines recorder with streaming transcription.

    This class wraps the AudioRecorder and StreamingTranscriber to provide
    a unified interface for recording and transcribing in real-time.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        compute_type: str = "default",
        language: Optional[str] = None,
        capture_microphone: bool = False,
        chunk_duration: float = 5.0,
        vad_filter: bool = True,
        vad_threshold: Optional[float] = None,
    ):
        """Initialize the streaming recorder-transcriber.

        Args:
            model_name: Whisper model to use.
            device: Device to use for inference.
            compute_type: Compute type for inference.
            language: Optional language code for transcription.
            capture_microphone: Whether to also capture microphone input.
            chunk_duration: Duration of audio chunks to process in seconds.
            vad_filter: Whether to use Voice Activity Detection.
            vad_threshold: VAD threshold (0.0-1.0). If None, uses 0.2 for mic, 0.3 otherwise.
        """
        from .recorder import AudioRecorder

        self.recorder = AudioRecorder(capture_microphone=capture_microphone)

        # Disable VAD by default for microphone input (mic levels are too low for VAD)
        # VAD works well for system audio but filters out quiet mic speech
        if capture_microphone and vad_filter:
            vad_filter = False  # Disable VAD for mic by default

        if vad_threshold is None:
            vad_threshold = 0.2 if capture_microphone else 0.3

        self.transcriber = StreamingTranscriber(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_filter=vad_filter,
            vad_threshold=vad_threshold,
        )
        self.transcriber.chunk_duration = chunk_duration
        self.transcriber.min_chunk_samples = int(chunk_duration * self.transcriber.sample_rate)

        self._original_handle_audio: Optional[Callable] = None
        self._running = False
        self._on_transcript: Optional[Callable[[str, bool], None]] = None

    def load_model(self):
        """Pre-load the transcription model."""
        self.transcriber.load_model()

    def start(
        self,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
    ) -> Path:
        """Start recording and streaming transcription.

        Args:
            on_transcript: Callback for transcription updates (text, is_partial).

        Returns:
            Path to the recording file.
        """
        self._on_transcript = on_transcript

        # Start the transcriber first
        self.transcriber.start(on_transcript=on_transcript)

        # Hook into the recorder's audio handling to feed transcriber
        self._original_handle_audio = self.recorder._handle_audio
        self._original_mic_callback = self.recorder._mic_callback

        def hooked_handle_audio(sample_buffer):
            # Call original handler to store audio
            self._original_handle_audio(sample_buffer)

            # Feed system audio to transcriber (only if not using mic)
            if not self.recorder.capture_microphone:
                with self.recorder._lock:
                    if self.recorder._audio_data:
                        latest_chunk = self.recorder._audio_data[-1]
                        self.transcriber.add_audio(latest_chunk)

        def hooked_mic_callback(indata, frames, time_info, status):
            # Call original handler to store audio
            self._original_mic_callback(indata, frames, time_info, status)

            # Feed mic audio to transcriber
            with self.recorder._lock:
                if self.recorder._mic_data:
                    latest_chunk = self.recorder._mic_data[-1]
                    self.transcriber.add_audio(latest_chunk)

        self.recorder._handle_audio = hooked_handle_audio
        self.recorder._mic_callback = hooked_mic_callback

        # Start recording
        self._running = True
        result = self.recorder.start()

        # Re-hook mic callback after start() creates the sounddevice stream
        # because start() replaces _mic_callback with a new reference
        if self.recorder._mic_stream:
            # The sounddevice stream uses a callback reference, we need to monkey-patch
            # the instance method that gets called
            self.recorder._mic_callback = hooked_mic_callback

        return result

    def stop(self) -> Tuple[Optional[Path], str]:
        """Stop recording and transcription.

        Returns:
            Tuple of (audio file path, final transcription text).
        """
        self._running = False

        # Stop recording first
        audio_file = self.recorder.stop()

        # Restore original handler
        if self._original_handle_audio:
            self.recorder._handle_audio = self._original_handle_audio

        # Stop transcription and get final text
        final_text = self.transcriber.stop()

        return audio_file, final_text

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recorder.is_recording()

    def get_recording_duration(self) -> float:
        """Get current recording duration in seconds."""
        return self.recorder.get_recording_duration()

    def get_current_text(self) -> str:
        """Get current accumulated transcription text."""
        return self.transcriber.get_full_transcript()

    def get_error(self) -> Optional[str]:
        """Get any recording error."""
        return self.recorder.get_error()

    def get_results(self, block: bool = False, timeout: float = 0.1):
        """Get transcription results from the queue.

        Yields:
            Tuples of (result_type, text).
        """
        yield from self.transcriber.get_results(block=block, timeout=timeout)

    def get_audio_levels(self) -> tuple[float, float]:
        """Get current audio levels (peak, rms) for monitoring."""
        return self.recorder.get_audio_levels()
