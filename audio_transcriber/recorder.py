"""Audio recording module using macOS ScreenCaptureKit for system audio capture."""

import os
import threading
import queue
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np

# macOS ScreenCaptureKit imports
import objc
import ctypes
from Foundation import NSObject, NSRunLoop, NSDate, NSDefaultRunLoopMode

# Load libdispatch for creating dispatch queues
_libdispatch = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_dispatch_queue_create = _libdispatch.dispatch_queue_create
_dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
_dispatch_queue_create.restype = ctypes.c_void_p
from AVFoundation import (
    AVAudioPCMBuffer,
    AVAudioFormat,
    AVAudioCommonFormat,
)
import ScreenCaptureKit
from ScreenCaptureKit import (
    SCShareableContent,
    SCContentFilter,
    SCStreamConfiguration,
    SCStream,
    SCStreamOutputTypeAudio,
    SCCaptureResolutionNominal,
)
import CoreMedia


class AudioRecorder:
    """Records system audio using macOS ScreenCaptureKit API.

    This captures all system audio output without requiring virtual audio devices.
    Optionally captures microphone input as well (using sounddevice).
    Requires macOS 13.0+ and Screen Recording permission.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        recordings_dir: Optional[Path] = None,
        capture_microphone: bool = False,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.capture_microphone = capture_microphone
        self.recordings_dir = recordings_dir or Path.home() / ".audio-transcriber" / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._audio_data: list = []
        self._stream: Optional[SCStream] = None
        self._current_file: Optional[Path] = None
        self._stream_output: Optional["StreamOutput"] = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        self._audio_queue = objc.objc_object(c_void_p=_dispatch_queue_create(b"com.audio-transcriber.audio", None))

        # Audio level monitoring
        self._peak_level: float = 0.0
        self._rms_level: float = 0.0

        # Microphone capture (using sounddevice instead of ScreenCaptureKit)
        self._mic_stream = None
        self._mic_data: list = []

    def _get_shareable_content(self) -> Optional[SCShareableContent]:
        """Get shareable content (displays, windows, apps) synchronously."""
        content_holder = {"content": None, "error": None}
        event = threading.Event()

        def completion_handler(content, error):
            if error:
                content_holder["error"] = str(error)
            else:
                content_holder["content"] = content
            event.set()

        SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)

        # Wait for completion with timeout
        event.wait(timeout=10.0)

        if content_holder["error"]:
            raise RuntimeError(f"Failed to get shareable content: {content_holder['error']}")

        return content_holder["content"]

    def _mic_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice microphone input."""
        if not self._recording:
            return

        # Copy the audio data (float32)
        audio_data = indata[:, 0].copy()  # Take first channel if stereo

        # Update audio level monitoring from mic
        if len(audio_data) > 0:
            self._peak_level = max(self._peak_level, float(np.abs(audio_data).max()))
            self._rms_level = float(np.sqrt(np.mean(audio_data ** 2)))

        with self._lock:
            if self._recording:
                self._mic_data.append(audio_data)

    def start(self) -> Path:
        """Start recording system audio.

        Returns:
            Path to the recording file that will be created.
        """
        if self._recording:
            raise RuntimeError("Already recording")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file = self.recordings_dir / f"recording_{timestamp}.wav"
        self._audio_data = []
        self._mic_data = []
        self._error = None

        # Start microphone capture if enabled (using sounddevice)
        if self.capture_microphone:
            try:
                import sounddevice as sd
                self._mic_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    callback=self._mic_callback,
                    blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                )
                self._mic_stream.start()
            except Exception as e:
                self._error = f"Microphone error: {e}"

        # Get shareable content
        content = self._get_shareable_content()
        if not content or not content.displays():
            raise RuntimeError("No displays found. Make sure Screen Recording permission is granted.")

        # Use the first display for audio capture
        display = content.displays()[0]

        # Create content filter for the display (we only want audio, but need a display target)
        content_filter = SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
            display, [], []
        )

        # Configure stream for audio capture
        config = SCStreamConfiguration.alloc().init()
        config.setCapturesAudio_(True)
        config.setExcludesCurrentProcessAudio_(True)  # Don't capture our own audio
        config.setSampleRate_(self.sample_rate)
        config.setChannelCount_(self.channels)
        # Note: We don't use ScreenCaptureKit's captureMicrophone - it doesn't work reliably
        # Instead we use sounddevice for mic capture

        # Minimize video capture since we only want audio
        config.setCaptureResolution_(SCCaptureResolutionNominal)
        config.setWidth_(2)
        config.setHeight_(2)
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 1))  # 1 fps minimum
        config.setShowsCursor_(False)

        # Create stream
        self._stream = SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, None
        )

        # Create and add output handler
        self._stream_output = StreamOutput.alloc().initWithCallback_(self._handle_audio)

        error_ptr = objc.nil
        success = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._stream_output,
            SCStreamOutputTypeAudio,
            self._audio_queue,
            error_ptr,
        )

        if not success:
            raise RuntimeError("Failed to add stream output")

        # Start the stream
        start_event = threading.Event()
        start_result = {"error": None}

        def start_handler(error):
            if error:
                start_result["error"] = str(error)
            start_event.set()

        self._stream.startCaptureWithCompletionHandler_(start_handler)

        # Pump the run loop while waiting for start to complete
        for _ in range(100):  # 10 second timeout (100 * 0.1s)
            if start_event.is_set():
                break
            NSRunLoop.mainRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

        if start_result["error"]:
            raise RuntimeError(f"Failed to start capture: {start_result['error']}")

        self._recording = True
        return self._current_file

    def _handle_audio(self, sample_buffer):
        """Handle incoming audio samples from ScreenCaptureKit (system audio)."""
        if not self._recording:
            return

        try:
            # Get audio buffer from CMSampleBuffer
            block_buffer = CoreMedia.CMSampleBufferGetDataBuffer(sample_buffer)
            if block_buffer is None:
                return

            # Get the audio data length
            length = CoreMedia.CMBlockBufferGetDataLength(block_buffer)
            if length == 0:
                return

            # Create a buffer to receive the audio data
            buffer = bytearray(length)
            result = CoreMedia.CMBlockBufferCopyDataBytes(block_buffer, 0, length, buffer)

            # Result is (status, filled_buffer) tuple
            if isinstance(result, tuple):
                status = result[0]
                filled_buffer = result[1]
            else:
                status = result
                filled_buffer = buffer

            if status == 0:  # Success
                # Convert to numpy array (float32 samples)
                audio_data = np.frombuffer(bytes(filled_buffer), dtype=np.float32).copy()

                # Update audio level monitoring (only if mic not capturing, to avoid overwriting)
                if not self.capture_microphone and len(audio_data) > 0:
                    self._peak_level = max(self._peak_level, float(np.abs(audio_data).max()))
                    self._rms_level = float(np.sqrt(np.mean(audio_data ** 2)))

                with self._lock:
                    if self._recording:
                        self._audio_data.append(audio_data)

        except Exception as e:
            self._error = str(e)

    def stop(self) -> Optional[Path]:
        """Stop recording and save the audio file.

        Returns:
            Path to the saved recording file, or None if no recording was active.
        """
        if not self._recording:
            return None

        self._recording = False

        # Stop microphone stream
        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None

        # Stop the ScreenCaptureKit stream
        if self._stream:
            stop_event = threading.Event()

            def stop_handler(error):
                stop_event.set()

            self._stream.stopCaptureWithCompletionHandler_(stop_handler)

            # Pump the run loop while waiting for stop to complete
            for _ in range(50):  # 5 second timeout (50 * 0.1s)
                if stop_event.is_set():
                    break
                NSRunLoop.mainRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

            self._stream = None

        # Combine and save the recorded audio
        with self._lock:
            # Get system audio
            system_audio = np.concatenate(self._audio_data) if self._audio_data else np.array([], dtype=np.float32)

            # Get microphone audio
            mic_audio = np.concatenate(self._mic_data) if self._mic_data else np.array([], dtype=np.float32)

            # Mix or select audio
            if len(mic_audio) > 0 and len(system_audio) > 0:
                # Mix both: align lengths and add together
                min_len = min(len(system_audio), len(mic_audio))
                mixed = system_audio[:min_len] + mic_audio[:min_len]
                # Clip to prevent clipping
                mixed = np.clip(mixed, -1.0, 1.0)
                self._save_wav(mixed)
            elif len(mic_audio) > 0:
                # Mic only
                self._save_wav(mic_audio)
            elif len(system_audio) > 0:
                # System audio only
                self._save_wav(system_audio)
            else:
                return None

            return self._current_file

    def _save_wav(self, audio_data: np.ndarray):
        """Save audio data to WAV file."""
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)

        # Handle stereo to mono conversion if needed
        if self.channels == 1 and len(audio_int16.shape) > 1:
            audio_int16 = audio_int16.mean(axis=1).astype(np.int16)

        with wave.open(str(self._current_file), 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_recording_duration(self) -> float:
        """Get the current recording duration in seconds."""
        with self._lock:
            # Use mic data length if capturing mic, otherwise system audio
            if self.capture_microphone and self._mic_data:
                total_samples = sum(len(chunk) for chunk in self._mic_data)
            elif self._audio_data:
                total_samples = sum(len(chunk) for chunk in self._audio_data)
            else:
                return 0.0
            return total_samples / self.sample_rate

    def get_error(self) -> Optional[str]:
        """Get any error that occurred during recording."""
        return self._error

    def get_audio_levels(self) -> tuple[float, float]:
        """Get current audio levels for monitoring.

        Returns:
            Tuple of (peak_level, rms_level) where values are 0.0-1.0.
            Peak is the max amplitude seen, RMS is root mean square of recent audio.
        """
        return (self._peak_level, self._rms_level)

    def reset_peak_level(self):
        """Reset the peak level tracker."""
        self._peak_level = 0.0


# Get the SCStreamOutput protocol
SCStreamOutput_Protocol = objc.protocolNamed("SCStreamOutput")


# Objective-C class for handling stream output
class StreamOutput(NSObject, protocols=[SCStreamOutput_Protocol]):
    """Objective-C delegate class for receiving ScreenCaptureKit audio output."""

    def initWithCallback_(self, callback: Callable):
        self = objc.super(StreamOutput, self).init()
        if self is None:
            return None
        self._callback = callback
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream, sample_buffer, output_type):
        """Called when new audio samples are available."""
        if output_type == SCStreamOutputTypeAudio:
            if self._callback:
                self._callback(sample_buffer)
