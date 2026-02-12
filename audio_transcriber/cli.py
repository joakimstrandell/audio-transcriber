"""Command-line interface for audio transcriber."""

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.text import Text

from .recorder import AudioRecorder
from .transcriber import Transcriber
from .streaming import StreamingRecorderTranscriber

console = Console()


@click.group()
@click.version_option()
def main():
    """Audio Transcriber - Record and transcribe system audio using ScreenCaptureKit."""
    pass


@main.command()
@click.option("--model", "-m", default="base", help="Whisper model: tiny, base, small, medium, large")
@click.option("--language", "-l", default=None, help="Language code (e.g., 'en', 'sv'). Auto-detects if not specified.")
@click.option("--mic", is_flag=True, help="Also capture microphone input")
def record(model: str, language: Optional[str], mic: bool):
    """Start recording system audio. Press Ctrl+C to stop and transcribe.

    This captures all audio playing on your system using macOS ScreenCaptureKit.
    Use --mic to also capture microphone input.
    Requires Screen Recording permission (will be requested on first run).
    """
    recorder = AudioRecorder(capture_microphone=mic)
    transcriber = Transcriber(model_name=model)

    mic_status = " + microphone" if mic else ""
    console.print(Panel.fit(
        f"[bold green]Starting audio recording{mic_status}...[/bold green]\n"
        "Press [bold yellow]Ctrl+C[/bold yellow] to stop recording and transcribe.\n\n"
        f"[dim]Using ScreenCaptureKit - captures system audio{mic_status}.[/dim]",
        title="Audio Transcriber",
    ))

    try:
        audio_file = recorder.start()
        console.print(f"[dim]Recording to: {audio_file}[/dim]\n")

        # Show recording status
        start_time = time.time()
        with Live(console=console, refresh_per_second=4) as live:
            while True:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)

                # Check for errors
                error = recorder.get_error()
                if error:
                    text = Text()
                    text.append("Recording... ", style="bold red")
                    text.append(f"{minutes:02d}:{seconds:02d}", style="bold cyan")
                    text.append(f" [Warning: {error}]", style="yellow")
                    live.update(text)
                else:
                    text = Text()
                    text.append("Recording... ", style="bold red")
                    text.append(f"{minutes:02d}:{seconds:02d}", style="bold cyan")
                    live.update(text)

                time.sleep(0.25)

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if "Screen Recording permission" in str(e) or "No displays found" in str(e):
            console.print("\n[yellow]To grant Screen Recording permission:[/yellow]")
            console.print("  1. Open System Settings > Privacy & Security > Screen Recording")
            console.print("  2. Enable permission for your terminal app")
            console.print("  3. Restart your terminal and try again")
        return

    console.print("\n\n[yellow]Stopping recording...[/yellow]")
    audio_file = recorder.stop()

    if audio_file and audio_file.exists():
        duration = recorder.get_recording_duration()
        console.print(f"[green]Audio saved to: {audio_file}[/green]")
        console.print(f"[dim]Duration: {duration:.1f} seconds[/dim]\n")

        if duration < 0.5:
            console.print("[yellow]Warning: Recording is very short. Make sure audio was playing.[/yellow]\n")

        console.print("[cyan]Transcribing audio...[/cyan]")
        with console.status("[bold cyan]Processing with Whisper..."):
            result = transcriber.transcribe(audio_file, language=language)

        if result["text"].strip():
            console.print()
            console.print(Panel.fit(
                result["text"],
                title=f"Transcription (Language: {result['language']})",
                border_style="green",
            ))
        else:
            console.print("\n[yellow]No speech detected in the recording.[/yellow]")

        console.print(f"\n[green]Transcription saved to:[/green]")
        console.print(f"  [dim]{result['output_file']}[/dim]")
    else:
        console.print("[red]No audio was recorded.[/red]")
        console.print("[dim]Make sure some audio is playing on your system during recording.[/dim]")


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "-m", default="base", help="Whisper model: tiny, base, small, medium, large")
@click.option("--language", "-l", default=None, help="Language code (e.g., 'en', 'sv'). Auto-detects if not specified.")
def transcribe(audio_file: Path, model: str, language: Optional[str]):
    """Transcribe an existing audio file."""
    transcriber = Transcriber(model_name=model)

    console.print(f"[cyan]Transcribing: {audio_file}[/cyan]")
    with console.status("[bold cyan]Processing with Whisper..."):
        result = transcriber.transcribe(audio_file, language=language)

    console.print()
    console.print(Panel.fit(
        result["text"],
        title=f"Transcription (Language: {result['language']})",
        border_style="green",
    ))

    console.print(f"\n[green]Transcription saved to:[/green]")
    console.print(f"  [dim]{result['output_file']}[/dim]")


@main.command()
@click.option("--model", "-m", default="base", help="Whisper model: tiny, base, small, medium, large-v2, large-v3")
@click.option("--language", "-l", default=None, help="Language code (e.g., 'en', 'sv'). Auto-detects if not specified.")
@click.option("--mic", is_flag=True, help="Also capture microphone input")
@click.option("--chunk", "-c", default=5.0, help="Chunk duration in seconds for streaming (default: 5.0)")
@click.option("--no-vad", is_flag=True, help="Disable Voice Activity Detection (try if mic not transcribing)")
def stream(model: str, language: Optional[str], mic: bool, chunk: float, no_vad: bool):
    """Start recording with real-time streaming transcription.

    Uses faster-whisper to transcribe audio in chunks while recording continues.
    Partial transcription results are displayed as they become available.

    Press Ctrl+C to stop recording.
    """
    # Track transcription segments for display
    transcript_lines = []
    transcript_lock = threading.Lock()

    def on_transcript(text: str, is_partial: bool):
        """Callback for new transcription segments."""
        with transcript_lock:
            transcript_lines.append(text)

    # Initialize the streaming recorder-transcriber
    streamer = StreamingRecorderTranscriber(
        model_name=model,
        language=language,
        capture_microphone=mic,
        chunk_duration=chunk,
        vad_filter=not no_vad,
    )

    mic_status = " + microphone" if mic else ""
    console.print(Panel.fit(
        f"[bold green]Starting streaming transcription{mic_status}...[/bold green]\n"
        "Press [bold yellow]Ctrl+C[/bold yellow] to stop recording.\n\n"
        f"[dim]Using faster-whisper for real-time transcription.[/dim]\n"
        f"[dim]Processing audio in {chunk:.1f}-second chunks.[/dim]",
        title="Streaming Transcriber",
    ))

    # Load model before starting
    with console.status("[bold cyan]Loading faster-whisper model..."):
        streamer.load_model()

    try:
        audio_file = streamer.start(on_transcript=on_transcript)
        console.print(f"[dim]Recording to: {audio_file}[/dim]\n")
        console.print("[bold cyan]Live Transcription:[/bold cyan]\n")

        # Show recording status and live transcription
        start_time = time.time()
        last_line_count = 0

        with Live(console=console, refresh_per_second=4) as live:
            while True:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)

                # Build the display
                display = Text()

                # Status line with audio level
                error = streamer.get_error()
                peak, rms = streamer.get_audio_levels()

                display.append("Recording... ", style="bold red")
                display.append(f"{minutes:02d}:{seconds:02d}", style="bold cyan")

                # Show audio level indicator
                level_bars = int(rms * 20)  # 0-20 bars based on RMS
                level_str = "█" * level_bars + "░" * (20 - level_bars)
                if rms > 0.01:
                    display.append(f"  [{level_str}]", style="green")
                else:
                    display.append(f"  [{level_str}]", style="dim")

                if error:
                    display.append(f" [Warning: {error}]", style="yellow")

                display.append("\n\n")

                # Show transcription so far
                with transcript_lock:
                    if transcript_lines:
                        current_text = " ".join(transcript_lines)
                        display.append(current_text, style="white")

                        # Show indicator if new text arrived
                        if len(transcript_lines) > last_line_count:
                            display.append(" ", style="default")
                            display.append("[new]", style="bold green")
                            last_line_count = len(transcript_lines)
                    else:
                        display.append("[dim]Waiting for speech...[/dim]")

                live.update(display)
                time.sleep(0.25)

    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if "Screen Recording permission" in str(e) or "No displays found" in str(e):
            console.print("\n[yellow]To grant Screen Recording permission:[/yellow]")
            console.print("  1. Open System Settings > Privacy & Security > Screen Recording")
            console.print("  2. Enable permission for your terminal app")
            console.print("  3. Restart your terminal and try again")
        return

    console.print("\n\n[yellow]Stopping recording...[/yellow]")
    audio_file, final_text = streamer.stop()

    if audio_file and audio_file.exists():
        duration = streamer.get_recording_duration()
        console.print(f"[green]Audio saved to: {audio_file}[/green]")
        console.print(f"[dim]Duration: {duration:.1f} seconds[/dim]\n")

        if final_text.strip():
            console.print()
            console.print(Panel.fit(
                final_text,
                title="Final Transcription",
                border_style="green",
            ))

            # Save the transcription
            transcriber = Transcriber(model_name=model)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcription_data = {
                "id": timestamp,
                "timestamp": datetime.now().isoformat(),
                "audio_file": str(audio_file),
                "text": final_text,
                "language": language or "auto",
                "model": model,
                "streaming": True,
            }

            # Save plain text file
            text_path = transcriber.transcriptions_dir / f"transcription_{timestamp}.txt"
            with open(text_path, "w") as f:
                f.write(f"Transcription (Streaming) - {transcription_data['timestamp']}\n")
                f.write(f"Audio file: {transcription_data['audio_file']}\n")
                f.write(f"Language: {transcription_data['language']}\n")
                f.write(f"Model: {transcription_data['model']}\n")
                f.write("-" * 50 + "\n\n")
                f.write(final_text)
                f.write("\n")

            # Save JSON file
            json_path = transcriber.transcriptions_dir / f"transcription_{timestamp}.json"
            with open(json_path, "w") as f:
                json.dump(transcription_data, f, indent=2)

            console.print(f"\n[green]Transcription saved to:[/green]")
            console.print(f"  [dim]{text_path}[/dim]")
        else:
            console.print("\n[yellow]No speech detected in the recording.[/yellow]")
    else:
        console.print("[red]No audio was recorded.[/red]")
        console.print("[dim]Make sure some audio is playing on your system during recording.[/dim]")


@main.command(name="list")
@click.option("--limit", "-n", default=10, help="Number of transcriptions to show")
def list_transcriptions(limit: int):
    """List all saved transcriptions."""
    transcriber = Transcriber()
    transcriptions = transcriber.list_transcriptions()

    if not transcriptions:
        console.print("[yellow]No transcriptions found.[/yellow]")
        console.print("[dim]Use 'transcriber record' to create your first recording.[/dim]")
        return

    table = Table(title=f"Recent Transcriptions (showing {min(limit, len(transcriptions))} of {len(transcriptions)})")
    table.add_column("ID", style="cyan")
    table.add_column("Date/Time", style="green")
    table.add_column("Language", style="yellow")
    table.add_column("Preview", style="white", max_width=50)

    for t in transcriptions[:limit]:
        table.add_row(
            t["id"],
            t["timestamp"][:19].replace("T", " "),
            t["language"],
            t["text_preview"],
        )

    console.print(table)
    console.print("\n[dim]Use 'transcriber show <ID>' to view a full transcription.[/dim]")


@main.command()
@click.argument("transcription_id")
def show(transcription_id: str):
    """Show a specific transcription by ID."""
    transcriber = Transcriber()
    data = transcriber.get_transcription(transcription_id)

    if not data:
        console.print(f"[red]Transcription '{transcription_id}' not found.[/red]")
        return

    console.print(Panel.fit(
        f"[bold]Date:[/bold] {data['timestamp'][:19].replace('T', ' ')}\n"
        f"[bold]Language:[/bold] {data['language']}\n"
        f"[bold]Model:[/bold] {data['model']}\n"
        f"[bold]Audio file:[/bold] {data['audio_file']}",
        title=f"Transcription {transcription_id}",
        border_style="cyan",
    ))

    console.print("\n" + Panel(
        data["text"],
        title="Content",
        border_style="green",
    ))


@main.command()
def path():
    """Show the path where transcriptions are stored."""
    transcriber = Transcriber()
    console.print(f"[green]Transcriptions directory:[/green] {transcriber.transcriptions_dir}")

    recorder = AudioRecorder()
    console.print(f"[green]Recordings directory:[/green] {recorder.recordings_dir}")


@main.command()
def check():
    """Check if ScreenCaptureKit is available and permissions are granted."""
    console.print("[cyan]Checking ScreenCaptureKit availability...[/cyan]\n")

    try:
        import ScreenCaptureKit
        console.print("[green]✓[/green] ScreenCaptureKit framework available")
    except ImportError as e:
        console.print(f"[red]✗[/red] ScreenCaptureKit not available: {e}")
        console.print("  [dim]Make sure you're on macOS 12.3 or later[/dim]")
        return

    try:
        recorder = AudioRecorder()
        content = recorder._get_shareable_content()
        if content and content.displays():
            console.print("[green]✓[/green] Screen Recording permission granted")
            console.print(f"  [dim]Found {len(content.displays())} display(s)[/dim]")
        else:
            console.print("[yellow]![/yellow] No displays found - permission may not be granted")
    except RuntimeError as e:
        console.print(f"[red]✗[/red] Permission check failed: {e}")
        console.print("\n[yellow]To grant Screen Recording permission:[/yellow]")
        console.print("  1. Open System Settings > Privacy & Security > Screen Recording")
        console.print("  2. Enable permission for your terminal app")
        console.print("  3. Restart your terminal and try again")
        return

    console.print("\n[green]Ready to record![/green] Run 'transcriber record' to start.")


if __name__ == "__main__":
    main()
