"""Command-line interface for audio transcriber."""

import sys
import time
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

console = Console()


@click.group()
@click.version_option()
def main():
    """Audio Transcriber - Record and transcribe system audio using ScreenCaptureKit."""
    pass


@main.command()
@click.option("--model", "-m", default="base", help="Whisper model: tiny, base, small, medium, large")
@click.option("--language", "-l", default=None, help="Language code (e.g., 'en', 'sv'). Auto-detects if not specified.")
def record(model: str, language: Optional[str]):
    """Start recording system audio. Press Ctrl+C to stop and transcribe.

    This captures all audio playing on your system using macOS ScreenCaptureKit.
    Requires Screen Recording permission (will be requested on first run).
    """
    recorder = AudioRecorder()
    transcriber = Transcriber(model_name=model)

    console.print(Panel.fit(
        "[bold green]Starting system audio recording...[/bold green]\n"
        "Press [bold yellow]Ctrl+C[/bold yellow] to stop recording and transcribe.\n\n"
        "[dim]Using ScreenCaptureKit - captures all system audio output.[/dim]",
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
            console.print("\n" + Panel.fit(
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

    console.print("\n" + Panel.fit(
        result["text"],
        title=f"Transcription (Language: {result['language']})",
        border_style="green",
    ))

    console.print(f"\n[green]Transcription saved to:[/green]")
    console.print(f"  [dim]{result['output_file']}[/dim]")


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
