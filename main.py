#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import evdev
from evdev import ecodes, UInput
import subprocess
import tempfile
import os
from pathlib import Path
from scipy.io import wavfile
from scipy import signal
import asyncio
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
import queue
import time
import json
import requests
import atexit
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
import readchar
from pydantic import BaseModel
from typing import Optional


class DeviceConfig(BaseModel):
    name: str
    path: str
    phys: Optional[str] = None


class AppConfig(BaseModel):
    device: Optional[DeviceConfig] = None


def get_config_path():
    """Get the path to the config file."""
    home = Path.home()
    config_dir = home / ".local" / "share" / "slap"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def load_config():
    """Load config from file or return default."""
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            return AppConfig.model_validate(data)
        except Exception as e:
            print(f"Error loading config: {e}")
    return AppConfig()


def save_config(config: AppConfig):
    """Save config to file."""
    config_path = get_config_path()
    try:
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


# SSE message queue
sse_queue = queue.Queue()

# OpenCode server globals
opencode_process = None
opencode_session_id = None
OPENCODE_PORT = 4096
OPENCODE_URL = f"http://127.0.0.1:{OPENCODE_PORT}"

app = Flask(__name__)


def start_opencode_server():
    """Start the opencode server in the background."""
    global opencode_process, opencode_session_id

    print("Starting OpenCode server...")
    opencode_process = subprocess.Popen(
        ["opencode", "serve", "--port", str(OPENCODE_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            # Try to connect to the server (any endpoint will do)
            response = requests.get(f"{OPENCODE_URL}/", timeout=1)
            print("OpenCode server started!")
            break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)
    else:
        print("Warning: OpenCode server may not be ready")

    # Create a session
    try:
        response = requests.post(
            f"{OPENCODE_URL}/session",
            json={"title": "Whisper Transcription Cleanup"},
            timeout=5,
        )
        if response.status_code == 200:
            opencode_session_id = response.json().get("id")
            print(f"Created OpenCode session: {opencode_session_id}")
        else:
            print(f"Failed to create session: {response.status_code}")
    except Exception as e:
        print(f"Error creating session: {e}")


def stop_opencode_server():
    """Stop the opencode server."""
    global opencode_process
    if opencode_process:
        print("Stopping OpenCode server...")
        opencode_process.terminate()
        try:
            opencode_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            opencode_process.kill()


def clean_transcription(raw_text):
    """Clean up transcription using OpenCode API."""
    global opencode_session_id

    if not opencode_session_id:
        print("No OpenCode session, returning raw text")
        return raw_text

    prompt = f"""Clean up this raw speech-to-text transcription. ONLY output the cleaned text with no additional commentary.

The transcription may have:
- Mistranscribed technical terms (programming terms, command names, etc.)
- Missing punctuation
- Awkward phrasing from spoken language

Raw transcription:
{raw_text}

Cleaned transcription:"""

    try:
        response = requests.post(
            f"{OPENCODE_URL}/session/{opencode_session_id}/message",
            json={
                "parts": [{"type": "text", "text": prompt}],
                "model": {"providerID": "anthropic", "modelID": "claude-haiku-4-5"},
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            # Extract text from parts
            parts = data.get("parts", [])
            cleaned_text = ""
            for part in parts:
                if part.get("type") == "text":
                    cleaned_text += part.get("text", "")

            if cleaned_text.strip():
                print(f"Cleaned: {raw_text} -> {cleaned_text.strip()}")
                return cleaned_text.strip()
        else:
            print(f"OpenCode API error: {response.status_code}")
    except Exception as e:
        print(f"Error cleaning transcription: {e}")

    return raw_text


# Register cleanup handler
atexit.register(stop_opencode_server)


def select_keyboard_device():
    """Interactive TUI for selecting a keyboard device."""
    console = Console()

    # Get all input devices
    device_paths = evdev.list_devices()

    if not device_paths:
        console.print("[red]No input devices accessible![/red]")
        console.print("\nYou need to either:")
        console.print("1. Add yourself to the 'input' group:")
        console.print("   sudo usermod -a -G input $USER")
        console.print("   (then log out and log back in)")
        console.print("2. Or run with sudo:")
        console.print("   sudo uv run main.py")
        return None

    # Load devices and deduplicate by (name, phys)
    devices = []
    seen = set()

    for path in device_paths:
        try:
            device = evdev.InputDevice(path)
            caps = device.capabilities(verbose=False)

            # Only include devices with keyboard capabilities
            if ecodes.EV_KEY in caps:
                # Create a unique key for deduplication
                unique_key = (device.name, device.phys)
                if unique_key not in seen:
                    seen.add(unique_key)
                    devices.append(device)
        except Exception as e:
            pass

    if not devices:
        console.print("[red]No keyboard devices found![/red]")
        return None

    # Sort devices by name for consistent ordering
    devices.sort(key=lambda d: d.name)

    selected_idx = 0

    def build_display(idx):
        """Build the display content."""
        from rich.console import Group

        # Create title panel
        title = Panel(
            "[bold cyan]Select Keyboard Device[/bold cyan]\n"
            "Use ↑/↓ or j/k to navigate, Enter to select, q to quit",
            border_style="cyan",
        )

        # Create device list table
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("", width=3)
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="dim")
        table.add_column("Physical Location", style="dim")

        for i, device in enumerate(devices):
            cursor = "[bold green]→[/bold green]" if i == idx else " "
            name_style = "bold green" if i == idx else "cyan"
            table.add_row(
                cursor,
                f"[{name_style}]{device.name}[/{name_style}]",
                device.path,
                device.phys or "N/A",
            )

        return Group(title, "", table)

    with Live(
        build_display(selected_idx), console=console, refresh_per_second=10
    ) as live:
        while True:
            # Read keyboard input
            key = readchar.readkey()

            # Handle navigation
            if key == readchar.key.UP or key == "k":
                selected_idx = (selected_idx - 1) % len(devices)
                live.update(build_display(selected_idx))
            elif key == readchar.key.DOWN or key == "j":
                selected_idx = (selected_idx + 1) % len(devices)
                live.update(build_display(selected_idx))
            elif key == readchar.key.ENTER or key == "\r" or key == "\n":
                selected = devices[selected_idx]
                live.stop()
                console.print(
                    f"[green]✓[/green] Selected: [bold]{selected.name}[/bold] ({selected.path})"
                )
                return selected
            elif key == "q" or key == readchar.key.ESC:
                live.stop()
                console.print("[yellow]Selection cancelled[/yellow]")
                return None


class WhisperRecorder:
    def __init__(self, whisper_path, model_path, recordings_dir):
        self.whisper_path = Path(whisper_path)
        self.model_path = Path(model_path)
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(exist_ok=True)
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.available_models = []

        # Get device's native sample rate and set whisper target
        device_info = sd.query_devices(kind="input")
        self.recording_sample_rate = int(device_info["default_samplerate"])
        self.whisper_sample_rate = 16000

        # Find available models (look in ./models relative to main.py)
        models_dir = Path(__file__).parent / "models"
        if models_dir.exists():
            self.available_models = [
                model
                for model in sorted(models_dir.glob("ggml-*.bin"))
                if "for-tests" not in model.name
            ]

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio recording."""
        if status:
            print(f"Audio status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            return

        print("Recording started... (Press F2 to stop)")
        self.is_recording = True
        self.audio_data = []

        # Start the audio stream at device's native sample rate
        self.stream = sd.InputStream(
            samplerate=self.recording_sample_rate,
            channels=1,
            callback=self.audio_callback,
            dtype=np.float32,
        )
        self.stream.start()

    def stop_recording(self):
        """Stop audio recording and process with whisper."""
        if not self.is_recording:
            return

        print("Recording stopped. Processing...")
        self.is_recording = False

        # Stop the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Check if we have audio data
        if not self.audio_data:
            print("No audio recorded!")
            return

        # Copy audio data before starting thread
        audio_data_copy = self.audio_data.copy()

        # Process audio in separate thread to not block event loop
        thread = threading.Thread(target=self._process_audio, args=(audio_data_copy,))
        thread.daemon = True
        thread.start()

    def _process_audio(self, audio_data):
        """Process audio data with whisper (runs in separate thread)."""
        # Combine all audio chunks
        audio_array = np.concatenate(audio_data, axis=0)

        # Resample to 16kHz if needed
        if self.recording_sample_rate != self.whisper_sample_rate:
            # Calculate number of samples after resampling
            num_samples = int(
                len(audio_array) * self.whisper_sample_rate / self.recording_sample_rate
            )
            audio_array = signal.resample(audio_array, num_samples)

        # Convert to int16 for WAV file
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            wavfile.write(tmp_filename, self.whisper_sample_rate, audio_int16)

        try:
            # Run whisper transcription
            result = subprocess.run(
                [
                    str(self.whisper_path),
                    "--model",
                    str(self.model_path),
                    "--file",
                    tmp_filename,
                    "--no-timestamps",
                    "--output-txt",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Try reading from output file first
                txt_file = tmp_filename.replace(".wav", ".txt")
                text = None

                if os.path.exists(txt_file):
                    with open(txt_file, "r") as f:
                        text = f.read().strip()
                    os.unlink(txt_file)
                else:
                    # Fall back to parsing stdout
                    text = result.stdout.strip()

                if text:
                    print(f"Raw transcription: {text}")
                    self._save_transcription(text)
                else:
                    print("No speech detected.")
            else:
                print(f"Whisper failed (code {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr[:200]}")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

    def _save_transcription(self, raw_text):
        """Save transcription to timestamped file and notify SSE clients."""
        # Clean up the transcription using OpenCode
        cleaned_text = clean_transcription(raw_text)

        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
        filename = f"{timestamp}.txt"
        filepath = self.recordings_dir / filename

        with open(filepath, "w") as f:
            f.write(cleaned_text)

        print(f"Saved to {filename}")

        # Copy to clipboard
        self._copy_to_clipboard(cleaned_text)

        # Notify SSE clients
        sse_queue.put(
            {"filename": filename, "text": cleaned_text, "timestamp": timestamp}
        )

    def _copy_to_clipboard(self, text):
        """Copy text to clipboard using wl-copy."""
        sudo_user = os.environ.get("SUDO_USER")

        if sudo_user:
            # Get the original user's UID
            import pwd

            pw = pwd.getpwnam(sudo_user)
            uid = pw.pw_uid
            runtime_dir = f"/run/user/{uid}"

            # Detect the actual Wayland display by checking for socket files
            wayland_display = "wayland-0"  # default
            try:
                runtime_path = Path(runtime_dir)
                if runtime_path.exists():
                    # Look for wayland-* files
                    for socket in runtime_path.glob("wayland-*"):
                        if socket.is_socket() or socket.name.startswith("wayland-"):
                            wayland_display = socket.name
                            break
            except:
                pass

            # Run wl-copy as original user with proper environment
            cmd = [
                "sudo",
                "-i",
                "-u",
                sudo_user,
                "env",
                f"XDG_RUNTIME_DIR={runtime_dir}",
                f"WAYLAND_DISPLAY={wayland_display}",
                "wl-copy",
            ]

            # Pass text via stdin
            try:
                subprocess.run(
                    cmd,
                    input=text,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                print("Copied to clipboard!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to copy to clipboard: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                print("Clipboard operation timed out")
            except FileNotFoundError:
                print("wl-copy not found. Install wl-clipboard package.")
        else:
            # Not running under sudo, use wl-copy directly
            try:
                subprocess.run(
                    ["wl-copy"], input=text, check=True, capture_output=True, text=True
                )
                print("Copied to clipboard!")
            except subprocess.CalledProcessError as e:
                print(f"Failed to copy to clipboard: {e.stderr.strip()}")
            except FileNotFoundError:
                print("wl-copy not found. Install wl-clipboard package.")

    def get_recordings(self):
        """Get list of recordings sorted by timestamp (newest first)."""
        recordings = []
        for file in sorted(self.recordings_dir.glob("*.txt"), reverse=True):
            with open(file, "r") as f:
                text = f.read()
            recordings.append(
                {
                    "filename": file.name,
                    "text": text,
                    "timestamp": file.name.replace(".txt", ""),
                }
            )
        return recordings

    def set_model(self, model_name):
        """Change the active model."""
        model_path = Path(__file__).parent / "models" / model_name
        if model_path.exists():
            self.model_path = model_path
            print(f"Switched to model: {model_name}")
            return True
        return False

    def toggle_recording(self):
        """Toggle recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def find_keyboard_device(self):
        """Find the configured keyboard input device."""
        config = load_config()
        device_paths = evdev.list_devices()

        if not device_paths:
            print("\nNo input devices accessible!")
            print("You need to either:")
            print("1. Add yourself to the 'input' group:")
            print("   sudo usermod -a -G input $USER")
            print("   (then log out and log back in)")
            print("2. Or run with sudo:")
            print("   sudo uv run main.py")
            return None

        devices = []
        for path in device_paths:
            try:
                devices.append(evdev.InputDevice(path))
            except Exception as e:
                print(f"Could not access {path}: {e}")

        if not devices:
            return None

        # If we have a configured device, try to find it first
        if config.device:
            # Try to match by path first (most reliable)
            for device in devices:
                if device.path == config.device.path:
                    print(f"Found configured device by path: {device.name}")
                    return device

            # Try to match by name and physical location
            for device in devices:
                if (
                    device.name == config.device.name
                    and device.phys == config.device.phys
                ):
                    print(f"Found configured device by name/phys: {device.name}")
                    return device

            # Try to match by name only (less reliable)
            for device in devices:
                if device.name == config.device.name:
                    print(f"Found configured device by name: {device.name}")
                    return device

            print(f"Configured device not found: {config.device.name}")

        # Fallback: find device with F2 key
        for device in devices:
            caps = device.capabilities(verbose=False)
            if ecodes.EV_KEY in caps and ecodes.KEY_F2 in caps[ecodes.EV_KEY]:
                print(f"Using fallback device with F2 key: {device.name}")
                return device

        # Final fallback: find any keyboard device
        for device in devices:
            caps = device.capabilities(verbose=False)
            if ecodes.EV_KEY in caps:
                # Check if it has typical keyboard keys
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_A in keys or ecodes.KEY_SPACE in keys:
                    print(f"Using fallback keyboard device: {device.name}")
                    return device

        return None

    async def listen_for_hotkey(self):
        """Async loop to listen for F2 key press."""
        device = self.find_keyboard_device()
        if not device:
            print("Error: No keyboard device found!")
            return

        print(f"Using input device: {device.name}")

        try:
            async for event in device.async_read_loop():
                # Check for F2 key press (key down event)
                if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_F2:
                    if event.value == 1:  # Key down (1 = press, 0 = release, 2 = hold)
                        self.toggle_recording()
        except Exception as e:
            print(f"Error reading events: {e}")


# Global recorder instance
recorder = None


# Flask routes
@app.route("/")
def index():
    """Render main page."""
    recordings = recorder.get_recordings()
    models = [model.name for model in recorder.available_models]
    current_model = recorder.model_path.name
    return render_template(
        "index.html", recordings=recordings, models=models, current_model=current_model
    )


@app.route("/recordings")
def recordings_list():
    """Return recordings list as HTML fragment."""
    recordings = recorder.get_recordings()
    return render_template("recordings.html", recordings=recordings)


@app.route("/set-model", methods=["POST"])
def set_model():
    """Change the active model."""
    model_name = request.form.get("model")
    if recorder.set_model(model_name):
        return f'<div class="success">Switched to {model_name}</div>'
    return '<div class="error">Failed to switch model</div>', 400


@app.route("/sse")
def sse():
    """Server-Sent Events endpoint for live updates."""

    def event_stream():
        while True:
            try:
                msg = sse_queue.get(timeout=30)
                yield f"event: recording\n"
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield f": heartbeat\n\n"

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


def run_hotkey_listener():
    """Run the F2 hotkey listener in event loop."""
    asyncio.run(recorder.listen_for_hotkey())


def dev_select_device():
    """Dev entry point to test device selector."""
    from rich.console import Console

    console = Console()

    console.print("[bold cyan]Device Selector Test[/bold cyan]\n")
    device = select_keyboard_device()

    if device:
        console.print(f"\n[green]Successfully selected:[/green] {device.name}")
        console.print(f"[dim]Path:[/dim] {device.path}")
        console.print(f"[dim]Physical:[/dim] {device.phys or 'N/A'}")
    else:
        console.print("[yellow]No device selected[/yellow]")


def main():
    global recorder
    import sys

    # Load config and check if setup is needed
    config = load_config()
    if not config.device:
        print("No device configured. Running setup...")
        device = select_keyboard_device()
        if device:
            config.device = DeviceConfig(
                name=device.name, path=device.path, phys=device.phys
            )
            save_config(config)
            print(f"Device saved to config: {device.name}")
        else:
            print("Setup cancelled. Exiting.")
            return

    # Path to whisper-cli binary
    whisper_path = Path(__file__).parent / "bin" / "whisper-cli"
    if not whisper_path.exists():
        print(f"Error: whisper-cli not found at {whisper_path}")
        return

    # Default to tiny model, or use path from command line
    model_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "models" / "ggml-tiny.bin"
    )

    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("\nAvailable models:")
        models_dir = Path(__file__).parent / "models"
        for model in sorted(models_dir.glob("ggml-*.bin")):
            if "for-tests" not in model.name:
                print(f"  {model}")
        return

    # Create recordings directory
    recordings_dir = Path(__file__).parent / "recordings"

    # Initialize recorder
    recorder = WhisperRecorder(whisper_path, model_path, recordings_dir)

    print(f"Whisper Hotkey Recorder [{recorder.model_path.name}]")
    print("Press F2 to start/stop recording")
    print("Web interface: http://localhost:5000\n")

    # Start OpenCode server for transcription cleanup
    start_opencode_server()

    # Start F2 listener in background thread
    listener_thread = threading.Thread(target=run_hotkey_listener, daemon=True)
    listener_thread.start()

    # Run Flask app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        dev_select_device()
    else:
        main()
