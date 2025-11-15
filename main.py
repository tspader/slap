#!/usr/bin/env python3
# Standard library imports
import asyncio
import atexit
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

# Third-party imports
import evdev
import numpy as np
import readchar
import requests
import sounddevice as sd
from evdev import ecodes
from flask import Flask, Response, render_template, request
from pydantic import BaseModel, field_validator
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from scipy import signal
from scipy.io import wavfile


class RichLogHandler(logging.Handler):
    """Custom logging handler that sends to RichLogger."""

    def __init__(self, rich_logger):
        super().__init__()
        self.rich_logger = rich_logger

    def emit(self, record):
        try:
            msg = self.format(record)
            self.rich_logger.server(msg)
        except Exception:
            self.handleError(record)


class RichLogger:
    """Rich-based logger with three panels for different message types."""

    def __init__(self):
        self.console = Console()
        self.opencode_messages = deque(maxlen=50)
        self.transcription_messages = deque(maxlen=50)
        self.server_messages = deque(maxlen=50)
        self.layout = Layout()
        self.live = None  # Will be set by main()
        self.model_name = None  # Will be set by main()
        self.llm_model = None  # Will be set by main()
        self._setup_layout()

    def _setup_layout(self):
        """Setup the three-panel layout."""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
        )
        self.layout["body"].split_row(
            Layout(name="opencode", ratio=1),
            Layout(name="transcription", ratio=1),
            Layout(name="server", ratio=1),
        )

    def _create_panel(self, messages, title, style):
        """Create a panel with messages."""
        if not messages:
            content = "[dim]No messages yet[/dim]"
        else:
            lines = []
            for msg in messages:
                lines.append(msg)
            content = "\n".join(lines[-10:])  # Show last 10 messages

        return Panel(content, title=title, border_style=style, padding=(0, 1))

    def update_display(self):
        """Update the live display."""
        title = "[bold cyan]slap[/bold cyan]"
        if self.model_name:
            title += f" [dim]{self.model_name}[/dim]"
        if self.llm_model:
            title += f" [dim]{self.llm_model}[/dim]"

        header = Panel(
            title,
            border_style="cyan",
        )

        opencode_panel = self._create_panel(self.opencode_messages, "opencode", "green")
        transcription_panel = self._create_panel(
            self.transcription_messages, "whisper", "blue"
        )
        server_panel = self._create_panel(self.server_messages, "server", "yellow")

        self.layout["header"].update(header)
        self.layout["opencode"].update(opencode_panel)
        self.layout["transcription"].update(transcription_panel)
        self.layout["server"].update(server_panel)

    def opencode(self, message):
        """Add OpenCode message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.opencode_messages.append(f"[dim]{timestamp}[/dim] {message}")
        self._refresh()

    def transcription(self, message):
        """Add transcription message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.transcription_messages.append(f"[dim]{timestamp}[/dim] {message}")
        self._refresh()

    def server(self, message):
        """Add server message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.server_messages.append(f"[dim]{timestamp}[/dim] {message}")
        self._refresh()

    def _refresh(self):
        """Update the display if live object is set."""
        self.update_display()
        if self.live:
            self.live.update(self.layout)


class DeviceConfig(BaseModel):
    name: str
    path: str
    phys: Optional[str] = None
    trigger_key: Optional[int] = None

    @field_validator("trigger_key", mode="before")
    @classmethod
    def normalize_trigger_key(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            if value.startswith("KEY_"):
                return getattr(ecodes, value, ecodes.KEY_F2)
            try:
                return int(value)
            except ValueError:
                return ecodes.KEY_F2
        return value


class AppConfig(BaseModel):
    device: Optional[DeviceConfig] = None
    whisper_model: Optional[str] = None
    llm_model: Optional[str] = None


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
            logger.server(f"Error loading config: {e}")
    return AppConfig()


def save_config(config: AppConfig):
    """Save config to file."""
    config_path = get_config_path()
    try:
        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)
    except Exception as e:
        logger.server(f"Error saving config: {e}")


def get_available_whisper_models():
    """Get list of available whisper models from models directory."""
    models_dir = Path(__file__).parent / "models"
    if not models_dir.exists():
        return []

    models = [
        model.name
        for model in sorted(models_dir.glob("ggml-*.bin"))
        if "for-tests" not in model.name
    ]
    return models


def get_available_llm_models():
    """Get list of available LLM models from opencode."""
    try:
        result = subprocess.run(
            ["opencode", "models"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return {}

        # Parse output format: "provider/model"
        models = {}
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line or "/" not in line:
                continue
            # Format is like: anthropic/claude-haiku-4-5
            models[line] = line

        return models
    except Exception as e:
        logger.server(f"Failed to get LLM models: {e}")
        return {}


def get_key_name(key_code):
    """Get human-readable name for key code."""
    key_names = {
        ecodes.KEY_F1: "F1",
        ecodes.KEY_F2: "F2",
        ecodes.KEY_F3: "F3",
        ecodes.KEY_F4: "F4",
        ecodes.KEY_F5: "F5",
        ecodes.KEY_F6: "F6",
        ecodes.KEY_F7: "F7",
        ecodes.KEY_F8: "F8",
        ecodes.KEY_F9: "F9",
        ecodes.KEY_F10: "F10",
        ecodes.KEY_F11: "F11",
        ecodes.KEY_F12: "F12",
        ecodes.KEY_A: "A",
        ecodes.KEY_B: "B",
        ecodes.KEY_C: "C",
        ecodes.KEY_D: "D",
        ecodes.KEY_E: "E",
        ecodes.KEY_F: "F",
        ecodes.KEY_G: "G",
        ecodes.KEY_H: "H",
        ecodes.KEY_I: "I",
        ecodes.KEY_J: "J",
        ecodes.KEY_K: "K",
        ecodes.KEY_L: "L",
        ecodes.KEY_M: "M",
        ecodes.KEY_N: "N",
        ecodes.KEY_O: "O",
        ecodes.KEY_P: "P",
        ecodes.KEY_Q: "Q",
        ecodes.KEY_R: "R",
        ecodes.KEY_S: "S",
        ecodes.KEY_T: "T",
        ecodes.KEY_U: "U",
        ecodes.KEY_V: "V",
        ecodes.KEY_W: "W",
        ecodes.KEY_X: "X",
        ecodes.KEY_Y: "Y",
        ecodes.KEY_Z: "Z",
        ecodes.KEY_1: "1",
        ecodes.KEY_2: "2",
        ecodes.KEY_3: "3",
        ecodes.KEY_4: "4",
        ecodes.KEY_5: "5",
        ecodes.KEY_6: "6",
        ecodes.KEY_7: "7",
        ecodes.KEY_8: "8",
        ecodes.KEY_9: "9",
        ecodes.KEY_0: "0",
        ecodes.KEY_SPACE: "Space",
        ecodes.KEY_ENTER: "Enter",
        ecodes.KEY_ESC: "Escape",
        ecodes.KEY_TAB: "Tab",
        ecodes.KEY_BACKSPACE: "Backspace",
        ecodes.KEY_LEFTCTRL: "Left Ctrl",
        ecodes.KEY_RIGHTCTRL: "Right Ctrl",
        ecodes.KEY_LEFTALT: "Left Alt",
        ecodes.KEY_RIGHTALT: "Right Alt",
        ecodes.KEY_LEFTSHIFT: "Left Shift",
        ecodes.KEY_RIGHTSHIFT: "Right Shift",
        ecodes.KEY_LEFTMETA: "Left Meta",
        ecodes.KEY_RIGHTMETA: "Right Meta",
    }
    return key_names.get(key_code, f"Key_{key_code}")


def select_trigger_key(device):
    """Interactive TUI for selecting trigger key."""
    console = Console()
    last_key_code = None
    confirmed_key = None
    error_message = None

    def build_display():
        """Build the display content."""

        title = Panel(
            "[bold cyan]Select Trigger Key[/bold cyan]\n"
            "Press keys to preview, Enter to confirm, Esc to cancel",
            border_style="cyan",
        )

        if last_key_code is not None:
            key_name = get_key_name(last_key_code)
            status = Panel(
                f"[bold green]Selected:[/bold green] {key_name} ({last_key_code})\n"
                "Press Enter to confirm or another key to change",
                border_style="green",
            )
        else:
            status = Panel(
                "[dim]Press any key to select it as the trigger...[/dim]",
                border_style="dim",
            )

        return Group(title, "", status)

    with Live(build_display(), console=console, refresh_per_second=10) as live:
        try:
            for event in device.read_loop():
                if event.type != ecodes.EV_KEY or event.value != 1:
                    continue

                key_code = event.code

                if key_code == ecodes.KEY_ESC:
                    break

                if key_code in (ecodes.KEY_ENTER, ecodes.KEY_KPENTER):
                    if last_key_code is not None:
                        confirmed_key = last_key_code
                        break
                    else:
                        continue

                last_key_code = key_code
                live.update(build_display())
        except KeyboardInterrupt:
            pass
        except Exception as e:
            error_message = str(e)

    if confirmed_key is not None:
        key_name = get_key_name(confirmed_key)
        console.print(f"[green]✓[/green] Selected trigger key: [bold]{key_name}[/bold]")
        return confirmed_key

    if error_message:
        console.print(f"[red]Error reading keys: {error_message}[/red]")
    else:
        console.print("[yellow]Key selection cancelled[/yellow]")

    return None


def configure_models(config: AppConfig, available_llm_models: dict):
    """Ensure the config has whisper and LLM models configured."""
    console = Console()
    needs_save = False

    # Check Whisper model
    whisper_models = get_available_whisper_models()
    if not whisper_models:
        console.print("[red]No Whisper models found in models directory![/red]")
        return False

    # Validate or select whisper model
    if not config.whisper_model or config.whisper_model not in whisper_models:
        if config.whisper_model:
            console.print(
                f"[yellow]Configured Whisper model '{config.whisper_model}' not found[/yellow]"
            )

        selected = select_whisper_model(whisper_models)
        if not selected:
            console.print("[red]Whisper model selection cancelled[/red]")
            return False

        config.whisper_model = selected
        needs_save = True

    # Check LLM model
    if not available_llm_models:
        console.print("[red]No LLM models available from OpenCode[/red]")
        return False

    # Validate or select LLM model
    if not config.llm_model or config.llm_model not in available_llm_models:
        if config.llm_model:
            console.print(
                f"[yellow]Configured LLM model '{config.llm_model}' not found[/yellow]"
            )

        selected = select_llm_model(available_llm_models)
        if not selected:
            console.print("[red]LLM model selection cancelled[/red]")
            return False

        config.llm_model = selected
        needs_save = True

    if needs_save:
        save_config(config)
        console.print(f"[green]✓[/green] Whisper model: {config.whisper_model}")
        console.print(f"[green]✓[/green] LLM model: {config.llm_model}")

    return True


def configure_device_and_trigger(
    config: AppConfig, prefer_existing_device: bool = False
):
    """Ensure the config has a device and trigger key."""
    device = None

    if prefer_existing_device and config.device:
        devices = get_accessible_keyboard_devices()
        for d in devices:
            if d.path == config.device.path:
                device = d
                logger.server(f"Using configured device: {config.device.name}")
                break
        else:
            logger.server(f"Could not access configured device at {config.device.path}")
            device = None

    if device is None:
        device = select_keyboard_device()
        if not device:
            logger.server("Device selection cancelled.")
            return False

    device_info = {
        "name": device.name,
        "path": device.path,
        "phys": device.phys,
    }

    try:
        trigger_key = select_trigger_key(device)
    finally:
        try:
            device.close()
        except Exception:
            pass

    if trigger_key is None:
        logger.server("Trigger key selection cancelled.")
        return False

    config.device = DeviceConfig(
        name=device_info["name"],
        path=device_info["path"],
        phys=device_info["phys"],
        trigger_key=trigger_key,
    )
    save_config(config)
    key_name = get_key_name(trigger_key)
    logger.server(f"Device saved to config: {device.name}")
    logger.server(f"Trigger key saved: {key_name}")
    return True


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

    logger.server("Starting OpenCode server...")
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
            logger.server("OpenCode server started!")
            break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)
    else:
        logger.server("Warning: OpenCode server may not be ready")

    # Create a session
    try:
        response = requests.post(
            f"{OPENCODE_URL}/session",
            json={"title": "Whisper Transcription Cleanup"},
            timeout=5,
        )
        if response.status_code == 200:
            opencode_session_id = response.json().get("id")
            logger.server(f"Created OpenCode session: {opencode_session_id}")
        else:
            logger.server(f"Failed to create session: {response.status_code}")
    except Exception as e:
        logger.server(f"Error creating session: {e}")


def stop_opencode_server():
    """Stop the opencode server."""
    global opencode_process
    if opencode_process:
        logger.server("Stopping OpenCode server...")
        opencode_process.terminate()
        try:
            opencode_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            opencode_process.kill()


def clean_transcription(raw_text, llm_model=None):
    """Clean up transcription using OpenCode API."""
    global opencode_session_id

    if not opencode_session_id:
        logger.opencode("No OpenCode session, returning raw text")
        return raw_text

    # Default to haiku if no model specified
    if not llm_model:
        llm_model = "anthropic/claude-haiku-4-5"

    # Parse provider and model from format "provider/model"
    if "/" in llm_model:
        provider_id, model_id = llm_model.split("/", 1)
    else:
        provider_id = "anthropic"
        model_id = llm_model

    prompt = f"""Clean this speech-to-text transcription by fixing only transcription errors:
    - Correct mistranscribed technical terms (programming terms, commands, etc.)
    - Add appropriate punctuation
    - Fix awkward phrasing from spoken language
    - Preserve profanity and semantic meaning

    Output ONLY the cleaned text, nothing else.

    Exception: If the transcription contains "Achilles", treat text between "Achilles" and "Tortoise" as meta-instructions. Clean those instructions for transcription errors first, then apply them while removing the markers and instructions from the output.

    Raw transcription, inside single quotes:
    '{raw_text}'
    """

    try:
        response = requests.post(
            f"{OPENCODE_URL}/session/{opencode_session_id}/message",
            json={
                "parts": [{"type": "text", "text": prompt}],
                "model": {"providerID": provider_id, "modelID": model_id},
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
                logger.opencode(f"Cleaned: {raw_text} -> {cleaned_text.strip()}")
                return cleaned_text.strip()
        else:
            logger.opencode(f"OpenCode API error: {response.status_code}")
    except Exception as e:
        logger.opencode(f"Error cleaning transcription: {e}")

    return raw_text


# Register cleanup handler
atexit.register(stop_opencode_server)


def get_accessible_keyboard_devices():
    """Get list of accessible keyboard devices, showing usermod message if needed."""
    console = Console()
    device_paths = evdev.list_devices()

    if not device_paths:
        console.print("[red]No input devices accessible![/red]")
        console.print("\nYou need to either:")
        console.print("1. Add yourself to the 'input' group:")
        console.print("   sudo usermod -a -G input $USER")
        console.print("   (then log out and log back in)")
        console.print("2. Or run with sudo:")
        console.print("   sudo uv run main.py")
        return []

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
        except Exception:
            pass

    return devices


def select_whisper_model(available_models):
    """Interactive TUI for selecting a Whisper model."""
    console = Console()

    if not available_models:
        console.print("[red]No Whisper models found in models directory![/red]")
        return None

    selected_idx = 0

    def build_display(idx):
        """Build the display content."""

        title = Panel(
            "[bold cyan]Select Whisper Model[/bold cyan]\n"
            "Use ↑/↓ or j/k to navigate, Enter to select, q to quit",
            border_style="cyan",
        )

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("", width=3)
        table.add_column("Model", style="cyan")

        for i, model in enumerate(available_models):
            cursor = "[bold green]→[/bold green]" if i == idx else " "
            name_style = "bold green" if i == idx else "cyan"
            table.add_row(cursor, f"[{name_style}]{model}[/{name_style}]")

        return Group(title, "", table)

    with Live(
        build_display(selected_idx), console=console, refresh_per_second=10
    ) as live:
        while True:
            key = readchar.readkey()

            if key == readchar.key.UP or key == "k":
                selected_idx = (selected_idx - 1) % len(available_models)
                live.update(build_display(selected_idx))
            elif key == readchar.key.DOWN or key == "j":
                selected_idx = (selected_idx + 1) % len(available_models)
                live.update(build_display(selected_idx))
            elif key == readchar.key.ENTER or key == "\r" or key == "\n":
                selected = available_models[selected_idx]
                live.stop()
                console.print(f"[green]✓[/green] Selected: [bold]{selected}[/bold]")
                return selected
            elif key == "q" or key == readchar.key.ESC:
                live.stop()
                console.print("[yellow]Selection cancelled[/yellow]")
                return None


def select_llm_model(available_models):
    """Interactive TUI for selecting an LLM model with fuzzy search."""
    console = Console()

    if not available_models:
        console.print("[red]No LLM models available![/red]")
        return None

    model_list = sorted(available_models.keys())
    filtered_models = model_list.copy()
    selected_idx = 0
    search_query = ""

    def build_display(idx, query, filtered):
        """Build the display content."""

        title_text = "[bold cyan]Select LLM Model[/bold cyan]\n"
        if query:
            title_text += f"Filter: [yellow]{query}[/yellow] | "
        title_text += "Type to filter, ↑/↓ to navigate, Enter to select, Esc to quit"

        title = Panel(title_text, border_style="cyan")

        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("", width=3)
        table.add_column("Model", style="cyan")

        display_count = min(15, len(filtered))
        for i in range(display_count):
            cursor = "[bold green]→[/bold green]" if i == idx else " "
            name_style = "bold green" if i == idx else "cyan"
            table.add_row(cursor, f"[{name_style}]{filtered[i]}[/{name_style}]")

        if len(filtered) > display_count:
            table.add_row("", f"[dim]...and {len(filtered) - display_count} more[/dim]")

        return Group(title, "", table)

    with Live(
        build_display(selected_idx, search_query, filtered_models),
        console=console,
        refresh_per_second=10,
    ) as live:
        while True:
            key = readchar.readkey()

            if key == "\x1b" or (
                hasattr(readchar.key, "ESC") and key == readchar.key.ESC
            ):
                live.stop()
                console.print("[yellow]Selection cancelled[/yellow]")
                return None
            elif key == readchar.key.UP:
                selected_idx = max(0, selected_idx - 1)
                live.update(build_display(selected_idx, search_query, filtered_models))
            elif key == readchar.key.DOWN:
                selected_idx = min(len(filtered_models) - 1, selected_idx + 1)
                live.update(build_display(selected_idx, search_query, filtered_models))
            elif key == readchar.key.ENTER or key == "\r" or key == "\n":
                if filtered_models:
                    selected = filtered_models[selected_idx]
                    live.stop()
                    console.print(f"[green]✓[/green] Selected: [bold]{selected}[/bold]")
                    return selected
            elif key == readchar.key.BACKSPACE or key == "\x7f":
                if search_query:
                    search_query = search_query[:-1]
                    filtered_models = [
                        m for m in model_list if search_query.lower() in m.lower()
                    ]
                    selected_idx = 0
                    live.update(
                        build_display(selected_idx, search_query, filtered_models)
                    )
            elif len(key) == 1 and key.isprintable():
                search_query += key
                filtered_models = [
                    m for m in model_list if search_query.lower() in m.lower()
                ]
                selected_idx = 0
                live.update(build_display(selected_idx, search_query, filtered_models))


def select_keyboard_device():
    """Interactive TUI for selecting a keyboard device."""
    console = Console()

    devices = get_accessible_keyboard_devices()
    if not devices:
        console.print("[red]No keyboard devices found![/red]")
        return None

    # Sort devices by name for consistent ordering
    devices.sort(key=lambda d: d.name)

    selected_idx = 0

    def build_display(idx):
        """Build the display content."""

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
    def __init__(self, whisper_path, model_path, recordings_dir, llm_model=None):
        self.whisper_path = Path(whisper_path)
        self.model_path = Path(model_path)
        self.recordings_dir = Path(recordings_dir)
        self.recordings_dir.mkdir(exist_ok=True)
        self.is_recording = False
        self.audio_data = []
        self.stream = None
        self.available_models = []
        self.llm_model = llm_model or "anthropic/claude-haiku-4-5"

        # Get device's native sample rate and set whisper target
        device_info = sd.query_devices(kind="input")
        if isinstance(device_info, dict):
            self.recording_sample_rate = int(
                device_info.get("default_samplerate", 44100)
            )
        else:
            self.recording_sample_rate = int(device_info.default_samplerate)
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
            logger.transcription(f"Audio status: {status}")
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        """Start audio recording."""
        if self.is_recording:
            return

        logger.transcription("Recording started... (press hotkey to stop)")
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

        logger.transcription("Recording stopped. Processing...")
        self.is_recording = False

        # Stop the stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Check if we have audio data
        if not self.audio_data:
            logger.transcription("No audio recorded!")
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
        audio_int16 = np.array(audio_array * 32767, dtype=np.int16)

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
                    logger.transcription(f"Raw transcription: {text}")
                    self._save_transcription(text)
                else:
                    logger.transcription("No speech detected.")
            else:
                logger.transcription(f"Whisper failed (code {result.returncode})")
                if result.stderr:
                    logger.transcription(f"Error: {result.stderr[:200]}")

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

    def _save_transcription(self, raw_text):
        """Save transcription to timestamped file and notify SSE clients."""
        # Clean up the transcription using OpenCode
        cleaned_text = clean_transcription(raw_text, self.llm_model)

        timestamp = datetime.now().isoformat().replace(":", "-").replace(".", "-")
        filename = f"{timestamp}.txt"
        filepath = self.recordings_dir / filename

        with open(filepath, "w") as f:
            f.write(cleaned_text)

        logger.transcription(f"Saved to {filename}")

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
            except Exception:
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
                logger.transcription("Copied to clipboard!")
            except subprocess.CalledProcessError as e:
                logger.transcription(f"Failed to copy to clipboard: {e.stderr.strip()}")
            except subprocess.TimeoutExpired:
                logger.transcription("Clipboard operation timed out")
            except FileNotFoundError:
                logger.transcription("wl-copy not found. Install wl-clipboard package.")
        else:
            # Not running under sudo, use wl-copy directly
            try:
                subprocess.run(
                    ["wl-copy"], input=text, check=True, capture_output=True, text=True
                )
                logger.transcription("Copied to clipboard!")
            except subprocess.CalledProcessError as e:
                logger.transcription(f"Failed to copy to clipboard: {e.stderr.strip()}")
            except FileNotFoundError:
                logger.transcription("wl-copy not found. Install wl-clipboard package.")

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
            logger.model_name = model_name
            logger.server(f"Switched to model: {model_name}")
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
        devices = get_accessible_keyboard_devices()

        if not devices:
            return None

        # If we have a configured device, try to find it first
        if config.device:
            # Try to match by path first (most reliable)
            for device in devices:
                if device.path == config.device.path:
                    logger.server(f"Found configured device by path: {device.name}")
                    return device

            # Try to match by name and physical location
            for device in devices:
                if (
                    device.name == config.device.name
                    and device.phys == config.device.phys
                ):
                    logger.server(
                        f"Found configured device by name/phys: {device.name}"
                    )
                    return device

            # Try to match by name only (less reliable)
            for device in devices:
                if device.name == config.device.name:
                    logger.server(f"Found configured device by name: {device.name}")
                    return device

            logger.server(f"Configured device not found: {config.device.name}")

        # Fallback: find device with F2 key
        for device in devices:
            caps = device.capabilities(verbose=False)
            if ecodes.EV_KEY in caps and ecodes.KEY_F2 in caps[ecodes.EV_KEY]:
                logger.server(f"Using fallback device with F2 key: {device.name}")
                return device

        # Final fallback: find any keyboard device
        for device in devices:
            caps = device.capabilities(verbose=False)
            if ecodes.EV_KEY in caps:
                # Check if it has typical keyboard keys
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_A in keys or ecodes.KEY_SPACE in keys:
                    logger.server(f"Using fallback keyboard device: {device.name}")
                    return device

        return None

    async def listen_for_hotkey(self):
        """Async loop to listen for trigger key press."""
        config = load_config()
        device = self.find_keyboard_device()
        if not device:
            logger.server("Error: No keyboard device found!")
            return

        logger.server(f"Using input device: {device.name}")

        # Get the trigger key from config
        trigger_key_code = (
            config.device.trigger_key
            if config.device and config.device.trigger_key is not None
            else ecodes.KEY_F2
        )
        key_name = get_key_name(trigger_key_code)
        logger.server(f"Listening for trigger key: {key_name}")

        try:
            async for event in device.async_read_loop():
                # Check for trigger key press (key down event)
                if event.type == ecodes.EV_KEY and event.code == trigger_key_code:
                    if event.value == 1:  # Key down (1 = press, 0 = release, 2 = hold)
                        self.toggle_recording()
        except Exception as e:
            logger.server(f"Error reading events: {e}")


# Global logger instance
logger = RichLogger()

# Global recorder instance
recorder = None


# Flask routes
@app.route("/")
def index():
    """Render main page."""
    if recorder is None:
        return "Recorder not initialized", 503
    recordings = recorder.get_recordings()
    models = [model.name for model in recorder.available_models]
    current_model = recorder.model_path.name
    return render_template(
        "index.html", recordings=recordings, models=models, current_model=current_model
    )


@app.route("/recordings")
def recordings_list():
    """Return recordings list as HTML fragment."""
    if recorder is None:
        return "Recorder not initialized", 503
    recordings = recorder.get_recordings()
    return render_template("recordings.html", recordings=recordings)


@app.route("/set-model", methods=["POST"])
def set_model():
    """Change the active model."""
    if recorder is None:
        return "Recorder not initialized", 503
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
                yield "event: recording\n"
                yield f"data: {json.dumps(msg)}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"

    response = Response(event_stream(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


def run_hotkey_listener():
    """Run the hotkey listener in event loop."""
    if recorder is None:
        logger.server("Error: Recorder not initialized!")
        return
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

    console = Console()

    # Load config and ensure setup is complete
    config = load_config()

    if not config.device:
        console.print("No device configured. Running setup...")
        if not configure_device_and_trigger(config):
            console.print("Setup cancelled. Exiting.")
            return
    elif config.device.trigger_key is None:
        console.print("Trigger key not configured. Running trigger key setup...")
        if not configure_device_and_trigger(config, prefer_existing_device=True):
            console.print("Trigger key setup cancelled. Exiting.")
            return

    # Get available LLM models once at startup
    console.print("[dim]Loading available LLM models...[/dim]")
    available_llm_models = get_available_llm_models()

    # Configure models if needed
    if not configure_models(config, available_llm_models):
        console.print("[red]Model configuration failed. Exiting.[/red]")
        return

    # Path to whisper-cli binary
    whisper_path = Path(__file__).parent / "bin" / "whisper-cli"
    if not whisper_path.exists():
        console.print(f"[red]Error: whisper-cli not found at {whisper_path}[/red]")
        return

    # Get whisper model path from config
    model_path = Path(__file__).parent / "models" / config.whisper_model

    if not model_path.exists():
        console.print(f"[red]Error: model not found at {model_path}[/red]")
        console.print("\n[yellow]Available models:[/yellow]")
        models_dir = Path(__file__).parent / "models"
        for model in sorted(models_dir.glob("ggml-*.bin")):
            if "for-tests" not in model.name:
                console.print(f"  {model}")
        return

    # Create recordings directory
    recordings_dir = Path(__file__).parent / "recordings"

    # Initialize recorder with configured LLM model
    recorder = WhisperRecorder(
        whisper_path, model_path, recordings_dir, llm_model=config.llm_model
    )

    trigger_key_code = (
        config.device.trigger_key
        if config.device and config.device.trigger_key is not None
        else ecodes.KEY_F2
    )
    trigger_key_name = get_key_name(trigger_key_code)

    logger.model_name = recorder.model_path.name
    logger.llm_model = (
        config.llm_model.split("/")[1] if "/" in config.llm_model else config.llm_model
    )
    logger.server(f"Whisper Hotkey Recorder [{recorder.model_path.name}]")
    logger.server(f"Press {trigger_key_name} to start/stop recording")
    logger.server("Web interface: http://127.0.0.1:5000\n")

    # Start live display
    logger.update_display()

    with Live(
        logger.layout, console=logger.console, refresh_per_second=2, screen=False
    ) as live:
        logger.live = live  # Set live reference for updates

        # Configure Flask logging to use RichLogger
        flask_handler = RichLogHandler(logger)
        flask_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure Flask app logger
        app.logger.handlers.clear()
        app.logger.addHandler(flask_handler)
        app.logger.setLevel(logging.INFO)

        # Configure Werkzeug (Flask development server) logger
        werkzeug_logger = logging.getLogger("werkzeug")
        werkzeug_logger.handlers.clear()
        werkzeug_logger.addHandler(flask_handler)
        werkzeug_logger.setLevel(logging.INFO)

        # Start OpenCode server in background (doesn't block)
        opencode_thread = threading.Thread(target=start_opencode_server, daemon=True)
        opencode_thread.start()

        # Start hotkey listener in background thread
        listener_thread = threading.Thread(target=run_hotkey_listener, daemon=True)
        listener_thread.start()

        # Run Flask app (use 127.0.0.1 to avoid DNS timeout on 0.0.0.0)
        app.run(
            host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        dev_select_device()
    else:
        main()
