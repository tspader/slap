# Whisper Hotkey Recorder

Press your configured hotkey (default F2) to record audio, press it again to transcribe and copy to clipboard.

Transcriptions are automatically cleaned using an LLM to fix technical terms and errors.

## Usage

```bash
# Use default tiny model (fast)
sudo uv run main.py

# Use specific model
sudo uv run main.py ../models/ggml-medium.bin
```

## Models

- `ggml-tiny.bin` - Fastest, less accurate
- `ggml-medium.bin` - Balanced
- `ggml-large-v3-turbo.bin` - Most accurate, slowest

## Requirements

- **Keyboard access**: You must be in the `input` group OR run with sudo
  ```bash
  # Add yourself to the input group (recommended)
  sudo usermod -a -G input $USER
  # Then log out and log back in for changes to take effect
  ```
- wl-clipboard (for Wayland clipboard support)
- opencode (for post-processing transcriptions)

## Configuration

The first time you run the recorder you'll see two TUIs powered by Rich:

1. Keyboard picker – choose which input device should be monitored
2. Trigger key picker – press any key to preview it, then press Enter to confirm

Your choices are saved to `~/.local/share/slap/config.json`. If the config file is missing or does not contain a trigger key, the picker TUIs run again automatically. To redo setup manually, delete that file (or remove the `trigger_key` entry) and relaunch.

## How It Works

The program listens for the configured trigger key (default F2) to start/stop recording. Key presses are not consumed, so other applications will still receive them.
