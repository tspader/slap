# Whisper Hotkey Recorder

Press F2 to record audio, press F2 again to transcribe and copy to clipboard.

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

## How It Works

The program listens for F2 key presses to trigger recording. F2 presses will still be seen by other applications (the key is not intercepted/consumed).
