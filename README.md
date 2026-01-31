# Project Rin

A local AI voice agent that generates chaotic, expressive replies and speaks them out loud — fully offline, no cloud APIs.

## How It Works

```
python run.py
       │
       ├─► Starts llama.cpp server (auto-managed subprocess)
       ├─► Waits for server health check
       │
       └─► Chat loop:
            User input → LLM (streaming) → Sentence splitter → TTS → Speaker
                                                 ↑ concurrent ↑
```

1. `run.py` automatically launches the llama.cpp server with your model
2. You type a message in the terminal
3. The LLM streams a reply token-by-token
4. As complete sentences form, they're immediately sent to the TTS engine
5. Kokoro-ONNX synthesizes speech and plays it through your speakers — **while the LLM is still generating**
6. The full conversation is saved to a local SQLite database
7. On exit, the server subprocess is cleanly terminated

## Prerequisites

- **Python 3.10+**
- A **.gguf model file** (e.g. from HuggingFace)
- A working audio output device

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Kokoro TTS models

Download these two files and place them in the `models/` directory:

```bash
# Option A: direct download (wget / curl)
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

```bash
# Option B: manual download
# Go to https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
# Download kokoro-v1.0.onnx (~310 MB) and voices-v1.0.bin
# Place both files in the models/ directory
```

> There are also `kokoro-v1.0.fp16.onnx` (~169 MB) and `kokoro-v1.0.int8.onnx` (~88 MB) quantized variants available at the same URL. If you use one of these, update `tts.model_path` in `config.yaml`.

After downloading, `models/` should contain:
```
models/
├── kokoro-v1.0.onnx
└── voices-v1.0.bin
```

### 3. Configure your model path

Edit `config.yaml` and set `server.model_path` to your `.gguf` file:

```yaml
server:
  model_path: "path/to/your/model.gguf"   # REQUIRED
```

### 4. Run

```bash
python run.py
```

This will automatically:
- Start the llama.cpp server with your model
- Wait for it to finish loading
- Launch the interactive chat

## Configuration

All settings live in `config.yaml` at the project root.

### Server (llama.cpp subprocess)

| Key              | Default                        | Description                                 |
|------------------|--------------------------------|---------------------------------------------|
| `enabled`        | `true`                         | Set `false` to manage the server yourself   |
| `executable`     | `llama.ccp/llama-server.exe`   | Path to the llama-server binary             |
| `model_path`     | —                              | **Required** — path to your `.gguf` model   |
| `context_size`   | `2048`                         | Total context window (`-c`)                 |
| `n_predict`      | `512`                          | Max completion tokens per reply (`-n`)      |
| `gpu_layers`     | `-1`                           | GPU offload (`-ngl`, -1 = all layers)       |
| `port`           | `8080`                         | Server listen port                          |
| `host`           | `127.0.0.1`                    | Server listen address                       |
| `extra_args`     | `[]`                           | Any additional CLI flags                    |

### LLM API

| Key              | Default          | Description                                  |
|------------------|------------------|----------------------------------------------|
| `temperature`    | `1.1`            | Higher = more chaotic replies                |
| `timeout`        | `30.0`           | Seconds before API request fails             |
| `max_retries`    | `3`              | Retries on empty LLM responses               |
| `max_tokens`     | *(unset)*        | Leave unset to use `server.n_predict`        |

> **Note:** `llm.base_url` is automatically derived from `server.host` and `server.port` when the server is enabled.

### TTS

| Key      | Default      | Description                |
|----------|--------------|----------------------------|
| `voice`  | `af_heart`   | Kokoro voice ID            |
| `speed`  | `1.1`        | Playback speed multiplier  |

### Available Kokoro Voices

Some voice options for the `tts.voice` field:

- `af_heart` — American female, warm (default)
- `af_bella` — American female
- `af_sarah` — American female
- `af_nicole` — American female
- `bf_emma` — British female
- `bf_isabella` — British female

### Running the server manually

If you prefer to manage the llama.cpp server yourself:

1. Set `server.enabled: false` in `config.yaml`
2. Set `llm.base_url` to point at your running server
3. Start the server however you like, then run `python run.py`

## Project Structure

```
Project-Rin/
├── llama.ccp/               # Pre-built llama.cpp binaries
├── src/
│   ├── __init__.py
│   ├── core/                # Core infrastructure
│   │   ├── config.py        #   YAML config loader with typed dataclasses
│   │   ├── prompt.py        #   AI personality system prompt
│   │   └── app.py           #   Main loop orchestrating all components
│   ├── llm/                 # LLM interaction
│   │   ├── server.py        #   llama.cpp subprocess lifecycle manager
│   │   └── client.py        #   Async OpenAI-compatible streaming client
│   ├── tts/                 # Text-to-speech
│   │   └── engine.py        #   Kokoro TTS synthesis + audio playback
│   └── db/                  # Database layer
│       └── chat.py          #   Async SQLite chat history persistence
├── models/                  # TTS model files (download separately)
├── data/                    # SQLite database (auto-created on first run)
├── config.yaml              # All tuneable settings
├── requirements.txt
├── run.py                   # Entry point
└── README.md
```

## Debugging

A log file `rin_debug.log` is created in the project root with:
- llama-server subprocess output
- LLM generation speed (tokens/sec) and finish reason
- TTS synthesis time vs audio duration
- Full text of every spoken sentence
- Any errors from LLM or TTS

## Troubleshooting

**"server.model_path is required"**
→ Edit `config.yaml` and set `server.model_path` to your `.gguf` model file.

**"llama-server executable not found"**
→ Check that `server.executable` in `config.yaml` points to a valid `llama-server` binary.

**Server didn't become healthy within 120s**
→ Your model may be very large. Check `rin_debug.log` for `[llama-server]` lines to see loading progress.

**No audio output**
→ Check that `sounddevice` can see your audio device: `python -c "import sounddevice; print(sounddevice.query_devices())"`

**"kokoro-onnx is not installed"**
→ Run `pip install kokoro-onnx onnxruntime`

**TTS model not found**
→ Make sure `kokoro-v1.0.onnx` and `voices-v1.0.bin` are in the `models/` directory.

**Response truncated warning in logs**
→ Increase `server.n_predict` in `config.yaml`.
