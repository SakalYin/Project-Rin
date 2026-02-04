# Project Rin

A local AI voice companion with GUI, speech recognition, text-to-speech, and dynamic personality — fully offline, no cloud APIs.

## Features

- **GUI Chat Window** — Dark-themed tkinter interface with session management
- **Voice Input** — Real-time speech recognition via Faster Whisper + Silero VAD
- **Voice Output** — Natural TTS with Kokoro-ONNX, concurrent with LLM streaming
- **Dynamic Personality** — AI personality that adapts based on interactions
- **Session Management** — Continue previous conversations or start fresh
- **Fully Local** — Everything runs on your machine (llama.cpp, Whisper, Kokoro)

## How It Works

```
python run.py
       │
       ├─► Starts llama.cpp server (auto-managed)
       ├─► Initializes STT, TTS, Persona systems
       ├─► Shows session selection dialog
       │
       └─► Chat loop (GUI window):
            Voice/Text → LLM (streaming) → TTS → Speaker
                              ↓
                    Background persona updater
```

1. Launch opens a session dialog — continue previous or start new
2. GUI window opens with chat history loaded
3. Speak or type your message
4. LLM streams a reply while TTS plays completed sentences
5. A background LLM call analyzes the exchange and updates AI personality state
6. Conversation is saved to SQLite database

## Prerequisites

- **Python 3.10+**
- **CUDA** (recommended for GPU acceleration)
- A **.gguf model file** (e.g. Llama 3.1 8B from HuggingFace)
- Working audio input/output devices

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download models

**Kokoro TTS** (required):
```bash
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

**Whisper STT** — auto-downloads on first run (`distil-medium.en` by default)

**LLM** — place your `.gguf` model and update `config.yaml`:
```yaml
llm:
  server:
    model_path: "models/your-model.gguf"
```

### 3. Run

```bash
python run.py
```

## Configuration

All settings in `config.yaml`:

### LLM
| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `openai_compat` | LLM provider |
| `temperature` | `1.2` | Higher = more chaotic |
| `server.model_path` | — | **Required** path to .gguf |
| `server.context_size` | `16384` | Context window |
| `server.gpu_layers` | `-1` | GPU offload (-1 = all) |

### TTS
| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `kokoro` | TTS provider |
| `voice` | `af_heart` | Voice ID |
| `speed` | `1.0` | Playback speed |

### STT
| Key | Default | Description |
|-----|---------|-------------|
| `provider` | `faster_whisper` | STT provider |
| `enabled` | `true` | Enable voice input |
| `model` | `distil-medium.en` | Whisper model |
| `silence_duration` | `1.5` | Seconds of silence to end utterance |

### Persona (Dynamic Personality)
| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable persona system |
| `state_file` | `src/utils/plugins/dynamic_personality/status.txt` | Persona state storage |
| `update_in_background` | `true` | Async persona updates |

## Project Structure

```
Project-Rin/
├── run.py                    # Entry point
├── config.yaml               # All settings
├── src/
│   ├── core/
│   │   ├── app.py            # Main application loop
│   │   ├── config.py         # YAML config loader
│   │   ├── prompt.py         # AI personality prompt
│   │   └── chat_window.py    # GUI window + session dialog
│   ├── orchestrator/
│   │   └── conversation.py   # Turn handler (LLM + TTS + persona)
│   ├── service/
│   │   ├── llm/
│   │   │   ├── engine.py     # LLM engine (provider abstraction)
│   │   │   └── providers/    # openai_compat, etc.
│   │   ├── tts/
│   │   │   ├── engine.py     # TTS engine + audio playback
│   │   │   └── providers/    # kokoro, etc.
│   │   └── stt/
│   │       ├── engine.py     # STT engine + VAD
│   │       └── providers/    # faster_whisper, etc.
│   ├── db/
│   │   └── db_manager.py     # Session-based SQLite storage
│   └── utils/
│       └── plugins/
│           └── dynamic_personality/
│               ├── status_manager.py  # Persona state manager
│               ├── memory_updater.py  # Background LLM updater
│               └── status.txt         # Persona state storage
├── models/                   # LLM + TTS models
└── llama.ccp/                # llama.cpp binaries
```

## Dynamic Personality System

The AI maintains personality state that evolves through interactions:

- **notes** — Facts about the user (name, preferences, hobbies)
- **mood** — AI's current emotional state (from AI's responses)
- **impression** — How the AI perceives the user's current mood/behavior
- **relationship** — The dynamic between AI and user

State is updated via a background LLM call after each turn. The updater analyzes:
- User messages → notes, impression
- AI responses → mood
- Both → relationship

Turn tracking prevents excessive updates (recently updated sections are less likely to change).

To disable: set `persona.enabled: false` in config.yaml

## Debugging

Logs are printed to terminal with timestamps. Key log messages:
- `[INFO] Memory [section] updated: ...` — persona state changed
- `[INFO] Turn complete in X.XXs` — response time
- `[DEBUG] Memory context injected` — persona state sent to LLM

## Troubleshooting

**No GUI appears**
→ Make sure tkinter is installed: `pip install tk`

**No voice input**
→ Check `stt.enabled: true` and microphone permissions

**Persona not updating**
→ Check `persona.enabled: true` and look for "Memory [section] updated" log messages

**Server didn't become healthy**
→ Large models take time to load. Check terminal for loading progress.

**No audio output**
→ Verify audio device: `python -c "import sounddevice; print(sounddevice.query_devices())"`
