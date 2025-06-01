# Voice Assistant

A cross-platform AI voice assistant that uses speech recognition, camera, and desktop screenshot features, powered by OpenAI and ElevenLabs APIs.

## Requirements
- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/installation/)
- API Keys:
  - `OPENAI_API_KEY` (or `OPENROUTER_API_KEY` if using OpenRouter)
  - `ELEVENLABS_API_KEY` (for TTS)

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/alloy-voice-assistant-main.git
   cd alloy-voice-assistant-main
   $ brew install portaudio
   ```

Create a virtual environment, update pip, and install the required packages:

```
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -U pip
$ pip install -r requirements.txt
```

Run the assistant:

```
$ python3 assistant.py
```
