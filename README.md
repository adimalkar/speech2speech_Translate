# Real-Time Speech-to-Speech Translation (S2ST)

This project implements a low-latency, concurrent Speech-to-Speech Translation system using state-of-the-art AI models. It is designed to capture audio from a microphone, transcribe it, translate it, and synthesize speech in a target language in real-time.

## Features

- **Real-time Audio Capture**: Low-latency audio streaming with Voice Activity Detection (VAD).
- **Concurrent Pipeline**: Three-stage pipeline (STT -> MT -> TTS) running in parallel threads.
- **State-of-the-Art Models**:
  - **STT**: `faster-whisper` (OpenAI Whisper) for accurate transcription.
  - **MT**: `Helsinki-NLP/MarianMT` (Transformers) for neural machine translation.
  - **TTS**: `facebook/mms-tts` (Meta Massively Multilingual Speech) for high-quality speech synthesis.
- **Modular Design**: Easy to swap out components or add new languages.

## Requirements

### System Dependencies
- Python 3.8+
- PortAudio (required for PyAudio)
  - **Ubuntu/Debian**: `sudo apt-get install portaudio19-dev`
  - **macOS**: `brew install portaudio`

### Python Libraries
Install the required libraries using pip:

```bash
pip install numpy torch transformers faster-whisper sounddevice pyaudio librosa loguru sacremoses sentencepiece
```

**Note**: For GPU acceleration (highly recommended), ensure you have the appropriate PyTorch version installed for your CUDA version. Visit [pytorch.org](https://pytorch.org/) for instructions.

## Project Structure

```
s2st-project/
├── config/             # Configuration settings
├── src/                # Source code
│   ├── audio/          # Audio capture and VAD
│   ├── mt/             # Machine Translation components
│   ├── stt/            # Speech-to-Text components
│   ├── tts/            # Text-to-Speech components
│   ├── pipeline/       # Pipeline orchestration
│   └── utils/          # Utilities (logging, etc.)
├── main.py             # Entry point
└── README.md           # This file
```

## Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd s2st-project
    ```

2.  **Run the application:**
    ```bash
    python main.py --source en --target es
    ```
    - `--source`: Source language code (e.g., `en` for English, `es` for Spanish).
    - `--target`: Target language code (e.g., `es` for Spanish, `fr` for French, `de` for German).

    **Example:** Translate English to French
    ```bash
    python main.py --source en --target fr
    ```

3.  **Interact:**
    - Speak into your microphone.
    - The system will print the transcription and translation to the console.
    - The translated speech will be played back through your speakers.
    - Press `Ctrl+C` to exit.

## Configuration

You can adjust audio settings (sample rate, chunk size, etc.) in `config/config.py`.

## Troubleshooting

-   **"PortAudio not found"**: Ensure you have installed the system dependencies for PyAudio.
-   **CUDA/GPU errors**: Make sure your PyTorch installation matches your CUDA driver version. Run `python -c "import torch; print(torch.cuda.is_available())"` to verify.
-   **Model loading errors**: The first run will download models from Hugging Face. Ensure you have an internet connection.

## License

[MIT License](LICENSE)
