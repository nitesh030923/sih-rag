# Audio Transcription Setup

Audio transcription is currently not working because it requires additional dependencies. Here's how to enable it:

## Requirements

1. **FFmpeg** - Required for audio processing
2. **PyTorch** - Deep learning framework for Whisper
3. **Whisper Model** - Will be downloaded automatically on first use (~1.5GB)

## Installation Steps

### 1. Install FFmpeg

**Windows:**
```powershell
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
# Add to PATH
```

**Verify installation:**
```powershell
ffmpeg -version
```

### 2. Install Audio Dependencies

```bash
# Install docling with audio support
uv pip install "docling[audio]"

# Or with pip
pip install "docling[audio]"
```

This will install:
- `torch` - Deep learning framework
- `torchaudio` - Audio processing
- `whisper` - OpenAI's speech recognition model
- Other audio processing libraries

### 3. Test Audio Transcription

```bash
uv run python docling_basics/03_audio_transcription.py
```

## Current Status

The ingestion script will now:
- ✅ Skip audio files gracefully with a warning
- ✅ Process all other document types normally
- ✅ Show clear error message if audio dependencies are missing

## Alternative: Pre-transcribe Audio Files

If you don't want to install the audio dependencies, you can:

1. Use an online transcription service (e.g., Whisper Web, AssemblyAI)
2. Save transcripts as `.txt` or `.md` files
3. Place them in the `documents/` folder
4. They'll be processed like regular text documents

## Troubleshooting

### "Audio transcription not available - missing dependencies"

Install audio dependencies:
```bash
uv pip install "docling[audio]"
```

### "FFmpeg not found"

Install FFmpeg and add it to your PATH.

### Out of Memory Errors

Whisper models require GPU or significant RAM:
- Turbo model: ~2GB VRAM or 4GB RAM
- Consider using smaller models or cloud transcription services

## Notes

- Audio transcription is **optional** - the RAG system works fine without it
- First transcription will download the Whisper model (~1.5GB)
- GPU acceleration recommended for faster transcription
- Each audio file takes 1-5 minutes to transcribe depending on length
