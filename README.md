# Welsh mutation analysis pipeline

This is an interactive pipeline for transcribing Welsh audio, detecting initial mutations, validating candidate findings against captions, reviewing results, and producing corpus figures.

## Setup

1. Install Python 3.11 or newer and FFmpeg, ensuring both are available on your command line.
2. Create and activate a virtual environment.
3. Install the Python dependencies with `python -m pip install -r requirements.txt`.
4. Install the Welsh spaCy model `cy_ud_cy_ccg` compatible with your spaCy version. The pipeline continues without it, but parser-based validation will be unavailable.
5. Copy `.env.example` to your preferred environment-variable setup. `WELSH_ANALYSIS_DIR` is optional; without it, data is stored in `~/welsh_analysis` rather than beside the source code.
6. Run `python welsh_pipeline.py`.

## State and recovery

Queue, cache, and output files live in `WELSH_ANALYSIS_DIR`. Queue failures are recorded in `failed_videos.json` and retried up to three times. Successfully processed local MP3 files are recorded in `processed_local_mp3s.json`; changing a file makes it eligible for processing again.

Never store API keys or email passwords in source files. Use environment variables instead.
