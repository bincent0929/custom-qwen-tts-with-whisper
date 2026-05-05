#!/bin/sh
# Optimized FFMPEG for qwentts input
INPUT_ARG="$1"
OUTPUT_ARG="$2"
# -i input.mp4 — your video file
# -vn — drop the video stream (audio only)
# -acodec pcm_s16le — uncompressed 16-bit WAV, which is what Qwen3-TTS expects
# -ar 24000 — resample to 24kHz (Qwen3-TTS's native rate; saves it from doing it internally)
# -ac 1 — mono (recommended for voice cloning — stereo can confuse speaker embedding)
ffmpeg -i "${INPUT_ARG}" -vn -acodec pcm_s16le -ar 24000 -ac 1 "${OUTPUT_ARG}"
