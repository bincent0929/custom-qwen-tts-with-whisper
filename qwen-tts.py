import os
import tempfile
import numpy as np
import torch
import soundfile as sf
import whisper
import gc
from qwen_tts import Qwen3TTSModel
from pyannote.audio import Pipeline

REF_AUDIO_PATH = "/path/to/reference.wav"
OUTPUT_TEXT = "This is the new text I want spoken in my cloned voice."
OUTPUT_PATH = "output.wav"

# --- Step 0: Diarize reference audio and prompt for speaker selection ---
# Diarization means "speaker" splitting basically.
print("Loading diarization pipeline...")
hf_token = os.environ.get("HF_TOKEN") or input("Enter your HuggingFace token: ").strip()
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token,
)
diarization_pipeline.to(torch.device("cuda"))

print("Running speaker diarization...")
diarization = diarization_pipeline(REF_AUDIO_PATH)

speaker_segments = {}
for turn, _, speaker in diarization.itertracks(yield_label=True):
    if speaker not in speaker_segments:
        speaker_segments[speaker] = []
    speaker_segments[speaker].append((turn.start, turn.end))

print("\nDetected speakers:")
speakers = sorted(speaker_segments.keys())
for i, speaker in enumerate(speakers):
    segments = speaker_segments[speaker]
    total_duration = sum(end - start for start, end in segments)
    print(f"  [{i}] {speaker} — {len(segments)} segment(s), {total_duration:.1f}s total")

choice = int(input("\nEnter the number of the speaker to emulate: "))
chosen_speaker = speakers[choice]
print(f"Selected: {chosen_speaker}")

audio_data, sample_rate = sf.read(REF_AUDIO_PATH)
segments = speaker_segments[chosen_speaker]
extracted = np.concatenate([
    audio_data[int(start * sample_rate):int(end * sample_rate)]
    for start, end in segments
])

tmp_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
tmp_ref.close()
sf.write(tmp_ref.name, extracted, sample_rate)
print(f"Extracted {len(segments)} segment(s) from {chosen_speaker}")

del diarization_pipeline
gc.collect()
torch.cuda.empty_cache()

# --- Step 1: Transcribe reference audio with Whisper ---
print("Loading Whisper...")
whisper_model = whisper.load_model("turbo", device="cuda")

print("Transcribing reference audio...")
result = whisper_model.transcribe(tmp_ref.name, verbose=False)
ref_text = result["text"].strip()
print(f"Transcribed: {ref_text!r}")

del whisper_model
gc.collect()
torch.cuda.empty_cache()

# --- Step 2: Generate cloned voice with Qwen3-TTS ---
print("Loading Qwen3-TTS...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    # If you have more powerful hardware
    # or maybe patience, you can use
    # the recommended `flash_attention_2`
    # here instead of `sdpa`
    attn_implementation="sdpa",
)

print("Generating speech...")
wavs, sr = model.generate_voice_clone(
    text=OUTPUT_TEXT,
    language="English",
    ref_audio=tmp_ref.name,
    ref_text=ref_text,
)

sf.write(OUTPUT_PATH, wavs[0], sr)
print(f"Saved to {OUTPUT_PATH}")

os.unlink(tmp_ref.name)
