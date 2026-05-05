import torch
import soundfile as sf
import whisper
import gc
from qwen_tts import Qwen3TTSModel

REF_AUDIO_PATH = "/path/to/reference.wav"
OUTPUT_TEXT = "This is the new text I want spoken in my cloned voice."
OUTPUT_PATH = "output.wav"

# --- Step 1: Transcribe reference audio with Whisper ---
print("Loading Whisper...")
whisper_model = whisper.load_model("turbo", device="cuda")

print("Transcribing reference audio...")
result = whisper_model.transcribe(REF_AUDIO_PATH, verbose=False)
ref_text = result["text"].strip()
print(f"Transcribed: {ref_text!r}")

# Free Whisper from VRAM before loading Qwen3-TTS
del whisper_model
gc.collect()
torch.cuda.empty_cache()

# --- Step 2: Generate cloned voice with Qwen3-TTS ---
print("Loading Qwen3-TTS...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

print("Generating speech...")
wavs, sr = model.generate_voice_clone(
    text=OUTPUT_TEXT,
    language="English",
    ref_audio=REF_AUDIO_PATH,
    ref_text=ref_text,
)

sf.write(OUTPUT_PATH, wavs[0], sr)
print(f"Saved to {OUTPUT_PATH}")
