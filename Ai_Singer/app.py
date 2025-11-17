import os
import time
import shutil

from flask import Flask, request, render_template, url_for
from pydub import AudioSegment

import torch
import scipy.io.wavfile as wavfile
import soundfile as sf

# ---- MusicGen via transformers ----
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# ---- TTS via SpeechT5 ----
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# ================== CONFIG ==================
app = Flask(__name__)

OUTPUT_DIR = os.path.join("static", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ================== LOAD MODELS ==================
print("Loading MusicGen (facebook/musicgen-small) ...")
mg_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
mg_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
mg_model.to(DEVICE)

print("Loading SpeechT5 TTS ...")
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)

print("Creating random speaker embedding ...")
speaker_embeddings = torch.randn(1, 512).to(DEVICE)

# ================== HELPERS ==================
def seconds_to_tokens(duration_sec: int) -> int:
    """Approx map duration (sec) -> max_new_tokens for MusicGen."""
    duration_sec = max(5, min(duration_sec, 30))  # clamp 5–30
    tokens = int(1503 * (duration_sec / 30.0))
    tokens = max(64, min(tokens, 1503))
    return tokens


def generate_music(prompt: str, duration: int = 20) -> str:
    """Generate backing track with MusicGen and save to WAV."""
    ts = int(time.time())
    out_name = f"backing_{ts}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"[MusicGen] Prompt={prompt!r}, duration={duration}s")
    inputs = mg_processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    max_tokens = seconds_to_tokens(duration)
    with torch.no_grad():
        audio_values = mg_model.generate(
            **inputs,
            do_sample=True,
            max_new_tokens=max_tokens,
        )

    waveform = audio_values[0, 0].cpu().numpy()
    sampling_rate = mg_model.config.audio_encoder.sampling_rate
    wavfile.write(out_path, rate=sampling_rate, data=waveform)
    print(f"[MusicGen] Saved backing track -> {out_path}")
    return out_path


def generate_tts(lyrics: str) -> str:
    """Generate raw voice audio from lyrics using SpeechT5."""
    ts = int(time.time())
    out_name = f"voice_raw_{ts}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"[TTS] Generating voice for {len(lyrics)} chars")

    inputs = tts_processor(text=lyrics, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)

    with torch.no_grad():
        speech = tts_model.generate_speech(
            input_ids,
            speaker_embeddings,
            vocoder=tts_vocoder,
        )

    sf.write(out_path, speech.cpu().numpy(), samplerate=16000)
    print(f"[TTS] Saved raw voice -> {out_path}")
    return out_path


def run_rvc(input_wav: str) -> str:
    """
    Placeholder for RVC (voice conversion).
    Currently just copies input_wav to a new file.
    """
    ts = int(time.time())
    out_name = f"voice_rvc_{ts}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    print(f"[RVC] Placeholder copy {input_wav} -> {out_path}")
    shutil.copy(input_wav, out_path)
    return out_path


def mix_tracks(backing_path: str, vocal_path: str, vocal_gain_db: float = -3.0) -> str:
    """
    Mix backing and vocals using pydub, return path to final WAV.
    """
    ts = int(time.time())
    out_name = f"final_song_{ts}.wav"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    print(f"[Mix] Loading backing: {backing_path}")
    backing = AudioSegment.from_file(backing_path)

    print(f"[Mix] Loading vocal: {vocal_path}")
    vocal = AudioSegment.from_file(vocal_path)

    min_len = min(len(backing), len(vocal))
    backing = backing[:min_len]
    vocal = vocal[:min_len]

    vocal = vocal + vocal_gain_db

    print("[Mix] Overlaying vocal on backing ...")
    mixed = backing.overlay(vocal)
    mixed.export(out_path, format="wav")
    print(f"[Mix] Final song saved -> {out_path}")
    return out_path


# ================== FLASK ROUTES ==================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/generate_full_song", methods=["POST"])
def generate_full_song():
    prompt_music = request.form.get("prompt_music", "").strip()
    lyrics = request.form.get("lyrics", "").strip()
    duration_str = request.form.get("duration", "20").strip()

    if not prompt_music:
        return render_template("index.html", error="Please enter a music prompt.")
    if not lyrics:
        return render_template("index.html", error="Please enter lyrics for vocals.")

    try:
        duration = int(duration_str)
    except ValueError:
        duration = 20
    duration = max(5, min(duration, 30))

    try:
        backing_path = generate_music(prompt_music, duration=duration)
        voice_raw_path = generate_tts(lyrics)
        voice_rvc_path = run_rvc(voice_raw_path)
        final_path = mix_tracks(backing_path, voice_rvc_path, vocal_gain_db=-3.0)
        final_filename = os.path.basename(final_path)
        final_url = url_for("static", filename=f"output/{final_filename}")
        return render_template("index.html", final_url=final_url)
    except Exception as e:
        print("[ERROR] in /generate_full_song:", e)
        return render_template("index.html", error=f"Error: {e}")


if __name__ == "__main__":
    # Run locally on 127.0.0.1:5000
    app.run(host="127.0.0.1", port=5000, debug=False)
