import ollama
import whisper
import sounddevice as sd
import numpy as np
import wave
import tempfile
import os
import sys

from piper.voice import PiperVoice

whisper_model = whisper.load_model("base")
voice = PiperVoice.load("en_US-lessac-medium.onnx")


def record_audio(duration=5, sample_rate=16000):
    """Record from mic for `duration` seconds and save to a temp WAV."""
    print("Listening...")
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except Exception as e:
        print(f"Recording failed: {e}")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp.name


def transcribe(audio_path):
    """Whisper speech-to-text."""
    return whisper_model.transcribe(audio_path)["text"].strip()


def get_llm_response(messages, user_text):
    """Send user text to Ollama and get a reply."""
    messages.append({"role": "user", "content": user_text})
    response = ollama.chat(model="llama3.2:3b", messages=messages)
    reply = response["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply


def speak(text):
    """Synthesize with Piper and stream audio to speakers via sounddevice."""
    if not text.strip():
        return
    try:
        stream = sd.OutputStream(
            samplerate=voice.config.sample_rate,
            channels=1,
            dtype="int16",
        )
        stream.start()
        for chunk in voice.synthesize(text):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            stream.write(audio)
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"TTS failed: {e}")
        print(f"[Would have said] {text}")


#main loop
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful voice assistant. "
            "Keep replies concise and conversational, 2-3 sentences max."
        ),
    }
]

print("Voice assistant ready. Speak now. Press Ctrl+C to exit.\n")

try:
    while True:
        audio_file = record_audio(duration=5)
        if audio_file is None:
            continue

        user_text = transcribe(audio_file)
        os.unlink(audio_file)

        if not user_text:
            print("(nothing heard, try again)\n")
            continue

        print(f"You: {user_text}")

        reply = get_llm_response(messages, user_text)
        print(f"Assistant: {reply}\n")

        speak(reply)
except KeyboardInterrupt:
    print("\nGoodbye.")
