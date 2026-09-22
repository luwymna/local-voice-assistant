from piper.voice import PiperVoice
import sounddevice as sd
import numpy as np

voice = PiperVoice.load("en_US-lessac-medium.onnx")

stream = sd.OutputStream(
    samplerate=voice.config.sample_rate,
    channels=1,
    dtype="int16",
)
stream.start()

# Just remove the speaker_id argument entirely
for chunk in voice.synthesize("Hello, this is a test of the new speak function."):
    stream.write(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))

stream.stop()
stream.close()
print("Done.")