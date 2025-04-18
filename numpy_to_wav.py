import wave
import numpy as np

samplerate = 44100

# A note on the left channel for 1 second.
t = np.linspace(0, 1, samplerate)
left_channel = 0.5 * np.sin(2 * np.pi * 440.0 * t)

# Noise on the right channel.
right_channel = np.random.random(size=samplerate)

# Put the channels together with shape (2, 44100).
audio = np.array([left_channel, right_channel]).T

# Convert to (little-endian) 16 bit integers.
audio = (audio * (2 ** 15 - 1)).astype("<h")

with wave.open("sound1.wav", "w") as f:
    # 2 Channels.
    f.setnchannels(2)
    # 2 bytes per sample.
    f.setsampwidth(2)
    f.setframerate(samplerate)
    f.writeframes(audio.tobytes())