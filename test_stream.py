import os
import sounddevice as sd

import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

print("Loading model...")
config = XttsConfig()
config.load_json("XTTS-v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir="XTTS-v2",
    use_deepspeed=True
    )
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["voices\\Arnold.wav"])


print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    sd.play(chunk.cpu(), 24000, blocking=False)
    wav_chuncks.append(chunk)

wav = torch.cat(wav_chuncks, dim=0).squeeze().unsqueeze(0).cpu()[0]

# sd.play(wav[0], 24000)

torchaudio.save("xtts_streaming.wav", wav, 24000)
