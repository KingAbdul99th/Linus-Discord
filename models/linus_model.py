
import os
import torch
import numpy as np
from llama_cpp import Llama
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"

from TTS.tts.configs.xtts_config import XttsConfig, XttsAudioConfig
from TTS.tts.models.xtts import Xtts, XttsArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from models.kcpp_client import generate_response

torch.set_default_device('cuda')
torch.serialization.add_safe_globals([XttsConfig])
torch.serialization.add_safe_globals([XttsAudioConfig])
torch.serialization.add_safe_globals([BaseDatasetConfig])
torch.serialization.add_safe_globals([XttsArgs])


class LinusModel:
    def __init__(self, llama_cpp: bool = False) -> None:
        self.voice = "voices\\Arnold.wav"
        if llama_cpp:
            self._load_llamacpp_model()
            self.respond = self.respond_llamacpp
        else:
            self.respond = generate_response
        
        self._load_tts_model()

    def _load_llamacpp_model(self):
        self.llm = Llama(
            model_path=os.path.join("gguf_models", "hermes-trismegistus-mistral-7b.Q4_K_M.gguf"),
            main_gpu=0,
            n_gpu_layers=-1,
            n_batch=512,
            stop=["USER:", "\n", "You are Linus"],
            verbose=True,
        )

    def _load_tts_model(self):
        print("Loading TTS model...")
        config = XttsConfig()
        config.load_json("XTTS-v2/config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_dir="XTTS-v2",
            use_deepspeed=True
            )
        model.cuda()
        self.tts = model
        self.gpt_cond_latent, self.speaker_embedding = self.tts.get_conditioning_latents(audio_path=[self.voice])

    def respond_llamacpp(self, message: str) -> str:
        prompt = f"""
            You are Linus. A businessman that reluctantly answers questions with sarcastic responses.
            USER: {message}
            ASSISTANT:
        """
        response = self.llm(
            prompt,
            max_tokens=300,
            stop=["USER:", "\n", "You are Linus"],
            echo=False
        )
        return response["choices"][0]["text"]

    def generate_voice(self, text: str):
        output = self.tts.inference(
            text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
            enable_text_splitting=True,
        )
        output_wav = torch.tensor(output["wav"]).unsqueeze(0).cpu()
        torchaudio.save("generated.wav", output_wav, 24000, encoding="PCM_S")

        return "generated.wav"

    def wav_postprocess(self, wav):
        """Post process the output waveform"""
        if isinstance(wav, list):
            wav = torch.cat(wav, dim=0)
        wav = wav.clone().detach().cpu().numpy()
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)
        return wav

    def generate_voice_stream(self, text: str):
        output = self.tts.inference_stream(
            text,
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
            enable_text_splitting=True,
        )

        for chunk in output:
            processed_chunk = self.wav_postprocess(chunk)
            processed_bytes = processed_chunk.tobytes()
            yield processed_bytes