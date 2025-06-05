
import os, threading, io
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


class TTSStreamReader(io.RawIOBase):
    def __init__(self, xtts_model, voice: str, text: str, language: str):
        super().__init__()
        self.text = text
        self.xtts_model = xtts_model
        self.voice = voice
        self.language = language
        self._generator = None
        self._buffer = b""
        self._eof = False
        self._lock = threading.Lock()
        self.gpt_cond_latent, self.speaker_embedding = self.xtts_model.get_conditioning_latents(audio_path=[self.voice])

        print(f"TTSStreamReader initialized for '{text[:20]}...'")

    def _initialize_generator(self):
        if self._generator is None and not self._eof:
            try:
                self._generator = self.xtts_model.inference_stream(
                    self.text,
                    self.language,
                    self.gpt_cond_latent,
                    self.speaker_embedding,
                    enable_text_splitting=True,
                )
            except Exception as e:
                print(f"TTSStreamReader: Error initializing XTTS generator: {repr(e)}")
                self._eof = True  # Mark as EOF if init fails
        else:
            # This path should ideally not be taken if called correctly
            pass

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        with self._lock:
            if self._eof:
                return 0  # Signal EOF

            # Initialize generator on first read
            if self._generator is None:
                self._initialize_generator()
                if self._eof:
                    return 0

            # Fill the buffer 'b'
            bytes_read = 0
            while bytes_read < len(b):
                # If we have data in our internal buffer, use it first
                if self._buffer:
                    chunk_len = min(len(self._buffer), len(b) - bytes_read)
                    b[bytes_read : bytes_read + chunk_len] = self._buffer[:chunk_len]
                    self._buffer = self._buffer[chunk_len:]
                    bytes_read += chunk_len
                    continue

                # Get next chunk from TTS generator
                try:
                    if self._generator:
                        chunk = next(self._generator)
                        pcm_data = (
                            (chunk.squeeze().cpu().detach() * 32767)
                            .numpy()
                            .astype(np.int16)
                            .tobytes()
                        )
                        self._buffer += pcm_data
                    else:
                        print("TTSStreamReader: Generator is None during read.")
                        self._eof = True
                        break

                except StopIteration:
                    print("TTSStreamReader: Generator finished.")
                    self._eof = True
                    break
                except Exception as e:
                    print(f"TTSStreamReader: Error reading from generator: {repr(e)}")
                    self._eof = True
                    break

            return bytes_read

    def close(self) -> None:
        print("TTSStreamReader: Closing stream.")
        with self._lock:
            self._eof = True
            self._generator = None  # Allow garbage collection
            self._buffer = b""
        super().close()

class LinusModel:
    def __init__(self, llama_cpp: bool = False, enable_voice: bool = False) -> None:
        self.voice = os.getenv("VOICE_PATH")
        if llama_cpp:
            self._load_llamacpp_model()
            self.respond = self.respond_llamacpp
        else:
            self.respond = generate_response
    
        if enable_voice:
            self._load_tts_model()

    def _load_llamacpp_model(self):
        self.llm = Llama(
            model_path=os.getenv("LLM_MODEL_PATH"),
            main_gpu=0,
            n_gpu_layers=-1,
            n_batch=512,
            stop=["USER:", "\n", "You are Linus"],
            verbose=True,
        )

    def _load_tts_model(self):
        print("Loading TTS model...")
        model_path = os.getenv("XTTS_MODEL_PATH")
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
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
        tts_reader = TTSStreamReader(self.tts, self.voice, text, "en")
        return tts_reader