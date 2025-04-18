conda create -n LinusBot python=3.11
conda activate LinusBot

pip install -r requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
set FORCE_CMAKE=1 && set CMAKE_ARGS=-DGGML_CUDA=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off
pip install --upgrade --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install coqui-tts

huggingface-cli download TheBloke/Hermes-Trismegistus-Mistral-7B-GGUF hermes-trismegistus-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-fp16.gguf --local-dir . --local-dir-use-symlinks False

python serve.py

https://gist.github.com/comhad/de830d6d1b7ae1f165b925492e79eac8
https://unix.stackexchange.com/questions/225401/how-to-see-full-log-from-systemctl-status-service
https://github.com/ausboss/DiscordLangAgent