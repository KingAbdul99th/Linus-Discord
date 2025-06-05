# Linus Discord bot
Yet another LLM Discord bot

Welp turns out discord.py has no viable way to get STT :'(
# Install
conda create -n linus python=3.12
conda activate linus
conda install -c conda-forge libstdcxx-ng

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# Install llama-cpp-python (gpu support not yet building successfully)
pip install llama-cpp-python
set FORCE_CMAKE=1 && set CMAKE_ARGS=-DGGML_CUDA=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off
pip install --upgrade --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off -DCMAKE_CUDA_ARCHITECTURES=all-major" pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

pip install coqui-tts

# Download Test local model
huggingface-cli download TheBloke/Hermes-Trismegistus-Mistral-7B-GGUF hermes-trismegistus-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Env file
Copy example.env to .env and fill with your values

# Run
python serve.py
