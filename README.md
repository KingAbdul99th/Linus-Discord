# Linus Discord bot
Yet another LLM Discord bot

# Install
conda create -n linus python=3.12
conda activate linus

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124

# Install llama-cpp-python (gpu support not yet building successfully)
set FORCE_CMAKE=1 && set CMAKE_ARGS=-DGGML_CUDA=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off
pip install --upgrade --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

pip install coqui-tts

# Download Test local model
huggingface-cli download TheBloke/Hermes-Trismegistus-Mistral-7B-GGUF hermes-trismegistus-mistral-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

# Env file
Copy example.env to .env and fill with your values

# Run
python serve.py
