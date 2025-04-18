FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        ffmpeg

# Install any python packages you need
COPY requirements.txt requirements.txt

# Upgrade pip
RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install -r requirements.txt

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu121/torch_stable.html

# RUN pip3 install deepspeed

COPY . .

# Set the entrypoint
ENTRYPOINT [ "python3", "serve.py"]