FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git locales && \
    locale-gen ja_JP.UTF-8 && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir transformers bitsandbytes accelerate sentencepiece protobuf tiktoken && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
ENV LANGUAGE=ja_JP:ja
ENV LC_ALL=ja_JP.UTF-8
ENV PYTHONIOENCODING=utf-8

WORKDIR /workspaces/runtime/projects/cuda-chan/src