# CUDAちゃん (Vecteus-v1 Runtime)
— A lightweight LLM runtime for RTX 4060, powered by 4-bit NF4 quantization.

[Link to Model (Hugging Face)](https://huggingface.co/Local-Novel-LLM-project/Vecteus-v1)

### Features
* **Instant Environment** — Fully containerized via Docker (Ubuntu 22.04 + CUDA 12.3.2).
* **Interactive Chat Loop** — Persistent memory session to eliminate redundant weight reloading.
* **Japanese Native Support** — `ja_JP.UTF-8` pre-configured for Japanese-proficient Vecteus-v1.
* **Normal Float 4 Optimization** — Specifically tuned for 8GB VRAM limits using `bitsandbytes`.

### Environments
* **OS**: Windows 11 (WSL2 / Docker)
* **GPU**: NVIDIA GeForce RTX 4060 (8GB VRAM)
* **Driver**: 566.36+
* **Framework**: PyTorch 2.x + Transformers + Accelerate

###Usage
1. Check Environment
```PowerShell
nvidia-smi
```
2. Launch Chat
```Bash
cd src
python3 main.py
```

### Quantization Config
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
```