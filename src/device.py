"""Device selection — CUDA if available, otherwise CPU."""

import torch


def get_device() -> str:
    """
    Pick compute device for embedding/inference.

    - cuda: NVIDIA GPU (Colab T4, Linux/Windows server, etc.)
    - cpu:  fallback (Mac M1, machines without CUDA)
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return "cuda"

    print("Using CPU (no CUDA GPU detected)")
    return "cpu"


def use_fp16(device: str) -> bool:
    """FP16 only makes sense on CUDA."""
    return device == "cuda"
