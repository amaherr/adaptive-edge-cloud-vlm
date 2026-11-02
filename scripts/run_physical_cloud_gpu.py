"""Launch the cloud CLI on WSL hosts where NVML device counting crashes.

The Windows CUDA driver remains usable, but this host's WSL NVML library
segfaults when PyTorch asks it to enumerate devices. Returning -1 from the
NVML probe tells PyTorch to use its CUDA runtime device count instead.
"""

from __future__ import annotations


def main() -> int:
    import torch

    torch.cuda._device_count_nvml = lambda: -1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this launcher outside the sandbox")

    from adaptive_vlm.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
