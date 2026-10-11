"""Bit-preserving tensor payloads for the distributed split boundary."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any

SUPPORTED_DTYPES = {"bfloat16": 2, "float16": 2, "float32": 4}


@dataclass(frozen=True)
class TensorPayload:
    shape: tuple[int, ...]
    dtype: str
    byte_order: str
    layout: str
    data: bytes

    def __post_init__(self) -> None:
        if not 1 <= len(self.shape) <= 8 or any(size <= 0 for size in self.shape):
            raise ValueError("tensor shape must contain 1 to 8 positive dimensions")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported tensor dtype: {self.dtype}")
        if self.byte_order not in {"little", "big"}:
            raise ValueError("tensor byte order must be little or big")
        if self.layout != "contiguous":
            raise ValueError("only contiguous tensor layout is supported")
        if len(self.data) != self.expected_nbytes:
            raise ValueError(
                f"tensor payload has {len(self.data)} bytes; "
                f"expected {self.expected_nbytes}"
            )

    @property
    def expected_nbytes(self) -> int:
        return reduce(mul, self.shape, 1) * SUPPORTED_DTYPES[self.dtype]

    def metadata(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_order": self.byte_order,
            "layout": self.layout,
            "nbytes": len(self.data),
        }

    @classmethod
    def from_torch(cls, tensor: Any) -> TensorPayload:
        """Copy one tensor to CPU without changing its native dtype bits."""

        torch = importlib.import_module("torch")
        contiguous = tensor.detach().contiguous().cpu()
        dtype = str(contiguous.dtype).removeprefix("torch.")
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"unsupported tensor dtype: {dtype}")
        data = contiguous.view(torch.uint8).numpy().tobytes(order="C")
        return cls(
            shape=tuple(int(size) for size in contiguous.shape),
            dtype=dtype,
            byte_order=sys.byteorder,
            layout="contiguous",
            data=data,
        )

    def to_torch(self, torch: Any, *, device: str) -> Any:
        """Reconstruct an owning tensor, rejecting cross-endian interpretation."""

        if self.byte_order != sys.byteorder:
            raise ValueError(
                f"tensor byte order {self.byte_order} differs from host {sys.byteorder}"
            )
        dtype = getattr(torch, self.dtype)
        buffer = bytearray(self.data)
        tensor = torch.frombuffer(buffer, dtype=dtype).reshape(self.shape).clone()
        return tensor.to(device)


@dataclass(frozen=True)
class TensorInferenceRequest:
    request_id: str
    prompt: str
    tensor: TensorPayload
