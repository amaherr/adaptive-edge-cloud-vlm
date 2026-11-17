"""Per-row int8 encoding of the projected tensor (protocol amendment A2a, rule 3).

The edge quantizes the serialized bf16 tensor; the cloud dequantizes it back to
bf16 bytes before the unchanged text path runs. Both directions work on CPU
bytes with fixed float32 arithmetic, so a given bf16 tensor always yields the
same int8 bytes and the same bf16 reconstruction, on any host or GPU.

Encoding, per row of the last dimension:

- ``scale = fp16(fp32(max|x|) / 127)``; a row whose scale is zero encodes as 0;
- ``q = clamp(round_half_even(fp32(x) / fp32(scale)), -127, 127)`` as int8;
- wire bytes: the int8 values, then one little-endian fp16 scale per row.

Decoding: ``bf16(fp32(q) * fp32(scale))``, rounded to nearest even.

Torch is used where it is installed (the model environments); otherwise a
pure-Python implementation of the same arithmetic serves the fake model and
tests. ``tests/transport/test_quantization.py`` checks they agree bit for bit.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
from array import array
from collections.abc import Callable
from typing import Any

from adaptive_vlm.timing import perf_counter_ns
from adaptive_vlm.transport.tensors import INT8_ROWWISE, TensorPayload


def _torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _rows(shape: tuple[int, ...]) -> tuple[int, int]:
    columns = shape[-1]
    rows = 1
    for size in shape[:-1]:
        rows *= size
    return rows, columns


def _require_bf16(payload: TensorPayload) -> None:
    if payload.dtype != "bfloat16" or payload.quantization is not None:
        raise ValueError("int8 quantization needs an unquantized bfloat16 tensor")
    if payload.byte_order != sys.byteorder:
        raise ValueError("tensor byte order differs from host")


def quantize_int8_rows(
    payload: TensorPayload, *, use_torch: bool | None = None
) -> TensorPayload:
    """Encode a bf16 payload as per-row int8 values and fp16 scales."""

    _require_bf16(payload)
    rows, columns = _rows(payload.shape)
    torch = _torch() if use_torch is not False else None
    if use_torch and torch is None:
        raise RuntimeError("torch is not installed")
    if torch is not None:
        values, scales = _quantize_torch(torch, payload.data, rows, columns)
    else:
        values, scales = _quantize_python(payload.data, rows, columns)
    return TensorPayload(
        shape=payload.shape,
        dtype="int8",
        byte_order=payload.byte_order,
        layout=payload.layout,
        data=values + scales,
        quantization=INT8_ROWWISE,
    )


def dequantize(
    payload: TensorPayload, *, use_torch: bool | None = None
) -> TensorPayload:
    """Decode an int8 payload back to bf16 bytes; bf16 passes through."""

    if payload.quantization is None:
        return payload
    if payload.quantization != INT8_ROWWISE or payload.dtype != "int8":
        raise ValueError(f"unsupported quantization: {payload.quantization}")
    if payload.byte_order != sys.byteorder:
        raise ValueError("tensor byte order differs from host")
    rows, columns = _rows(payload.shape)
    values = payload.data[: rows * columns]
    scales = payload.data[rows * columns :]
    torch = _torch() if use_torch is not False else None
    if use_torch and torch is None:
        raise RuntimeError("torch is not installed")
    if torch is not None:
        data = _dequantize_torch(torch, values, scales, rows, columns)
    else:
        data = _dequantize_python(values, scales, rows, columns)
    return TensorPayload(
        shape=payload.shape,
        dtype="bfloat16",
        byte_order=payload.byte_order,
        layout=payload.layout,
        data=data,
    )


def timed(
    function: Callable[[TensorPayload], TensorPayload], payload: TensorPayload
) -> tuple[TensorPayload, float]:
    started = perf_counter_ns()
    result = function(payload)
    return result, (perf_counter_ns() - started) / 1_000_000


# -- torch ------------------------------------------------------------------


def _quantize_torch(
    torch: Any, data: bytes, rows: int, columns: int
) -> tuple[bytes, bytes]:
    x = (
        torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
        .reshape(rows, columns)
        .float()
    )
    if not bool(torch.isfinite(x).all()):
        raise ValueError("cannot quantize a tensor with non-finite values")
    scale = (x.abs().amax(dim=1) / 127).to(torch.float16)
    if not bool(torch.isfinite(scale).all()):
        raise ValueError("int8 row scale overflows fp16")
    divisor = scale.float().unsqueeze(1)
    zero = divisor == 0
    q = torch.round(x / torch.where(zero, torch.ones_like(divisor), divisor))
    q = torch.where(zero, torch.zeros_like(q), q).clamp(-127, 127).to(torch.int8)
    return (
        q.contiguous().view(torch.uint8).numpy().tobytes(),
        scale.contiguous().view(torch.uint8).numpy().tobytes(),
    )


def _dequantize_torch(
    torch: Any, values: bytes, scales: bytes, rows: int, columns: int
) -> bytes:
    q = torch.frombuffer(bytearray(values), dtype=torch.int8).reshape(rows, columns)
    scale = torch.frombuffer(bytearray(scales), dtype=torch.float16).float()
    restored = (q.float() * scale.unsqueeze(1)).to(torch.bfloat16)
    data: bytes = restored.contiguous().view(torch.uint8).numpy().tobytes()
    return data


# -- pure Python ------------------------------------------------------------

_F32 = struct.Struct("=f")
_F16 = struct.Struct("=e")


def _f32(value: float) -> float:
    result: float = _F32.unpack(_F32.pack(value))[0]
    return result


def _bf16_bits(value: float) -> int:
    """Round a float32 value to bf16 bits, nearest even (as torch does)."""

    bits: int = struct.unpack("=I", _F32.pack(value))[0]
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def _quantize_python(data: bytes, rows: int, columns: int) -> tuple[bytes, bytes]:
    halves = array("H")
    halves.frombytes(data)
    values = bytearray(rows * columns)
    scales = bytearray()
    zero_row = bytes(columns * 2)
    for row in range(rows):
        raw = data[row * columns * 2 : (row + 1) * columns * 2]
        if raw == zero_row:
            scales += _F16.pack(0.0)
            continue
        floats = [
            _F32.unpack(struct.pack("=I", half << 16))[0]
            for half in halves[row * columns : (row + 1) * columns]
        ]
        if not all(math.isfinite(value) for value in floats):
            raise ValueError("cannot quantize a tensor with non-finite values")
        try:
            packed = _F16.pack(_f32(max(abs(v) for v in floats) / 127))
        except OverflowError as exc:
            raise ValueError("int8 row scale overflows fp16") from exc
        scale = _F16.unpack(packed)[0]
        scales += _F16.pack(scale)
        if scale == 0:
            continue
        for column, value in enumerate(floats):
            q = max(-127, min(127, round(_f32(value / scale))))
            values[row * columns + column] = q & 0xFF
    return bytes(values), bytes(scales)


def _dequantize_python(values: bytes, scales: bytes, rows: int, columns: int) -> bytes:
    signed = array("b")
    signed.frombytes(values)
    out = array("H", bytes(rows * columns * 2))
    for row in range(rows):
        scale = _F16.unpack(scales[row * 2 : row * 2 + 2])[0]
        if scale == 0:
            continue
        start = row * columns
        for column in range(columns):
            q = signed[start + column]
            if q:
                out[start + column] = _bf16_bits(_f32(q * scale))
    return out.tobytes()
