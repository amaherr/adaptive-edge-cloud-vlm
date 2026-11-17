"""Per-row int8 encoding of the projected tensor (amendment A2a rule 3)."""

from __future__ import annotations

import importlib.util
import random
import struct
import sys

import pytest

from adaptive_vlm.transport.protocol import Frame, MessageType, ProtocolError
from adaptive_vlm.transport.quantization import dequantize, quantize_int8_rows
from adaptive_vlm.transport.tensors import INT8_ROWWISE, TensorPayload

FINGERPRINT = "a" * 64
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _bf16(values: list[float]) -> bytes:
    """Truncate float32 to bf16 bits (the test inputs are exact in bf16)."""

    return b"".join(
        struct.pack("=H", struct.unpack("=I", struct.pack("=f", v))[0] >> 16)
        for v in values
    )


def _floats(data: bytes) -> list[float]:
    halves = struct.unpack(f"={len(data) // 2}H", data)
    return [struct.unpack("=f", struct.pack("=I", h << 16))[0] for h in halves]


def _tensor(rows: int, columns: int, seed: int = 3) -> TensorPayload:
    generator = random.Random(seed)
    values = [
        _floats(_bf16([generator.gauss(0, 2.0 ** generator.randint(-4, 3))]))[0]
        for _ in range(rows * columns)
    ]
    values[columns : 2 * columns] = [0.0] * columns  # one all-zero row
    return TensorPayload(
        shape=(rows, columns),
        dtype="bfloat16",
        byte_order=sys.byteorder,
        layout="contiguous",
        data=_bf16(values),
    )


def test_round_trip_is_within_half_a_step_per_row() -> None:
    original = _tensor(6, 64)
    encoded = quantize_int8_rows(original, use_torch=False)
    assert encoded.quantization == INT8_ROWWISE and encoded.dtype == "int8"
    assert len(encoded.data) == 6 * 64 + 6 * 2
    decoded = dequantize(encoded, use_torch=False)
    assert decoded.dtype == "bfloat16" and decoded.quantization is None
    before, after = _floats(original.data), _floats(decoded.data)
    for row in range(6):
        values = before[row * 64 : (row + 1) * 64]
        step = max(abs(v) for v in values) / 127
        for x, y in zip(values, after[row * 64 : (row + 1) * 64], strict=True):
            # Half a quantization step, plus bf16 rounding of the result.
            assert abs(x - y) <= step * 0.51 + abs(y) * 2**-8
    assert after[64:128] == [0.0] * 64
    assert dequantize(original) is original  # bf16 passes through


def test_wire_bytes_are_half_of_bf16_for_the_projected_shape() -> None:
    bf16 = 729 * 2048 * 2
    int8 = TensorPayload(
        shape=(729, 2048),
        dtype="int8",
        byte_order=sys.byteorder,
        layout="contiguous",
        data=bytes(729 * 2048 + 729 * 2),
        quantization=INT8_ROWWISE,
    )
    assert abs(len(int8.data) / bf16 - 0.50) <= 0.01
    assert int8.metadata()["quantization"] == INT8_ROWWISE


def test_encoding_is_deterministic_and_rejects_bad_input() -> None:
    original = _tensor(3, 32, seed=9)
    first = quantize_int8_rows(original, use_torch=False)
    assert quantize_int8_rows(original, use_torch=False) == first
    with pytest.raises(ValueError, match="bfloat16"):
        quantize_int8_rows(first, use_torch=False)
    infinite = TensorPayload(
        shape=(1, 2),
        dtype="bfloat16",
        byte_order=sys.byteorder,
        layout="contiguous",
        data=_bf16([float("inf"), 1.0]),
    )
    with pytest.raises(ValueError, match="non-finite"):
        quantize_int8_rows(infinite, use_torch=False)
    with pytest.raises(ValueError, match="quantization"):
        TensorPayload(
            shape=(1, 2),
            dtype="int8",
            byte_order=sys.byteorder,
            layout="contiguous",
            data=bytes(4),
            quantization="int4",
        )
    with pytest.raises(ValueError, match="dequantize"):
        first.to_torch(object(), device="cpu")


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_torch_and_python_encodings_agree_bit_for_bit() -> None:
    original = _tensor(8, 128, seed=11)
    python = quantize_int8_rows(original, use_torch=False)
    torch = quantize_int8_rows(original, use_torch=True)
    assert python == torch
    assert dequantize(python, use_torch=False) == dequantize(torch, use_torch=True)


def _tensor_frame(metadata_extra: dict[str, object], payload: bytes) -> Frame:
    return Frame(
        1,
        MessageType.TENSOR_REQUEST,
        "r1",
        {
            "prompt": "What?",
            "model_fingerprint": FINGERPRINT,
            "shape": [2, 4],
            "byte_order": sys.byteorder,
            "layout": "contiguous",
            "pipeline": "synchronous",
            **metadata_extra,
        },
        payload,
    )


def test_protocol_accepts_int8_and_rejects_unknown_quantization() -> None:
    payload = bytes(2 * 4 + 2 * 2)
    _tensor_frame(
        {"dtype": "int8", "nbytes": len(payload), "quantization": INT8_ROWWISE},
        payload,
    )
    with pytest.raises(ProtocolError, match="quantization"):
        _tensor_frame(
            {"dtype": "int8", "nbytes": len(payload), "quantization": "int4"},
            payload,
        )
    with pytest.raises(ProtocolError, match="unsupported"):
        _tensor_frame({"dtype": "int8", "nbytes": 8}, bytes(8))  # no scales
