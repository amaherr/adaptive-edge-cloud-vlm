from __future__ import annotations

import sys

import pytest

from adaptive_vlm.transport.tensors import TensorPayload


def test_tensor_payload_preserves_arbitrary_bfloat16_bits() -> None:
    data = bytes(range(16))
    payload = TensorPayload(
        shape=(2, 4),
        dtype="bfloat16",
        byte_order=sys.byteorder,
        layout="contiguous",
        data=data,
    )

    assert payload.data == data
    assert payload.expected_nbytes == 16
    assert payload.metadata() == {
        "shape": [2, 4],
        "dtype": "bfloat16",
        "byte_order": sys.byteorder,
        "layout": "contiguous",
        "nbytes": 16,
    }


def test_tensor_payload_rejects_inconsistent_byte_count() -> None:
    with pytest.raises(ValueError, match="expected 16"):
        TensorPayload(
            shape=(2, 4),
            dtype="bfloat16",
            byte_order=sys.byteorder,
            layout="contiguous",
            data=b"short",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "int8", "dtype"),
        ("byte_order", "middle", "byte order"),
        ("layout", "strided", "layout"),
        ("shape", (0, 4), "shape"),
    ],
)
def test_tensor_payload_rejects_unsupported_metadata(
    field: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "shape": (2, 4),
        "dtype": "bfloat16",
        "byte_order": sys.byteorder,
        "layout": "contiguous",
        "data": bytes(16),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        TensorPayload(**arguments)  # type: ignore[arg-type]
