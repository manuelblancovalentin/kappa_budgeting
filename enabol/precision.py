from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from .dtypes import HLSDataType


RESERVED_NAMES = {"loss", "input", "__default__"}


class PrecisionDict(dict[str, dict[str, HLSDataType | None]]):
    """Layer-indexed precision map.

    Keys are semantic layer names such as ``dense0`` or reserved names such as
    ``input`` and ``loss``. Values are dictionaries whose fields describe the
    storage or signal being quantized: ``weight``, ``bias``, ``activation``,
    ``gradient``, ``update``, ``accumulator``, or ``value``.
    """

    def __init__(self, data: Mapping[str, Mapping[str, Any]] | None = None):
        super().__init__()
        if data:
            for layer_name, fields in data.items():
                self[layer_name] = {
                    field: self._parse_dtype(dtype)
                    for field, dtype in fields.items()
                }

    @staticmethod
    def _parse_dtype(dtype: Any) -> HLSDataType | None:
        if dtype is None:
            return None
        return HLSDataType.from_dtype(dtype)

    def dtype(self, layer_name: str, field: str, default: HLSDataType | None = None) -> HLSDataType | None:
        if layer_name in self and field in self[layer_name]:
            return self[layer_name][field]
        if "__default__" in self and field in self["__default__"]:
            return self["__default__"][field]
        return default

    def has(self, layer_name: str, field: str) -> bool:
        return self.dtype(layer_name, field) is not None

    def layers(self) -> list[str]:
        return [name for name in self.keys() if name != "__default__"]

    def fields(self, layer_name: str) -> list[str]:
        return list(self.get(layer_name, {}).keys())

    def validate_model(self, model: tf.keras.Model, *, allow_missing: bool = True) -> None:
        layer_names = {layer.name for layer in model.layers}
        if "loss" in layer_names:
            raise ValueError("'loss' is reserved for loss precision and cannot be used as a layer name")
        if "input" in layer_names:
            raise ValueError("'input' is reserved for input precision and cannot be used as a layer name")

        unknown = set(self.layers()) - layer_names - {"loss", "input"}
        if unknown:
            raise ValueError(f"PrecisionDict contains entries that do not match model layers: {sorted(unknown)}")

        if not allow_missing:
            missing = layer_names - set(self.layers())
            if missing:
                raise ValueError(f"PrecisionDict is missing model layers: {sorted(missing)}")

    def describe(self) -> str:
        lines = ["PrecisionDict("]
        for layer_name, fields in self.items():
            lines.append(f"  {layer_name}:")
            for field, dtype in fields.items():
                lines.append(f"    {field}: {dtype}")
        lines.append(")")
        return "\n".join(lines)

    def summary(self) -> str:
        if not self:
            return "PrecisionDict: disabled"

        grouped: dict[str, list[str]] = {}
        for layer_name, fields in self.items():
            for field, dtype in fields.items():
                label = "None" if dtype is None else str(dtype)
                grouped.setdefault(label, []).append(f"{layer_name}.{field}")

        lines = [
            f"PrecisionDict: {len(self.layers())} layer entries | {len(grouped)} unique dtypes",
        ]
        for dtype_label, uses in grouped.items():
            preview = ", ".join(uses[:4])
            if len(uses) > 4:
                preview += f", +{len(uses) - 4} more"
            lines.append(f"  {dtype_label}: {preview}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return self.describe()

def ensure_precision_dict(precision: PrecisionDict | Mapping[str, Mapping[str, Any]] | None) -> PrecisionDict | None:
    if precision is None:
        return None
    if isinstance(precision, PrecisionDict):
        return precision
    return PrecisionDict(precision)
