from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf

from .dtypes import HLSDataType, ap_fixed


RESERVED_NAMES = {"loss", "input", "__default__"}

STANDARD_LAYER_PRECISION_FIELDS = {
    "weight": "weight",
    "bias": "bias",
    "activation": "result",
    "result": "result",
    "accumulator": "accum",
    "accum": "accum",
}

TRAINABLE_PRECISION_FIELDS = {
    "loss": "loss",
    "value": "loss",
    "loss_grad": "loss_grad",
    "gradient": "grad_out",
    "grad_in": "grad_in",
    "grad_out": "grad_out",
    "weight_grad": "weight_grad",
    "bias_grad": "bias_grad",
    "gradient_accum": "gradient_accum",
    "raw_update": "raw_update",
    "update": "update",
    "optimizer_state": "optimizer_state",
    "controller_metric": "controller_metric",
    "alpha": "alpha",
}

DEFAULT_TRAINABLE_PRECISION = {
    "loss": ap_fixed(32, 16),
    "loss_grad": ap_fixed(20, 8),
    "grad_in": ap_fixed(20, 8),
    "grad_out": ap_fixed(20, 8),
    "weight_grad": ap_fixed(20, 8),
    "bias_grad": ap_fixed(20, 8),
    "gradient_accum": ap_fixed(28, 14),
    "raw_update": ap_fixed(20, 6),
    "update": ap_fixed(20, 6),
    "optimizer_state": ap_fixed(20, 6),
    "controller_metric": ap_fixed(32, 16),
    "alpha": ap_fixed(16, 4),
}


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


def dtype_to_hls(dtype: HLSDataType | str | None) -> str | None:
    """Convert an ENABOL dtype object or dtype string into an hls4ml config string."""

    if dtype is None:
        return None
    if isinstance(dtype, HLSDataType):
        return repr(dtype)
    return str(dtype)


def set_standard_hls_precision(layer_config: dict[str, Any], field: str, dtype: HLSDataType | None) -> None:
    """Map an ENABOL semantic precision field onto ordinary hls4ml layer precision."""

    hls_field = STANDARD_LAYER_PRECISION_FIELDS.get(field)
    hls_dtype = dtype_to_hls(dtype)
    if hls_field is None or hls_dtype is None:
        return
    layer_config.setdefault("Precision", {})[hls_field] = hls_dtype


def set_trainable_hls_precision(training_config: dict[str, Any], field: str, dtype: HLSDataType | None) -> None:
    """Map one hls4ml trainable precision field onto a training config block."""

    hls_field = TRAINABLE_PRECISION_FIELDS.get(field)
    hls_dtype = dtype_to_hls(dtype)
    if hls_field is None or hls_dtype is None:
        return
    training_config.setdefault("Precision", {})[hls_field] = hls_dtype


def set_trainable_hls_precision_aliases(training_config: dict[str, Any], field: str, dtype: HLSDataType | None) -> None:
    """Expand ENABOL semantic precision aliases into explicit hls4ml trainable fields."""

    if field == "value":
        set_trainable_hls_precision(training_config, "loss", dtype)
        return
    if field == "gradient":
        for alias in ("grad_in", "grad_out", "weight_grad", "bias_grad", "loss_grad"):
            set_trainable_hls_precision(training_config, alias, dtype)
        return
    if field == "update":
        for alias in ("raw_update", "update", "optimizer_state"):
            set_trainable_hls_precision(training_config, alias, dtype)
        return
    if field == "accumulator":
        for alias in ("gradient_accum", "controller_metric"):
            set_trainable_hls_precision(training_config, alias, dtype)
        return

    set_trainable_hls_precision(training_config, field, dtype)


def fill_default_trainable_hls_precision(hls_config: dict[str, Any]) -> None:
    """Fill missing model-level trainable precision fields with conservative defaults."""

    model_training = hls_config.setdefault("Model", {}).setdefault("Training", {})
    model_precision = model_training.setdefault("Precision", {})

    for field, dtype in DEFAULT_TRAINABLE_PRECISION.items():
        model_precision.setdefault(field, dtype_to_hls(dtype))


def apply_hls_precision_config(hls_config: dict[str, Any], precision: PrecisionDict | None) -> None:
    """Apply an ENABOL PrecisionDict to an hls4ml config in-place.

    Ordinary layer precisions are written under ``LayerName.<layer>.Precision``.
    Trainable precisions are written under ``Model.Training.Precision`` or
    ``LayerName.<layer>.Training.Precision``.
    """

    model_config = hls_config.setdefault("Model", {})
    model_training = model_config.setdefault("Training", {})

    if precision is None:
        fill_default_trainable_hls_precision(hls_config)
        return

    default_fields = precision.get("__default__", {})
    for field, dtype in default_fields.items():
        hls_dtype = dtype_to_hls(dtype)
        if hls_dtype is None:
            continue
        if field in STANDARD_LAYER_PRECISION_FIELDS:
            model_config.setdefault("Precision", {})[STANDARD_LAYER_PRECISION_FIELDS[field]] = hls_dtype
        set_trainable_hls_precision_aliases(model_training, field, dtype)

    for layer_name, fields in precision.items():
        if layer_name in {"__default__", "input"}:
            continue
        if layer_name == "loss":
            for field, dtype in fields.items():
                set_trainable_hls_precision_aliases(model_training, field, dtype)
            continue

        layer_config = hls_config.setdefault("LayerName", {}).setdefault(layer_name, {})
        layer_training = layer_config.setdefault("Training", {})
        for field, dtype in fields.items():
            set_standard_hls_precision(layer_config, field, dtype)
            set_trainable_hls_precision_aliases(layer_training, field, dtype)

    fill_default_trainable_hls_precision(hls_config)
