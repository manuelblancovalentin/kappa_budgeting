from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from .dtypes import HLSDataType, ap_fixed, ap_int, ap_ufixed, ap_uint


@dataclass(frozen=True)
class RailStats:
    min_value: float
    max_value: float
    saturation_fraction: float
    near_rail_fraction: float
    underflow_fraction: float


def quantize_np(x: Any, dtype: HLSDataType | None) -> Any:
    if dtype is None:
        return x
    return dtype(x)


def quantize_tensor(x: tf.Tensor, dtype: HLSDataType | None, *, ste: bool = True) -> tf.Tensor:
    """Quantize a TensorFlow tensor.

    With ``ste=True``, the forward value is quantized but the gradient is the
    identity straight-through estimator. This is the right default for fake
    quantization inside ``GradientTape``.
    """
    if dtype is None:
        return x
    q = _quantize_tensor_value(tf.cast(x, tf.float32), dtype)
    if ste:
        return x + tf.stop_gradient(q - x)
    return q


def rail_stats(x: Any, dtype: HLSDataType | None, *, near_ratio: float = 0.95) -> RailStats:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or dtype is None:
        return RailStats(np.nan, np.nan, 0.0, 0.0, 0.0)

    lo, hi = dtype.value_range()
    sat = np.mean((arr <= lo) | (arr >= hi))
    rail = max(abs(lo), abs(hi))
    near = np.mean(np.abs(arr) >= near_ratio * rail) if rail > 0 else 0.0

    quantum = getattr(dtype, "quantum", None)
    if quantum is None:
        underflow = 0.0
    else:
        underflow = np.mean((arr != 0.0) & (np.abs(arr) < 0.5 * float(quantum)))

    return RailStats(float(lo), float(hi), float(sat), float(near), float(underflow))


def _quantize_tensor_value(x: tf.Tensor, dtype: HLSDataType) -> tf.Tensor:
    if isinstance(dtype, (ap_fixed, ap_ufixed)):
        frac_bits = dtype.fractional_bits
        scale = float(1 << frac_bits) if frac_bits > 0 else 1.0
        qmode = dtype.QMODE
        omode = dtype.OMODE
        wl = dtype.WL
        signed = isinstance(dtype, ap_fixed)
    elif isinstance(dtype, (ap_int, ap_uint)):
        scale = 1.0
        qmode = "AP_TRN"
        omode = "AP_WRAP"
        wl = dtype.WL
        signed = isinstance(dtype, ap_int)
    else:
        raise TypeError(f"Unsupported TensorFlow quantized dtype: {dtype}")

    x_scaled = x * scale
    rounded = _round_tensor(x_scaled, qmode)

    if omode == "AP_WRAP":
        q_int = _wrap_tensor(rounded, wl, signed=signed)
    elif omode in {"AP_SAT", "AP_SAT_ZERO", "AP_SAT_SYM"}:
        q_int = _clip_tensor(rounded, wl, signed=signed)
    else:
        raise NotImplementedError(f"OMODE '{omode}' is not implemented")
    return q_int / scale


def _round_tensor(x: tf.Tensor, qmode: str) -> tf.Tensor:
    if qmode in {"AP_TRN", "AP_TRN_ZERO"}:
        return tf.where(x >= 0, tf.floor(x), tf.math.ceil(x))
    if qmode in {"AP_RND", "AP_RND_CONV"}:
        return tf.round(x)
    if qmode == "AP_RND_INF":
        return tf.floor(x + 0.5)
    if qmode == "AP_RND_MIN_INF":
        return tf.math.ceil(x - 0.5)
    if qmode == "AP_RND_ZERO":
        rounded = tf.where(x >= 0, tf.floor(x + 0.5), tf.math.ceil(x - 0.5))
        frac = x - tf.where(x >= 0, tf.floor(x), tf.math.ceil(x))
        tie = tf.abs(tf.abs(frac) - 0.5) < 1e-6
        trunc = tf.where(x >= 0, tf.floor(x), tf.math.ceil(x))
        return tf.where(tie, trunc, rounded)
    raise NotImplementedError(f"QMODE '{qmode}' is not implemented")


def _clip_tensor(x: tf.Tensor, WL: int, *, signed: bool) -> tf.Tensor:
    if signed:
        lo = float(-(1 << (WL - 1)))
        hi = float((1 << (WL - 1)) - 1)
    else:
        lo = 0.0
        hi = float((1 << WL) - 1)
    return tf.clip_by_value(x, lo, hi)


def _wrap_tensor(x: tf.Tensor, WL: int, *, signed: bool) -> tf.Tensor:
    mod = float(1 << WL)
    wrapped = tf.math.floormod(x, mod)
    if signed:
        sign_bit = float(1 << (WL - 1))
        wrapped = tf.where(wrapped >= sign_bit, wrapped - mod, wrapped)
    return wrapped
