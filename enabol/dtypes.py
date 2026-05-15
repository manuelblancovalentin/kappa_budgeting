from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np


VALID_QMODES = {"AP_RND", "AP_TRN", "AP_RND_CONV", "AP_RND_ZERO", "AP_RND_MIN_INF", "AP_RND_INF"}
VALID_OMODES = {"AP_WRAP", "AP_SAT", "AP_SAT_ZERO", "AP_SAT_SYM"}


def _fraction_bits(WL: int, IWL: int) -> int:
    return max(WL - IWL, 0)


def _scale(F: int) -> float:
    return float(1 << F) if F > 0 else 1.0


def _round_by_qmode(x_scaled: np.ndarray, qmode: str) -> np.ndarray:
    if qmode in {"AP_TRN", "AP_TRN_ZERO"}:
        return np.trunc(x_scaled)
    if qmode in {"AP_RND", "AP_RND_CONV"}:
        return np.rint(x_scaled)
    if qmode == "AP_RND_ZERO":
        frac, _ = np.modf(x_scaled)
        tie = np.isclose(np.abs(frac), 0.5)
        out = np.where(x_scaled >= 0, np.floor(x_scaled + 0.5), np.ceil(x_scaled - 0.5))
        return np.where(tie, np.trunc(x_scaled), out)
    if qmode == "AP_RND_INF":
        frac, _ = np.modf(x_scaled)
        tie = np.isclose(np.abs(frac), 0.5)
        out = np.floor(x_scaled + 0.5)
        return np.where((x_scaled < 0) & tie, np.ceil(x_scaled - 0.5), out)
    if qmode == "AP_RND_MIN_INF":
        frac, _ = np.modf(x_scaled)
        tie = np.isclose(np.abs(frac), 0.5)
        out = np.ceil(x_scaled - 0.5)
        return np.where((x_scaled > 0) & tie, np.floor(x_scaled + 0.5), out)
    raise NotImplementedError(f"QMODE '{qmode}' is not implemented")


def _clip_int(v: np.ndarray, WL: int, signed: bool) -> np.ndarray:
    if signed:
        lo = -(1 << (WL - 1))
        hi = (1 << (WL - 1)) - 1
    else:
        lo = 0
        hi = (1 << WL) - 1
    return np.clip(v, lo, hi)


def _wrap_int(v: np.ndarray, WL: int, signed: bool) -> np.ndarray:
    mod = 1 << WL
    v = np.mod(v, mod)
    if signed:
        sign_bit = 1 << (WL - 1)
        v = np.where(v >= sign_bit, v - mod, v)
    return v


@dataclass(frozen=True)
class HLSDataType(ABC):
    dtype: str = field(init=False)

    @staticmethod
    def from_dtype(dtype: Any, **kwargs) -> "HLSDataType":
        if isinstance(dtype, HLSDataType):
            if kwargs:
                raise ValueError("Cannot override parameters on an HLSDataType instance")
            return dtype
        if not isinstance(dtype, str):
            raise TypeError(f"Expected dtype string or HLSDataType, got {type(dtype).__name__}")

        raw = dtype.strip()
        type_name = raw
        tokens: list[str] = []
        if "<" in raw:
            if not raw.endswith(">"):
                raise ValueError(f"Malformed dtype string: {raw}")
            type_name, args = raw.split("<", 1)
            tokens = [t.strip() for t in args[:-1].split(",") if t.strip()]

        parsed: dict[str, Any] = {}
        if type_name in {"ap_fixed", "ap_ufixed"}:
            if len(tokens) < 2:
                raise ValueError(f"{type_name} requires at least WL and IWL")
            parsed["WL"] = int(tokens[0], 0)
            parsed["IWL"] = int(tokens[1], 0)
            if len(tokens) >= 3:
                parsed["QMODE"] = tokens[2]
            if len(tokens) >= 4:
                parsed["OMODE"] = tokens[3]
            if len(tokens) >= 5:
                parsed["SAT_BITS"] = int(tokens[4], 0)
            if len(tokens) > 5:
                raise ValueError(f"Too many parameters for {type_name}: {tokens}")
        elif type_name in {"ap_int", "ap_uint"}:
            if len(tokens) != 1:
                raise ValueError(f"{type_name} requires exactly WL")
            parsed["WL"] = int(tokens[0], 0)
        else:
            raise ValueError(f"Unsupported data type: {type_name}")

        parsed.update(kwargs)
        if type_name == "ap_fixed":
            return ap_fixed(**parsed)
        if type_name == "ap_ufixed":
            return ap_ufixed(**parsed)
        if type_name == "ap_int":
            return ap_int(**parsed)
        if type_name == "ap_uint":
            return ap_uint(**parsed)
        raise ValueError(f"Unsupported data type: {type_name}")

    @abstractmethod
    def value_range(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def double_precision(self) -> "HLSDataType":
        pass

    @abstractmethod
    def signed(self) -> "HLSDataType":
        pass

    @abstractmethod
    def unsigned(self) -> "HLSDataType":
        pass

    @abstractmethod
    def __call__(self, value: Any, *, return_int: bool = False) -> Any:
        pass


@dataclass(frozen=True)
class ap_fixed(HLSDataType):
    WL: int = 16
    IWL: int = 6
    QMODE: str = "AP_TRN"
    OMODE: str = "AP_WRAP"
    SAT_BITS: int = 0
    dtype: str = field(init=False, default="ap_fixed")

    def __post_init__(self):
        if self.IWL > self.WL:
            raise ValueError("IWL must be <= WL")
        if self.QMODE not in VALID_QMODES:
            raise ValueError(f"QMODE must be one of {VALID_QMODES}")
        if self.OMODE not in VALID_OMODES:
            raise ValueError(f"OMODE must be one of {VALID_OMODES}")

    @property
    def fractional_bits(self) -> int:
        return _fraction_bits(self.WL, self.IWL)

    @property
    def quantum(self) -> float:
        return 2.0 ** (-self.fractional_bits)

    def value_range(self) -> Tuple[float, float]:
        max_val = 2.0 ** (self.IWL - 1) - self.quantum
        min_val = -(2.0 ** (self.IWL - 1))
        return float(min_val), float(max_val)

    def double_precision(self) -> HLSDataType:
        return ap_fixed(2 * self.WL, min(2 * self.IWL, 2 * self.WL), self.QMODE, self.OMODE, self.SAT_BITS)

    def signed(self) -> "ap_fixed":
        return self

    def unsigned(self) -> "ap_ufixed":
        return ap_ufixed(self.WL, self.IWL, self.QMODE, self.OMODE, self.SAT_BITS)

    def __call__(self, value: Any, *, return_int: bool = False) -> Any:
        return _quantize_np(value, self.WL, self.IWL, self.QMODE, self.OMODE, signed=True, return_int=return_int)

    def __repr__(self) -> str:
        return f"ap_fixed<{self.WL},{self.IWL},{self.QMODE},{self.OMODE}>"


@dataclass(frozen=True)
class ap_ufixed(HLSDataType):
    WL: int = 16
    IWL: int = 6
    QMODE: str = "AP_TRN"
    OMODE: str = "AP_WRAP"
    SAT_BITS: int = 0
    dtype: str = field(init=False, default="ap_ufixed")

    def __post_init__(self):
        if self.IWL > self.WL:
            raise ValueError("IWL must be <= WL")
        if self.QMODE not in VALID_QMODES:
            raise ValueError(f"QMODE must be one of {VALID_QMODES}")
        if self.OMODE not in VALID_OMODES:
            raise ValueError(f"OMODE must be one of {VALID_OMODES}")

    @property
    def fractional_bits(self) -> int:
        return _fraction_bits(self.WL, self.IWL)

    @property
    def quantum(self) -> float:
        return 2.0 ** (-self.fractional_bits)

    def value_range(self) -> Tuple[float, float]:
        return 0.0, float(2.0 ** self.IWL - self.quantum)

    def double_precision(self) -> HLSDataType:
        return ap_ufixed(2 * self.WL, min(2 * self.IWL, 2 * self.WL), self.QMODE, self.OMODE, self.SAT_BITS)

    def signed(self) -> ap_fixed:
        return ap_fixed(self.WL, self.IWL, self.QMODE, self.OMODE, self.SAT_BITS)

    def unsigned(self) -> "ap_ufixed":
        return self

    def __call__(self, value: Any, *, return_int: bool = False) -> Any:
        return _quantize_np(value, self.WL, self.IWL, self.QMODE, self.OMODE, signed=False, return_int=return_int)

    def __repr__(self) -> str:
        return f"ap_ufixed<{self.WL},{self.IWL},{self.QMODE},{self.OMODE}>"


@dataclass(frozen=True)
class ap_int(HLSDataType):
    WL: int = 16
    dtype: str = field(init=False, default="ap_int")

    def value_range(self) -> Tuple[float, float]:
        return float(-(2 ** (self.WL - 1))), float(2 ** (self.WL - 1) - 1)

    def double_precision(self) -> HLSDataType:
        return ap_int(2 * self.WL)

    def signed(self) -> "ap_int":
        return self

    def unsigned(self) -> "ap_uint":
        return ap_uint(self.WL)

    def __call__(self, value: Any, *, return_int: bool = False) -> Any:
        return _quantize_np(value, self.WL, self.WL, "AP_TRN", "AP_WRAP", signed=True, return_int=return_int)

    def __repr__(self) -> str:
        return f"ap_int<{self.WL}>"


@dataclass(frozen=True)
class ap_uint(HLSDataType):
    WL: int = 16
    dtype: str = field(init=False, default="ap_uint")

    def value_range(self) -> Tuple[float, float]:
        return 0.0, float(2 ** self.WL - 1)

    def double_precision(self) -> HLSDataType:
        return ap_uint(2 * self.WL)

    def signed(self) -> ap_int:
        return ap_int(self.WL)

    def unsigned(self) -> "ap_uint":
        return self

    def __call__(self, value: Any, *, return_int: bool = False) -> Any:
        return _quantize_np(value, self.WL, self.WL, "AP_TRN", "AP_WRAP", signed=False, return_int=return_int)

    def __repr__(self) -> str:
        return f"ap_uint<{self.WL}>"


def _quantize_np(
    value: Any,
    WL: int,
    IWL: int,
    QMODE: str,
    OMODE: str,
    *,
    signed: bool,
    return_int: bool = False,
) -> Any:
    arr = np.asarray(value, dtype=float)
    scale = _scale(_fraction_bits(WL, IWL))
    rounded = _round_by_qmode(arr * scale, QMODE)
    if OMODE == "AP_WRAP":
        clipped = _wrap_int(rounded, WL, signed=signed)
    elif OMODE in {"AP_SAT", "AP_SAT_ZERO", "AP_SAT_SYM"}:
        clipped = _clip_int(rounded, WL, signed=signed)
    else:
        raise NotImplementedError(f"OMODE '{OMODE}' is not implemented")
    out = clipped.astype(np.int64) if return_int else clipped / scale
    return out.item() if np.ndim(value) == 0 else out
