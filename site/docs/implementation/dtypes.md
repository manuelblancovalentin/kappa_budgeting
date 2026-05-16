---
status:
  - valid
  - inprogress
tags:
  - implementation
  - dtypes
  - precision
last_modified: 2026-05-15
source: "enabol/dtypes.py"
source_url: "https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/dtypes.py"
---
# 🗄️ Dtypes Module Reference
<PageMeta />
---

<TBox type="summary" title="What this page covers">
This page is a coder-facing reference for `enabol/dtypes.py`: HLS-style numeric descriptors, NumPy quantization behavior, rounding modes, overflow modes, and extension points for future precision experiments.
</TBox>

TensorFlow fake quantization lives in [`enabol/quantization.py`](https://github.com/manuelblancovalentin/kappa_budgeting/blob/main/enabol/quantization.py).

## Constants

### `VALID_QMODES`

```python
VALID_QMODES = {
    "AP_RND",
    "AP_TRN",
    "AP_RND_CONV",
    "AP_RND_ZERO",
    "AP_RND_MIN_INF",
    "AP_RND_INF",
}
```

Supported rounding modes. These names mirror HLS `ap_fixed` conventions.

### `VALID_OMODES`

```python
VALID_OMODES = {
    "AP_WRAP",
    "AP_SAT",
    "AP_SAT_ZERO",
    "AP_SAT_SYM",
}
```

Supported overflow modes. The first quantization ablations should use `AP_SAT` because saturation is easier to interpret than wraparound.

## Helper Functions

These helpers are internal but important if you need to extend dtype behavior.

### `_fraction_bits(WL, IWL)`

```python
def _fraction_bits(WL: int, IWL: int) -> int:
    return max(WL - IWL, 0)
```

Computes:

```math
F = WL - IWL.
```

If `IWL >= WL`, the type has no fractional bits.

### `_scale(F)`

```python
def _scale(F: int) -> float:
    return float(1 << F) if F > 0 else 1.0
```

Returns:

```math
2^F.
```

Used to move between real values and integer fixed-point storage.

### `_round_by_qmode(x_scaled, qmode)`

```python
def _round_by_qmode(x_scaled: np.ndarray, qmode: str) -> np.ndarray:
    ...
```

Applies the selected HLS-style rounding mode in the scaled integer domain.

Examples:

```python
AP_TRN       -> truncate toward zero
AP_RND       -> NumPy round-to-nearest
AP_RND_CONV  -> same as NumPy rint / banker rounding
AP_RND_ZERO  -> ties toward zero
```

If a new rounding mode is needed, add it here and mirror it in `quantization.py` for TensorFlow tensors.

### `_clip_int(v, WL, signed)`

```python
def _clip_int(v: np.ndarray, WL: int, signed: bool) -> np.ndarray:
    ...
```

Clips integer-domain values to the representable integer range.

Signed:

```math
[-2^{WL-1}, 2^{WL-1}-1].
```

Unsigned:

```math
[0, 2^{WL}-1].
```

### `_wrap_int(v, WL, signed)`

```python
def _wrap_int(v: np.ndarray, WL: int, signed: bool) -> np.ndarray:
    ...
```

Applies modulo wraparound in the integer domain. For signed types, values above the sign bit are interpreted as negative two's-complement values.

### `_quantize_np(...)`

```python
def _quantize_np(
    value,
    WL,
    IWL,
    QMODE,
    OMODE,
    *,
    signed,
    return_int=False,
):
    ...
```

Shared NumPy quantization backend. Steps:

1. Convert input to float NumPy array.
2. Scale by `2^F`.
3. Round according to `QMODE`.
4. Overflow according to `OMODE`.
5. Return either raw integer storage or dequantized real values.

Usage through a dtype object:

```python
t = dtypes.ap_fixed(8, 3, "AP_RND", "AP_SAT")
real_values = t([0.1, 0.2, 12.0])
raw_ints = t([0.1, 0.2, 12.0], return_int=True)
```

## `HLSDataType`

```python
@dataclass(frozen=True)
class HLSDataType(ABC):
    dtype: str = field(init=False)
```

Abstract base class for all dtype descriptors.

### `from_dtype(dtype, **kwargs)`

```python
@staticmethod
def from_dtype(dtype: Any, **kwargs) -> "HLSDataType":
    ...
```

Factory method. Accepts either an existing dtype object or an HLS-style string.

Examples:

```python
from enabol.dtypes import HLSDataType

t0 = HLSDataType.from_dtype("ap_fixed<12,4,AP_RND,AP_SAT>")
t1 = HLSDataType.from_dtype("ap_ufixed<10,3>")
t2 = HLSDataType.from_dtype("ap_int<16>")
```

Used by `PrecisionDict`, so YAML/string configs can become dtype objects automatically.

### Abstract Methods

Every dtype must implement:

```python
value_range()
double_precision()
signed()
unsigned()
__call__(value, return_int=False)
```

These methods let the rest of the package query rails, construct related dtypes, and quantize values without caring about the concrete dtype class.

## `ap_fixed`

```python
@dataclass(frozen=True)
class ap_fixed(HLSDataType):
    WL: int = 16
    IWL: int = 6
    QMODE: str = "AP_TRN"
    OMODE: str = "AP_WRAP"
    SAT_BITS: int = 0
```

Signed fixed-point type.

### `__post_init__()`

Validates:

- `IWL <= WL`,
- `QMODE` is supported,
- `OMODE` is supported.

### `fractional_bits`

```python
@property
def fractional_bits(self) -> int:
    return WL - IWL
```

### `quantum`

```python
@property
def quantum(self) -> float:
    return 2.0 ** (-self.fractional_bits)
```

This is the minimum representable spacing.

### `value_range()`

```python
def value_range(self) -> Tuple[float, float]:
    ...
```

Returns:

```math
[-2^{IWL-1}, 2^{IWL-1}-2^{-F}].
```

### `double_precision()`

Returns a wider signed fixed-point dtype with doubled word length and integer length.

### `signed()` and `unsigned()`

`signed()` returns itself. `unsigned()` returns the corresponding `ap_ufixed`.

### `__call__(value, return_int=False)`

Quantizes NumPy-compatible values.

```python
t = dtypes.ap_fixed(12, 4, "AP_RND", "AP_SAT")
xq = t([0.1, 0.2, 32.0])
```

## `ap_ufixed`

Unsigned fixed-point type. Same fields as `ap_fixed`, but the range is:

```math
[0, 2^{IWL}-2^{-F}].
```

Use this for nonnegative values such as ReLU activations when the sign bit is not needed.

Example:

```python
act_t = dtypes.ap_ufixed(10, 4, "AP_RND", "AP_SAT")
```

## `ap_int`

```python
@dataclass(frozen=True)
class ap_int(HLSDataType):
    WL: int = 16
```

Signed integer type. Range:

```math
[-2^{WL-1}, 2^{WL-1}-1].
```

Use this for integer counters or raw integer-domain tests.

## `ap_uint`

Unsigned integer type. Range:

```math
[0, 2^{WL}-1].
```

## Extension Checklist

When extending this file:

1. Add NumPy behavior in `_round_by_qmode()` or `_quantize_np()`.
2. Add matching TensorFlow behavior in `quantization.py`.
3. Add `value_range()` if a new dtype class is introduced.
4. Keep dtype objects immutable with `frozen=True`.
5. Avoid putting experiment-specific policy in this file. This module should describe numeric formats only.
