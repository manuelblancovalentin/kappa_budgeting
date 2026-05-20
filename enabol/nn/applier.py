from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from ..precision import PrecisionDict
from ..quantization import quantize_tensor, rail_stats


@dataclass(frozen=True)
class ApplyResult:
    update_saturation_fraction_max: float
    update_underflow_fraction_max: float


class UpdateApplier:
    """Applies controller-approved updates to model variables."""

    def __init__(
        self,
        *,
        layer_and_field_for_variable: Callable[[tf.Variable], tuple[str, str]],
        quantize_variable_storage: Callable[[tf.Variable, PrecisionDict], None],
    ):
        self.layer_and_field_for_variable = layer_and_field_for_variable
        self.quantize_variable_storage = quantize_variable_storage

    def describe(self) -> str:
        return "UpdateApplier()"

    def __repr__(self) -> str:
        return self.describe()

    def apply(
        self,
        *,
        variables: list[tf.Variable],
        updates: list[tf.Tensor],
        precision: PrecisionDict | None,
    ) -> ApplyResult:
        update_saturation_max = 0.0
        update_underflow_max = 0.0

        for var, update in zip(variables, updates):
            if precision is None:
                var.assign_add(update)
                continue

            layer_name, _ = self.layer_and_field_for_variable(var)
            update_dtype = precision.dtype(layer_name, "update")
            stats = rail_stats(update.numpy(), update_dtype)
            update_saturation_max = max(update_saturation_max, stats.saturation_fraction)
            update_underflow_max = max(update_underflow_max, stats.underflow_fraction)

            update_q = quantize_tensor(update, update_dtype, ste=False)
            var.assign_add(update_q)
            self.quantize_variable_storage(var, precision)

        return ApplyResult(
            update_saturation_fraction_max=update_saturation_max,
            update_underflow_fraction_max=update_underflow_max,
        )
