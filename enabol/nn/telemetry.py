from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from ..string import print_table
from ..utils import tensor_l2_norm


@dataclass(frozen=True)
class SensorReadout:
    theta_norm: tf.Tensor
    grad_norm: tf.Tensor
    raw_update_norm: tf.Tensor
    curvature_proxy: tf.Tensor
    curvature_ema: tf.Tensor
    update_quantum: tf.Tensor


@dataclass(frozen=True)
class UpdateGeometry:
    actual_update_norm: tf.Tensor
    update_cosine: tf.Tensor
    update_angle_rad: tf.Tensor
    update_radius_ratio: tf.Tensor


PROFILE_FIELDS: dict[str, tuple[str, ...]] = {
    "core": (
        "loss",
        "rmse",
        "grad_norm",
        "raw_update_norm",
        "actual_update_norm",
        "curvature_proxy",
        "curvature_ema",
        "alpha",
        "alpha_would",
        "eta_eff",
        "diverged",
    ),
    "geometry": (
        "theta_norm",
        "update_cosine",
        "update_angle_rad",
        "update_radius_ratio",
    ),
    "stability": (
        "curvature_for_control",
        "hessian_lambda_max",
        "hessian_lambda_min",
        "hessian_spectral_norm",
        "stability_margin_lambda_raw",
        "stability_margin_lambda_ctrl",
        "stability_margin_norm_raw",
        "stability_margin_norm_ctrl",
        "spectral_radius_raw",
        "spectral_radius_ctrl",
    ),
    "teacher": (
        "forward_gain_spectral",
        "weight_error_fro",
    ),
    "finite": (
        "loss_is_finite",
        "grad_is_finite",
        "theta_is_finite_before",
        "theta_is_finite_after",
    ),
    "quantization": (
        "weight_saturation_fraction_max",
        "weight_near_rail_fraction_max",
        "gradient_saturation_fraction_max",
        "gradient_near_rail_fraction_max",
        "update_saturation_fraction_max",
        "update_underflow_fraction_max",
    ),
    "controller_bounds": (
        "alpha_min_bound",
        "alpha_max_bound",
        "controller_feasible",
    ),
}


METRIC_DESCRIPTIONS: dict[str, str] = {
    "loss": "Training loss for the current batch.",
    "rmse": "Root mean-square output error.",
    "grad_norm": "Global gradient norm.",
    "raw_update_norm": "Norm of the optimizer-proposed update before control.",
    "actual_update_norm": "Norm of the update that actually reached the variables.",
    "curvature_proxy": "Online curvature proxy from gradient change over parameter change.",
    "curvature_ema": "Smoothed curvature proxy.",
    "alpha": "Controller throttle applied to the raw update.",
    "alpha_would": "Algebraic safe throttle implied by the current curvature estimate.",
    "eta_eff": "Effective learning rate after throttling.",
    "diverged": "True when finite checks fail and training stops.",
    "theta_norm": "Global parameter norm before the update.",
    "update_cosine": "Cosine between raw and actual update directions.",
    "update_angle_rad": "Angle between raw and actual update directions.",
    "update_radius_ratio": "Actual update norm divided by raw update norm.",
    "curvature_for_control": "Curvature signal passed into the controller law.",
    "hessian_lambda_max": "Largest analytic Hessian eigenvalue for the current batch.",
    "hessian_lambda_min": "Smallest analytic Hessian eigenvalue for the current batch.",
    "hessian_spectral_norm": "Analytic Hessian spectral norm.",
    "stability_margin_lambda_raw": "Raw eta times Hessian lambda max.",
    "stability_margin_lambda_ctrl": "Controlled eta times Hessian lambda max.",
    "stability_margin_norm_raw": "Raw eta times Hessian spectral norm.",
    "stability_margin_norm_ctrl": "Controlled eta times Hessian spectral norm.",
    "spectral_radius_raw": "Spectral radius of the raw linearized SGD update map.",
    "spectral_radius_ctrl": "Spectral radius of the controlled linearized update map.",
    "forward_gain_spectral": "Spectral norm of the first Dense kernel in math convention.",
    "weight_error_fro": "Frobenius error against reference_A when supplied.",
    "loss_is_finite": "Finite check for the scalar loss.",
    "grad_is_finite": "Finite check for the flattened gradient.",
    "theta_is_finite_before": "Finite check for parameters before the update.",
    "theta_is_finite_after": "Finite check for parameters after the update.",
    "weight_saturation_fraction_max": "Maximum storage saturation fraction over weights and biases.",
    "weight_near_rail_fraction_max": "Maximum near-rail fraction over weights and biases.",
    "gradient_saturation_fraction_max": "Maximum gradient saturation fraction.",
    "gradient_near_rail_fraction_max": "Maximum gradient near-rail fraction.",
    "update_saturation_fraction_max": "Maximum update saturation fraction.",
    "update_underflow_fraction_max": "Maximum update underflow fraction.",
    "alpha_min_bound": "Lower alpha bound active for this controller step.",
    "alpha_max_bound": "Upper alpha bound active for this controller step.",
    "controller_feasible": "True when dynamic lower/upper alpha bounds are feasible.",
}


def _metric_profile(field: str) -> str:
    for profile, fields in PROFILE_FIELDS.items():
        if field in fields:
            return profile
    return "custom"


class MetricsListGuide:
    """Printable reference for all known metric fields."""

    def describe(self) -> str:
        rows: dict[str, Any] = {}
        for profile, fields in PROFILE_FIELDS.items():
            rows[profile] = None
            for field in fields:
                rows[field] = METRIC_DESCRIPTIONS.get(field, "")
        return print_table(rows, "Metrics Reference", key_header="Metric")

    def __repr__(self) -> str:
        return self.describe()


ALL_METRIC_FIELDS = (
    "loss",
    "rmse",
    "theta_norm",
    "grad_norm",
    "raw_update_norm",
    "actual_update_norm",
    "update_cosine",
    "update_angle_rad",
    "update_radius_ratio",
    "curvature_proxy",
    "curvature_ema",
    "alpha",
    "alpha_would",
    "alpha_min_bound",
    "alpha_max_bound",
    "controller_feasible",
    "curvature_for_control",
    "eta_eff",
    "forward_gain_spectral",
    "hessian_lambda_max",
    "hessian_lambda_min",
    "hessian_spectral_norm",
    "stability_margin_lambda_raw",
    "stability_margin_lambda_ctrl",
    "stability_margin_norm_raw",
    "stability_margin_norm_ctrl",
    "spectral_radius_raw",
    "spectral_radius_ctrl",
    "weight_error_fro",
    "loss_is_finite",
    "grad_is_finite",
    "theta_is_finite_before",
    "theta_is_finite_after",
    "weight_saturation_fraction_max",
    "weight_near_rail_fraction_max",
    "gradient_saturation_fraction_max",
    "gradient_near_rail_fraction_max",
    "update_saturation_fraction_max",
    "update_underflow_fraction_max",
    "diverged",
)


@dataclass(frozen=True)
class MetricsConfig:
    """Selects which instrumented-training metrics are recorded."""

    profiles: tuple[str, ...] = ("core",)
    reference_A: Any = None
    compute_analytic_hessian: bool = True
    include_fields: tuple[str, ...] = ()
    exclude_fields: tuple[str, ...] = ()
    legacy_full: bool = False

    @classmethod
    def all(cls) -> "MetricsConfig":
        return cls(
            profiles=tuple(PROFILE_FIELDS),
            compute_analytic_hessian=True,
            legacy_full=True,
        )

    @property
    def fields(self) -> tuple[str, ...]:
        if self.legacy_full:
            base = ALL_METRIC_FIELDS
        else:
            unknown = tuple(profile for profile in self.profiles if profile not in PROFILE_FIELDS)
            if unknown:
                raise ValueError(
                    f"Unknown metric profile(s): {unknown}. "
                    f"Known profiles: {tuple(PROFILE_FIELDS)}"
                )
            base = tuple(
                field
                for profile in self.profiles
                for field in PROFILE_FIELDS[profile]
            )

        ordered = list(dict.fromkeys((*base, *self.include_fields)))
        excluded = set(self.exclude_fields)
        return tuple(field for field in ordered if field not in excluded)

    def includes(self, *fields: str) -> bool:
        selected = set(self.fields)
        return any(field in selected for field in fields)

    @property
    def requires_teacher(self) -> bool:
        return self.includes("forward_gain_spectral", "weight_error_fro")

    @property
    def requires_hessian(self) -> bool:
        return self.compute_analytic_hessian and self.includes(
            "hessian_lambda_max",
            "hessian_lambda_min",
            "hessian_spectral_norm",
            "stability_margin_lambda_raw",
            "stability_margin_lambda_ctrl",
            "stability_margin_norm_raw",
            "stability_margin_norm_ctrl",
            "spectral_radius_raw",
            "spectral_radius_ctrl",
        )

    @property
    def requires_quantization_metrics(self) -> bool:
        return self.includes(
            "weight_saturation_fraction_max",
            "weight_near_rail_fraction_max",
            "gradient_saturation_fraction_max",
            "gradient_near_rail_fraction_max",
        )

    def summary(self) -> str:
        mode = "legacy-full" if self.legacy_full else ", ".join(self.profiles)
        parts = [
            f"profiles={mode}",
            f"fields={len(self.fields)}",
            f"analytic_hessian={self.compute_analytic_hessian and self.requires_hessian}",
        ]
        if self.reference_A is not None:
            shape = getattr(self.reference_A, "shape", None)
            parts.append(f"reference_A={tuple(shape) if shape is not None else type(self.reference_A).__name__}")
        return " | ".join(parts)

    @property
    def reference(self) -> MetricsListGuide:
        return MetricsListGuide()

    def describe(self) -> str:
        profile_text = "legacy-full" if self.legacy_full else ", ".join(self.profiles)
        reference_shape = getattr(self.reference_A, "shape", None)
        settings = {
            "profiles": profile_text,
            "fields": len(self.fields),
            "compute_analytic_hessian": self.compute_analytic_hessian,
            "requires_hessian": self.requires_hessian,
            "requires_teacher": self.requires_teacher,
            "requires_quantization_metrics": self.requires_quantization_metrics,
            "reference_A": tuple(reference_shape) if reference_shape is not None else self.reference_A is not None,
            "include_fields": self.include_fields or "[]",
            "exclude_fields": self.exclude_fields or "[]",
        }
        return print_table(settings, "MetricsConfig", key_header="Setting")

    def __repr__(self) -> str:
        return self.describe()


class CurvatureSensor:
    """Tensor-native online sensor for controller-facing signals."""

    def __init__(self, *, ema_rho: float = 0.05, eps: float = 1e-12):
        self.ema_rho = float(ema_rho)
        self.eps = float(eps)
        self.prev_theta: tf.Tensor | None = None
        self.prev_grad: tf.Tensor | None = None
        self.curvature_ema = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def reset(self) -> None:
        self.prev_theta = None
        self.prev_grad = None
        self.curvature_ema.assign(0.0)

    def describe(self) -> str:
        has_previous = self.prev_theta is not None and self.prev_grad is not None
        return "\n".join(
            [
                "CurvatureSensor(",
                f"  ema_rho: {self.ema_rho}",
                f"  eps: {self.eps}",
                f"  has_previous: {has_previous}",
                f"  curvature_ema: {float(self.curvature_ema.numpy()):.6g}",
                ")",
            ]
        )

    def __repr__(self) -> str:
        return self.describe()

    def observe(
        self,
        *,
        theta: tf.Tensor,
        grad: tf.Tensor,
        raw_update: tf.Tensor,
        update_quantum: float | tf.Tensor,
    ) -> SensorReadout:
        theta = tf.cast(theta, tf.float32)
        grad = tf.cast(grad, tf.float32)
        raw_update = tf.cast(raw_update, tf.float32)

        if self.prev_theta is None or self.prev_grad is None:
            curvature_proxy = tf.constant(0.0, dtype=tf.float32)
        else:
            d_grad = grad - self.prev_grad
            d_theta = theta - self.prev_theta
            curvature_proxy = (
                tensor_l2_norm(d_grad, self.eps)
                / (tensor_l2_norm(d_theta, self.eps) + tf.cast(self.eps, tf.float32))
            )

        curvature_ema = (
            (1.0 - self.ema_rho) * self.curvature_ema
            + self.ema_rho * curvature_proxy
        )
        self.curvature_ema.assign(curvature_ema)
        self.prev_theta = tf.identity(theta)
        self.prev_grad = tf.identity(grad)

        return SensorReadout(
            theta_norm=tensor_l2_norm(theta, self.eps),
            grad_norm=tensor_l2_norm(grad, self.eps),
            raw_update_norm=tensor_l2_norm(raw_update, self.eps),
            curvature_proxy=curvature_proxy,
            curvature_ema=curvature_ema,
            update_quantum=tf.cast(update_quantum, tf.float32),
        )


def update_geometry(
    *,
    raw_update: tf.Tensor,
    actual_update: tf.Tensor,
    eps: float = 1e-12,
) -> UpdateGeometry:
    raw_update = tf.cast(raw_update, tf.float32)
    actual_update = tf.cast(actual_update, tf.float32)

    actual_update_norm = tensor_l2_norm(actual_update, eps)
    raw_update_norm_plain = tf.norm(raw_update)
    actual_update_norm_plain = tf.norm(actual_update)
    update_dot = tf.reduce_sum(actual_update * raw_update)
    update_denom = raw_update_norm_plain * actual_update_norm_plain
    update_cosine = tf.where(
        update_denom > eps,
        update_dot / update_denom,
        tf.constant(1.0, dtype=tf.float32),
    )
    update_cosine = tf.clip_by_value(update_cosine, -1.0, 1.0)
    update_angle_rad = tf.acos(update_cosine)
    update_radius_ratio = tf.where(
        raw_update_norm_plain > eps,
        actual_update_norm_plain / raw_update_norm_plain,
        tf.constant(0.0, dtype=tf.float32),
    )

    return UpdateGeometry(
        actual_update_norm=actual_update_norm,
        update_cosine=update_cosine,
        update_angle_rad=update_angle_rad,
        update_radius_ratio=update_radius_ratio,
    )


class HistoryRecorder:
    """Small append-only recorder that materializes NumPy arrays at the end."""

    DEFAULT_KEYS = ALL_METRIC_FIELDS

    def __init__(self, keys: tuple[str, ...] | None = None):
        self.history: dict[str, list[float]] = {
            key: [] for key in (keys or self.DEFAULT_KEYS)
        }

    def describe(self) -> str:
        n_steps = 0
        if self.history:
            n_steps = len(next(iter(self.history.values())))
        return "\n".join(
            [
                "HistoryRecorder(",
                f"  keys: {len(self.history)}",
                f"  steps: {n_steps}",
                ")",
            ]
        )

    def __repr__(self) -> str:
        return self.describe()

    def append(self, **values: Any) -> None:
        for key in self.history:
            value = values.get(key, np.nan)
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            self.history[key].append(float(value))

    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.asarray(values) for key, values in self.history.items()}
