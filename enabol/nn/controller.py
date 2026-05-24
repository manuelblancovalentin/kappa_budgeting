from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import tensorflow as tf


def _tensor_value(value: tf.Tensor | float) -> float:
    if isinstance(value, tf.Tensor):
        try:
            return float(value.numpy())
        except Exception:
            return float("nan")
    return float(value)


def _reference_summary(reference: Any) -> str:
    if reference is None:
        return "None"
    shape = getattr(reference, "shape", None)
    dtype = getattr(reference, "dtype", None)
    if shape is not None:
        if dtype is not None:
            return f"shape={tuple(shape)}, dtype={dtype}"
        return f"shape={tuple(shape)}"
    return type(reference).__name__


def _format_float(value: float) -> str:
    return f"{value:.6g}"


@dataclass(frozen=True)
class ControllerStep:
    """Result of one controller update."""

    alpha: tf.Tensor
    alpha_would: tf.Tensor
    curvature_for_control: tf.Tensor
    eta_eff: tf.Tensor
    alpha_min: tf.Tensor
    alpha_max: tf.Tensor
    feasible: tf.Tensor


@dataclass(frozen=True)
class ControlledUpdate:
    """Controller output: scaled updates plus controller diagnostics."""

    updates: list[tf.Tensor]
    update_flat: tf.Tensor
    step: ControllerStep


class BaseController(ABC):
    """Base class for update controllers used by instrumented training loops.

    Controllers sit after the optimizer proposes a raw update and before the
    update is applied. Global-throttle controllers return a scalar alpha that
    multiplies the full update, preserving the raw update direction before
    fixed-point quantization effects.
    """

    controller_id = "CTRL-BASE"
    direction_preserving = True
    LOG_FIELDS = (
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

    def __init__(
        self,
        *,
        chi: float = 1.5,
        eps: float = 1e-12,
        curvature_ema_rho: float = 0.05,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
        use_ema_max: bool = True,
        name: str | None = None,
    ):
        self.chi = float(chi)
        self.eps = float(eps)
        self.curvature_ema_rho = float(curvature_ema_rho)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.use_ema_max = bool(use_ema_max)
        self.name = name or self.controller_id

    def reset(self) -> None:
        """Reset controller state before a fresh run."""

    @property
    def log_fields(self) -> tuple[str, ...]:
        return self.LOG_FIELDS

    def equations(self) -> tuple[str, ...]:
        curvature_source = "max(C, ⟨C⟩)" if self.use_ema_max else "C"
        return (
            "C = ǁĠǁ / (ǁθ̇ǁ + ε)",
            f"Ĉ = {curvature_source}",
            "α = controller_law(Ĉ)",
            "θ̇ = α θ̇_raw",
        )

    def parameter_summary(self, *, learning_rate: float | None = None) -> str:
        parts = [
            f"χ={_format_float(self.chi)}",
            f"ρ={_format_float(self.curvature_ema_rho)}",
            f"ε={_format_float(self.eps)}",
            f"α∈[{_format_float(self.alpha_min)}, {_format_float(self.alpha_max)}]",
        ]
        if learning_rate is not None:
            parts.insert(1, f"η={_format_float(float(learning_rate))}")
        return " | ".join(parts)

    def summary(self, *, learning_rate: float | None = None) -> str:
        lines = [self.controller_id]
        lines.extend(self.equations())
        lines.append(self.parameter_summary(learning_rate=learning_rate))
        return "\n".join(lines)

    def describe(self) -> str:
        lines = [f"{self.controller_id}("]
        lines.extend(f"  {equation}" for equation in self.equations())
        lines.append(f"  {self.parameter_summary()}")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()

    def curvature_signal(
        self,
        curvature_proxy: tf.Tensor | float,
        curvature_ema: tf.Tensor | float,
    ) -> tf.Tensor:
        proxy = tf.cast(curvature_proxy, tf.float32)
        ema = tf.cast(curvature_ema, tf.float32)
        if self.use_ema_max:
            return tf.maximum(proxy, ema)
        return proxy

    def algebraic_safe_alpha(
        self,
        *,
        learning_rate: tf.Tensor | float,
        curvature_for_control: tf.Tensor | float,
    ) -> tf.Tensor:
        eta = tf.cast(learning_rate, tf.float32)
        curvature = tf.cast(curvature_for_control, tf.float32)
        safe = tf.where(
            curvature <= self.eps,
            tf.ones_like(curvature, dtype=tf.float32),
            tf.cast(self.chi, tf.float32) / (eta * (curvature + tf.cast(self.eps, tf.float32))),
        )
        return tf.clip_by_value(
            safe,
            tf.cast(self.alpha_min, tf.float32),
            tf.cast(self.alpha_max, tf.float32),
        )

    def step(
        self,
        *,
        learning_rate: tf.Tensor | float,
        curvature_proxy: tf.Tensor | float,
        curvature_ema: tf.Tensor | float,
        grad_norm: tf.Tensor | float | None = None,
        update_quantum: tf.Tensor | float | None = None,
        **_: Any,
    ) -> ControllerStep:
        eta = tf.cast(learning_rate, tf.float32)
        curvature = self.curvature_signal(curvature_proxy, curvature_ema)
        alpha_would = self.algebraic_safe_alpha(
            learning_rate=eta,
            curvature_for_control=curvature,
        )
        alpha = self._alpha(
            learning_rate=eta,
            curvature_for_control=curvature,
            alpha_would=alpha_would,
            grad_norm=grad_norm,
            update_quantum=update_quantum,
        )
        alpha = tf.clip_by_value(
            tf.cast(alpha, tf.float32),
            tf.cast(self.alpha_min, tf.float32),
            tf.cast(self.alpha_max, tf.float32),
        )
        return ControllerStep(
            alpha=alpha,
            alpha_would=alpha_would,
            curvature_for_control=curvature,
            eta_eff=alpha * eta,
            alpha_min=tf.cast(self.alpha_min, tf.float32),
            alpha_max=tf.cast(self.alpha_max, tf.float32),
            feasible=tf.constant(True),
        )

    def control(
        self,
        *,
        raw_updates: list[tf.Tensor],
        raw_update_flat: tf.Tensor,
        learning_rate: tf.Tensor | float,
        curvature_proxy: tf.Tensor | float,
        curvature_ema: tf.Tensor | float,
        grad_norm: tf.Tensor | float | None = None,
        update_quantum: tf.Tensor | float | None = None,
        **kwargs: Any,
    ) -> ControlledUpdate:
        step = self.step(
            learning_rate=learning_rate,
            curvature_proxy=curvature_proxy,
            curvature_ema=curvature_ema,
            grad_norm=grad_norm,
            update_quantum=update_quantum,
            **kwargs,
        )
        updates = [step.alpha * update for update in raw_updates]
        return ControlledUpdate(
            updates=updates,
            update_flat=step.alpha * raw_update_flat,
            step=step,
        )

    @abstractmethod
    def _alpha(
        self,
        *,
        learning_rate: tf.Tensor,
        curvature_for_control: tf.Tensor,
        alpha_would: tf.Tensor,
        grad_norm: tf.Tensor | float | None,
        update_quantum: tf.Tensor | float | None,
    ) -> tf.Tensor:
        pass


class NoController(BaseController):
    """Baseline controller that applies the raw update unchanged."""

    controller_id = "CTRL-NONE"

    def equations(self) -> tuple[str, ...]:
        return (
            "α = 1",
            "θ̇ = θ̇_raw",
        )

    def _alpha(
        self,
        *,
        learning_rate: tf.Tensor,
        curvature_for_control: tf.Tensor,
        alpha_would: tf.Tensor,
        grad_norm: tf.Tensor | float | None,
        update_quantum: tf.Tensor | float | None,
    ) -> tf.Tensor:
        return tf.ones_like(alpha_would, dtype=tf.float32)


class GlobalThrottleOrder0Controller(BaseController):
    """CTRL-GT-ORDER-0: algebraic safe-gain global throttle."""

    controller_id = "CTRL-GT-ORDER-0"

    def equations(self) -> tuple[str, ...]:
        curvature_source = "max(C, ⟨C⟩)" if self.use_ema_max else "C"
        return (
            "C = ǁĠǁ / (ǁθ̇ǁ + ε)",
            f"Ĉ = {curvature_source}",
            "α = clip(χ / (η(Ĉ + ε)), α_min, α_max)",
            "θ̇ = α θ̇_raw",
        )

    def _alpha(
        self,
        *,
        learning_rate: tf.Tensor,
        curvature_for_control: tf.Tensor,
        alpha_would: tf.Tensor,
        grad_norm: tf.Tensor | float | None,
        update_quantum: tf.Tensor | float | None,
    ) -> tf.Tensor:
        return alpha_would


class GlobalThrottleOrder1Controller(BaseController):
    """CTRL-GT-ORDER-1: first-order alpha-state global throttle."""

    controller_id = "CTRL-GT-ORDER-1"

    def __init__(self, *, k_alpha: float = 0.1, initial_alpha: float = 0.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.k_alpha = float(k_alpha)
        self.initial_alpha = float(initial_alpha)
        self._alpha_state = tf.Variable(initial_alpha, trainable=False, dtype=tf.float32)

    def reset(self) -> None:
        self._alpha_state.assign(self.initial_alpha)

    def equations(self) -> tuple[str, ...]:
        curvature_source = "max(C, ⟨C⟩)" if self.use_ema_max else "C"
        return (
            "C = ǁĠǁ / (ǁθ̇ǁ + ε)",
            f"Ĉ = {curvature_source}",
            "m = χ - ηαĈ",
            "α̇ = k_α m",
            "θ̇ = α θ̇_raw",
        )

    def parameter_summary(self, *, learning_rate: float | None = None) -> str:
        return " | ".join(
            [
                super().parameter_summary(learning_rate=learning_rate),
                f"k_α={_format_float(self.k_alpha)}",
                f"α_state={_format_float(_tensor_value(self._alpha_state))}",
            ]
        )

    def _alpha(
        self,
        *,
        learning_rate: tf.Tensor,
        curvature_for_control: tf.Tensor,
        alpha_would: tf.Tensor,
        grad_norm: tf.Tensor | float | None,
        update_quantum: tf.Tensor | float | None,
    ) -> tf.Tensor:
        margin_error = (
            tf.cast(self.chi, tf.float32)
            - learning_rate * self._alpha_state * curvature_for_control
        )
        next_alpha = self._alpha_state + tf.cast(self.k_alpha, tf.float32) * margin_error
        next_alpha = tf.clip_by_value(
            next_alpha,
            tf.cast(self.alpha_min, tf.float32),
            tf.cast(self.alpha_max, tf.float32),
        )
        self._alpha_state.assign(next_alpha)
        return next_alpha


class GlobalThrottleOrder2Controller(BaseController):
    """CTRL-GT-ORDER-2: damped second-order alpha controller."""

    controller_id = "CTRL-GT-ORDER-2"

    def __init__(
        self,
        *,
        k_alpha: float = 0.1,
        beta: float = 0.8,
        initial_alpha: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.k_alpha = float(k_alpha)
        self.beta = float(beta)
        self.initial_alpha = float(initial_alpha)
        self._alpha_state = tf.Variable(initial_alpha, trainable=False, dtype=tf.float32)
        self._velocity = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def reset(self) -> None:
        self._alpha_state.assign(self.initial_alpha)
        self._velocity.assign(0.0)

    def equations(self) -> tuple[str, ...]:
        curvature_source = "max(C, ⟨C⟩)" if self.use_ema_max else "C"
        return (
            "C = ǁĠǁ / (ǁθ̇ǁ + ε)",
            f"Ĉ = {curvature_source}",
            "m = χ - ηαĈ",
            "v̇ = βv + k_αm",
            "α̇ = v",
            "θ̇ = α θ̇_raw",
        )

    def parameter_summary(self, *, learning_rate: float | None = None) -> str:
        return " | ".join(
            [
                super().parameter_summary(learning_rate=learning_rate),
                f"k_α={_format_float(self.k_alpha)}",
                f"β={_format_float(self.beta)}",
                f"α_state={_format_float(_tensor_value(self._alpha_state))}",
                f"v={_format_float(_tensor_value(self._velocity))}",
            ]
        )

    def _alpha(
        self,
        *,
        learning_rate: tf.Tensor,
        curvature_for_control: tf.Tensor,
        alpha_would: tf.Tensor,
        grad_norm: tf.Tensor | float | None,
        update_quantum: tf.Tensor | float | None,
    ) -> tf.Tensor:
        margin_error = (
            tf.cast(self.chi, tf.float32)
            - learning_rate * self._alpha_state * curvature_for_control
        )
        velocity = (
            tf.cast(self.beta, tf.float32) * self._velocity
            + tf.cast(self.k_alpha, tf.float32) * margin_error
        )
        next_alpha = self._alpha_state + velocity
        next_alpha = tf.clip_by_value(
            next_alpha,
            tf.cast(self.alpha_min, tf.float32),
            tf.cast(self.alpha_max, tf.float32),
        )
        self._velocity.assign(velocity)
        self._alpha_state.assign(next_alpha)
        return next_alpha


class QuantizationAwareOrder2Controller(GlobalThrottleOrder2Controller):
    """CTRL-GT-ORDER-2-QA: second-order throttle with update-floor bounds."""

    controller_id = "CTRL-GT-ORDER-2-QA"

    def equations(self) -> tuple[str, ...]:
        return (
            *super().equations(),
            "α_min,q = q_update / (ηǁGǁ + ε)",
            "α = clip(α, α_min,q, α_max,safe)",
        )

    def step(
        self,
        *,
        learning_rate: tf.Tensor | float,
        curvature_proxy: tf.Tensor | float,
        curvature_ema: tf.Tensor | float,
        grad_norm: tf.Tensor | float | None = None,
        update_quantum: tf.Tensor | float | None = None,
        **kwargs: Any,
    ) -> ControllerStep:
        eta = tf.cast(learning_rate, tf.float32)
        curvature = self.curvature_signal(curvature_proxy, curvature_ema)
        alpha_max_safe = self.algebraic_safe_alpha(
            learning_rate=eta,
            curvature_for_control=curvature,
        )

        if grad_norm is None or update_quantum is None:
            dynamic_alpha_min = tf.cast(self.alpha_min, tf.float32)
        else:
            dynamic_alpha_min = (
                tf.cast(update_quantum, tf.float32)
                / (eta * tf.cast(grad_norm, tf.float32) + tf.cast(self.eps, tf.float32))
            )
            dynamic_alpha_min = tf.maximum(dynamic_alpha_min, tf.cast(self.alpha_min, tf.float32))

        feasible = dynamic_alpha_min <= alpha_max_safe
        alpha_raw = self._alpha(
            learning_rate=eta,
            curvature_for_control=curvature,
            alpha_would=alpha_max_safe,
            grad_norm=grad_norm,
            update_quantum=update_quantum,
        )
        upper = tf.minimum(tf.cast(self.alpha_max, tf.float32), alpha_max_safe)
        lower = tf.minimum(dynamic_alpha_min, upper)
        alpha = tf.clip_by_value(tf.cast(alpha_raw, tf.float32), lower, upper)
        return ControllerStep(
            alpha=alpha,
            alpha_would=alpha_max_safe,
            curvature_for_control=curvature,
            eta_eff=alpha * eta,
            alpha_min=dynamic_alpha_min,
            alpha_max=upper,
            feasible=feasible,
        )


class Controller:
    """Factory facade for controller construction."""

    @staticmethod
    def from_str(controller: str | BaseController | None, **kwargs: Any) -> BaseController:
        return make_controller(controller, **kwargs)

    @staticmethod
    def from_controller(controller: str | BaseController | None, **kwargs: Any) -> BaseController:
        return make_controller(controller, **kwargs)


CONTROLLER_REGISTRY: dict[str, type[BaseController]] = {
    "none": NoController,
    "ctrl-none": NoController,
    "CTRL-NONE": NoController,
    "gt-order-0": GlobalThrottleOrder0Controller,
    "ctrl-gt-order-0": GlobalThrottleOrder0Controller,
    "CTRL-GT-ORDER-0": GlobalThrottleOrder0Controller,
    "gt-order-1": GlobalThrottleOrder1Controller,
    "ctrl-gt-order-1": GlobalThrottleOrder1Controller,
    "CTRL-GT-ORDER-1": GlobalThrottleOrder1Controller,
    "gt-order-2": GlobalThrottleOrder2Controller,
    "ctrl-gt-order-2": GlobalThrottleOrder2Controller,
    "CTRL-GT-ORDER-2": GlobalThrottleOrder2Controller,
    "gt-order-2-qa": QuantizationAwareOrder2Controller,
    "ctrl-gt-order-2-qa": QuantizationAwareOrder2Controller,
    "CTRL-GT-ORDER-2-QA": QuantizationAwareOrder2Controller,
}


def make_controller(controller: str | BaseController | None, **kwargs: Any) -> BaseController:
    if controller is None:
        return NoController(**kwargs)
    if isinstance(controller, BaseController):
        return controller
    key = controller if controller in CONTROLLER_REGISTRY else controller.lower()
    if key not in CONTROLLER_REGISTRY:
        raise ValueError(
            f"Unknown controller '{controller}'. "
            f"Known controllers: {sorted(set(CONTROLLER_REGISTRY))}"
        )
    return CONTROLLER_REGISTRY[key](**kwargs)
