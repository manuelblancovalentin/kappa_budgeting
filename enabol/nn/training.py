from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import tensorflow as tf

from .controller import BaseController
from .optimizer import SGDUpdateRule
from .telemetry import CurvatureSensor, HistoryRecorder, MetricsConfig, update_geometry
from .applier import UpdateApplier
from ..history import FitHistory
from ..precision import PrecisionDict
from ..utils import (
    analytic_single_dense_hessian,
    flatten_tensors,
    half_mse_batch_loss,
    hessian_metrics_np,
    spectral_norm_np,
    stability_metrics_from_hessian,
)

LossBound = float | Literal["auto"] | None


@dataclass(frozen=True)
class InstrumentationConfig:
    loss_mode: str = "half_mse"
    verbose: bool | str = False
    progress_interval: int | None = None
    stop_loss: LossBound = None
    stop_patience: int = 1
    stop_min_steps: int = 0
    min_loss_change: float = 1e-4
    diverge_loss: LossBound = None

    def describe(self) -> str:
        return "\n".join(
            [
                "InstrumentationConfig(",
                f"  loss_mode: {self.loss_mode}",
                f"  verbose: {self.verbose}",
                f"  progress_interval: {self.progress_interval}",
                f"  stop_loss: {self.stop_loss}",
                f"  stop_patience: {self.stop_patience}",
                f"  stop_min_steps: {self.stop_min_steps}",
                f"  min_loss_change: {self.min_loss_change}",
                f"  diverge_loss: {self.diverge_loss}",
                ")",
            ]
        )

    def __repr__(self) -> str:
        return self.describe()


class InstrumentedTrainer:
    """Owns the instrumented online-learning loop for ablation experiments."""

    def __init__(
        self,
        *,
        model_host: Any,
        controller: BaseController,
        precision: PrecisionDict | None,
        learning_rate: float,
        config: InstrumentationConfig,
        metrics: MetricsConfig | None = None,
    ):
        if model_host.model is None:
            raise ValueError("Model has not been built yet.")

        self.host = model_host
        self.model = model_host.model
        self.controller = controller
        self.precision = precision
        self.optimizer = SGDUpdateRule(learning_rate)
        self.config = self._resolve_config(config, precision)
        self.metrics = metrics or MetricsConfig.all()
        self.sensor = CurvatureSensor(
            ema_rho=controller.curvature_ema_rho,
            eps=controller.eps,
        )
        self.applier = UpdateApplier(
            layer_and_field_for_variable=model_host._layer_and_field_for_variable,
            quantize_variable_storage=model_host._quantize_variable_storage,
        )
        self.recorder = HistoryRecorder(keys=self.metrics.fields)

        if config.loss_mode == "half_mse":
            self.loss_fn = half_mse_batch_loss
            self.keras_mse_scaling = False
        elif config.loss_mode == "keras_mse":
            self.loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred - y_true))
            self.keras_mse_scaling = True
        else:
            raise ValueError(f"Unknown loss_mode: {config.loss_mode}")

    def _resolve_config(
        self,
        config: InstrumentationConfig,
        precision: PrecisionDict | None,
    ) -> InstrumentationConfig:
        stop_loss = self._resolve_stop_loss(config.stop_loss, precision)
        diverge_loss = self._resolve_diverge_loss(config.diverge_loss, precision)
        return InstrumentationConfig(
            loss_mode=config.loss_mode,
            verbose=config.verbose,
            progress_interval=config.progress_interval,
            stop_loss=stop_loss,
            stop_patience=config.stop_patience,
            stop_min_steps=config.stop_min_steps,
            diverge_loss=diverge_loss,
        )

    def _loss_dtype(self, precision: PrecisionDict | None) -> Any:
        if precision is None:
            return None
        return precision.dtype("loss", "value")

    def _resolve_stop_loss(self, stop_loss: LossBound, precision: PrecisionDict | None) -> float | None:
        if stop_loss != "auto":
            return stop_loss
        dtype = self._loss_dtype(precision)
        quantum = getattr(dtype, "quantum", None)
        if quantum is None:
            return None
        return 2.0 * float(quantum)

    def _resolve_diverge_loss(self, diverge_loss: LossBound, precision: PrecisionDict | None) -> float | None:
        if diverge_loss != "auto":
            return diverge_loss
        dtype = self._loss_dtype(precision)
        if dtype is None or not hasattr(dtype, "value_range"):
            return None
        _, hi = dtype.value_range()
        return 0.95 * float(hi)

    def describe(self) -> str:
        return "\n".join(
            [
                "InstrumentedTrainer(",
                f"  model: {self.model.name}",
                f"  optimizer: {self.optimizer.optimizer_id}",
                f"  controller: {self.controller.controller_id}",
                f"  precision: {self.precision is not None}",
                f"  loss_mode: {self.config.loss_mode}",
                f"  telemetry_fields: {len(self.metrics.fields)}",
                ")",
            ]
        )

    def __repr__(self) -> str:
        return self.describe()

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        shuffle: bool,
    ) -> FitHistory:
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False)

        self.controller.reset()
        self.optimizer.reset()
        self.sensor.reset()

        global_step = 0
        total_steps = int(epochs * np.ceil(len(X) / batch_size))
        progressbar = self.config.verbose == "progressbar"
        progress_interval = self._progress_interval(total_steps)
        stop_streak = 0

        for epoch in range(epochs):
            for x_batch, y_batch in ds:
                diverged = self._train_batch(x_batch, y_batch, Y)
                global_step += 1
                if progressbar and (global_step == 1 or global_step == total_steps or global_step % progress_interval == 0):
                    self._print_progress(global_step, total_steps)
                if diverged:
                    if progressbar:
                        print()
                    self._print_diverged(global_step)
                    return FitHistory(**self.recorder.arrays())
                stop_streak = self._updated_stop_streak(stop_streak)
                if self._should_stop(global_step, stop_streak):
                    if progressbar:
                        print()
                    return FitHistory(**self.recorder.arrays())

            if self.config.verbose is True:
                history = self.recorder.history
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"loss={history['loss'][-1]:.6f}, "
                    f"grad={history['grad_norm'][-1]:.3e}, "
                    f"C={history['curvature_proxy'][-1]:.3e}, "
                    f"alpha_would={history['alpha_would'][-1]:.3f}"
                )

        if progressbar:
            print()

        return FitHistory(**self.recorder.arrays())

    def _progress_interval(self, total_steps: int) -> int:
        if self.config.progress_interval is not None:
            return max(1, int(self.config.progress_interval))
        return max(1, int(np.ceil(total_steps * 0.05)))

    def _updated_stop_streak(self, stop_streak: int) -> int:
        if self.config.stop_loss is None:
            return 0
        history = self.recorder.history
        if not history.get("loss"):
            return 0
        loss = history["loss"][-1]
        # Check if change in loss is below threshold and finite.
        if np.isfinite(loss) and loss <= self.config.stop_loss:
            return stop_streak + 1

        prev_loss = history["loss"][-2] if len(history["loss"]) >= 2 else np.inf
        dloss = prev_loss - loss

        # Plateau only means tiny absolute change, not "loss got worse".
        plateau = np.isfinite(dloss) and abs(dloss) <= self.config.min_loss_change

        # Optional: only allow plateau stopping if loss is not huge.
        if plateau:
            return stop_streak + 1
        return 0

    def _should_stop(self, global_step: int, stop_streak: int) -> bool:
        if self.config.stop_loss is None:
            return False
        if global_step < self.config.stop_min_steps:
            return False
        return stop_streak >= max(1, self.config.stop_patience)

    def _print_progress(self, step: int, total_steps: int) -> None:
        width = 28
        fraction = min(1.0, step / max(total_steps, 1))
        filled = int(round(width * fraction))
        bar = "█" * filled + "░" * (width - filled)
        history = self.recorder.history
        loss = history["loss"][-1] if history.get("loss") else np.nan
        alpha = history["alpha"][-1] if history.get("alpha") else np.nan
        print(
            f"\r[{bar}] {step}/{total_steps} "
            f"loss={loss:.4e} α={alpha:.3f}",
            end="",
            flush=True,
        )

    def _train_batch(
        self,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        Y_all: np.ndarray,
    ) -> bool:
        variables = self.model.trainable_variables
        theta_before = flatten_tensors(variables)

        with tf.GradientTape() as tape:
            if self.precision is None:
                y_pred = self.model(x_batch, training=True)
            else:
                y_pred = self.host._forward_with_precision(
                    x_batch,
                    self.precision,
                    training=True,
                )
            loss_value = self.loss_fn(y_batch, y_pred)
            if self.precision is not None:
                loss_value = self.host._quantize_loss(loss_value, self.precision)

        grads = tape.gradient(loss_value, variables)
        grads = [
            tf.zeros_like(var) if grad is None else grad
            for grad, var in zip(grads, variables)
        ]
        if self.precision is not None:
            grads = self.host._quantize_gradients(grads, variables, self.precision)

        grad_flat = flatten_tensors(grads)
        optimizer_step = self.optimizer.propose(grads, variables)
        update_quantum = self.host._max_update_quantum(variables, self.precision)
        sensor = self.sensor.observe(
            theta=theta_before,
            grad=grad_flat,
            raw_update=optimizer_step.update_flat,
            update_quantum=update_quantum,
        )
        controlled = self.controller.control(
            raw_updates=optimizer_step.updates,
            raw_update_flat=optimizer_step.update_flat,
            learning_rate=optimizer_step.learning_rate,
            curvature_proxy=sensor.curvature_proxy,
            curvature_ema=sensor.curvature_ema,
            grad_norm=sensor.grad_norm,
            update_quantum=sensor.update_quantum,
        )
        apply_result = self.applier.apply(
            variables=variables,
            updates=controlled.updates,
            precision=self.precision,
        )

        theta_after = flatten_tensors(variables)
        actual_update_flat = theta_after - theta_before
        geometry = update_geometry(
            raw_update=optimizer_step.update_flat,
            actual_update=actual_update_flat,
            eps=self.controller.eps,
        )

        step_metrics = self._step_metrics(
            x_batch=x_batch,
            y_batch=y_batch,
            y_pred=y_pred,
            Y_all=Y_all,
            variables=variables,
            grads=grads,
            theta_before=theta_before,
            theta_after=theta_after,
            grad_flat=grad_flat,
            loss_value=loss_value,
            alpha=float(controlled.step.alpha.numpy()),
        )

        loss_scalar = float(loss_value.numpy())
        loss_exceeded = (
            self.config.diverge_loss is not None
            and np.isfinite(loss_scalar)
            and loss_scalar >= self.config.diverge_loss
        )
        diverged = loss_exceeded or not (
            step_metrics["loss_is_finite"]
            and step_metrics["grad_is_finite"]
            and step_metrics["theta_is_finite_before"]
            and step_metrics["theta_is_finite_after"]
        )

        self.recorder.append(
            loss=loss_value,
            theta_norm=sensor.theta_norm,
            grad_norm=sensor.grad_norm,
            raw_update_norm=sensor.raw_update_norm,
            actual_update_norm=geometry.actual_update_norm,
            update_cosine=geometry.update_cosine,
            update_angle_rad=geometry.update_angle_rad,
            update_radius_ratio=geometry.update_radius_ratio,
            curvature_proxy=sensor.curvature_proxy,
            curvature_ema=sensor.curvature_ema,
            alpha=controlled.step.alpha,
            alpha_would=controlled.step.alpha_would,
            alpha_min_bound=controlled.step.alpha_min,
            alpha_max_bound=controlled.step.alpha_max,
            controller_feasible=controlled.step.feasible,
            curvature_for_control=controlled.step.curvature_for_control,
            eta_eff=controlled.step.eta_eff,
            update_saturation_fraction_max=apply_result.update_saturation_fraction_max,
            update_underflow_fraction_max=apply_result.update_underflow_fraction_max,
            diverged=diverged,
            **step_metrics,
        )
        return bool(diverged)

    def _step_metrics(
        self,
        *,
        x_batch: tf.Tensor,
        y_batch: tf.Tensor,
        y_pred: tf.Tensor,
        Y_all: np.ndarray,
        variables: list[tf.Variable],
        grads: list[tf.Tensor],
        theta_before: tf.Tensor,
        theta_after: tf.Tensor,
        grad_flat: tf.Tensor,
        loss_value: tf.Tensor,
        alpha: float,
    ) -> dict[str, float | bool]:
        residual = y_pred - y_batch
        rmse = tf.sqrt(tf.reduce_mean(tf.square(residual)))

        loss_is_finite = bool(np.isfinite(float(loss_value.numpy())))
        grad_is_finite = bool(np.all(np.isfinite(grad_flat.numpy())))
        theta_is_finite_before = bool(np.all(np.isfinite(theta_before.numpy())))
        theta_is_finite_after = bool(np.all(np.isfinite(theta_after.numpy())))

        if self.metrics.requires_quantization_metrics:
            weight_saturation_max, weight_near_rail_max = self.host._rail_max_for_variables(
                variables,
                self.precision,
                fields=("weight", "bias"),
            )
            gradient_saturation_max, gradient_near_rail_max = self.host._rail_max_for_tensors(
                grads,
                variables,
                self.precision,
                field="gradient",
            )
        else:
            weight_saturation_max = np.nan
            weight_near_rail_max = np.nan
            gradient_saturation_max = np.nan
            gradient_near_rail_max = np.nan

        needs_kernel = self.metrics.requires_teacher or self.metrics.requires_hessian
        forward_gain, weight_error, kernel_np = self._one_layer_kernel_metrics(
            variables,
            enabled=needs_kernel,
        )
        hessian_metrics = self._hessian_metrics(
            x_batch=x_batch,
            Y_all=Y_all,
            kernel_np=kernel_np,
            alpha=alpha,
        )

        return {
            "rmse": float(rmse.numpy()),
            "forward_gain_spectral": float(forward_gain),
            "weight_error_fro": float(weight_error),
            "loss_is_finite": loss_is_finite,
            "grad_is_finite": grad_is_finite,
            "theta_is_finite_before": theta_is_finite_before,
            "theta_is_finite_after": theta_is_finite_after,
            "weight_saturation_fraction_max": float(weight_saturation_max),
            "weight_near_rail_fraction_max": float(weight_near_rail_max),
            "gradient_saturation_fraction_max": float(gradient_saturation_max),
            "gradient_near_rail_fraction_max": float(gradient_near_rail_max),
            **hessian_metrics,
        }

    def _one_layer_kernel_metrics(
        self,
        variables: list[tf.Variable],
        *,
        enabled: bool,
    ) -> tuple[float, float, np.ndarray | None]:
        forward_gain = np.nan
        weight_error = np.nan
        kernel_np = None

        if not enabled:
            return forward_gain, weight_error, kernel_np

        if variables:
            first = variables[0].numpy()
            if first.ndim == 2:
                kernel_np = first.T
                forward_gain = spectral_norm_np(kernel_np)
                reference_A = self.metrics.reference_A
                if reference_A is not None and reference_A.shape == kernel_np.shape:
                    weight_error = float(np.linalg.norm(kernel_np - reference_A, ord="fro"))

        return forward_gain, weight_error, kernel_np

    def _hessian_metrics(
        self,
        *,
        x_batch: tf.Tensor,
        Y_all: np.ndarray,
        kernel_np: np.ndarray | None,
        alpha: float,
    ) -> dict[str, float]:
        out = {
            "hessian_lambda_max": np.nan,
            "hessian_lambda_min": np.nan,
            "hessian_spectral_norm": np.nan,
            "stability_margin_lambda_raw": np.nan,
            "stability_margin_lambda_ctrl": np.nan,
            "stability_margin_norm_raw": np.nan,
            "stability_margin_norm_ctrl": np.nan,
            "spectral_radius_raw": np.nan,
            "spectral_radius_ctrl": np.nan,
        }

        if not self.metrics.requires_hessian or kernel_np is None:
            return out

        H = analytic_single_dense_hessian(
            x_batch.numpy(),
            d_out=Y_all.shape[1],
            keras_mse_scaling=self.keras_mse_scaling,
        )
        hm = hessian_metrics_np(H)
        sm_raw = stability_metrics_from_hessian(
            H,
            eta=float(self.optimizer.learning_rate.numpy()),
            alpha=1.0,
        )
        sm_ctrl = stability_metrics_from_hessian(
            H,
            eta=float(self.optimizer.learning_rate.numpy()),
            alpha=alpha,
        )

        out.update(
            hessian_lambda_max=hm["hessian_lambda_max"],
            hessian_lambda_min=hm["hessian_lambda_min"],
            hessian_spectral_norm=hm["hessian_spectral_norm"],
            stability_margin_lambda_raw=sm_raw["stability_margin_lambda"],
            stability_margin_lambda_ctrl=sm_ctrl["stability_margin_lambda"],
            stability_margin_norm_raw=sm_raw["stability_margin_norm"],
            stability_margin_norm_ctrl=sm_ctrl["stability_margin_norm"],
            spectral_radius_raw=sm_raw["spectral_radius_update_map"],
            spectral_radius_ctrl=sm_ctrl["spectral_radius_update_map"],
        )
        return out

    def _print_diverged(self, global_step: int) -> None:
        history = self.recorder.history
        loss = history.get("loss", [np.nan])[-1]
        theta_norm = history.get("theta_norm", [np.nan])[-1]
        grad_norm = history.get("grad_norm", [np.nan])[-1]
        print(
            f"[DIVERGED] step={global_step}, "
            f"loss={loss}, "
            f"theta_norm={theta_norm}, "
            f"grad_norm={grad_norm}"
        )
        if self.host.verbose:
            print("Aborting training due to divergence.")
            print("============================================================")
