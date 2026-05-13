from abc import ABC
import tensorflow as tf
from typing import Dict, Union, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from .dataset import BaseDataset
from .utils import flatten_tensors, half_mse_batch_loss, matrix_norms_np, tensor_l2_norm, safe_cosine, \
                    spectral_norm_np, matrix_norms_np, analytic_single_dense_hessian, \
                        hessian_metrics_np, stability_metrics_from_hessian

from .history import FitHistory

@dataclass(eq=False)
class BaseModel(ABC):
    dataset: BaseDataset
    loss: Union[str, tf.keras.losses.Loss] = "mse"
    optimizer: Union[str, tf.keras.optimizers.Optimizer] = "sgd"
    metrics: list = field(default_factory=lambda: [])
    model: Optional[tf.keras.Model] = field(init=False, default=None)
    input_shape: Optional[Tuple[int, ...]] = field(init=False, default=None)
    output_shape: Optional[Tuple[int, ...]] = field(init=False, default=None)
    name: str = "BaseModel"
    verbose: bool = False
    seed: Optional[int] = None

    def __post_init__(self):
        self.input_shape = self.dataset.input_shape[1:]
        self.output_shape = self.dataset.output_shape[1:]
    
    def _compile(self, optimizer=None, loss=None, metrics=None, **kwargs):
        if optimizer is not None:
            self.optimizer = optimizer
        if loss is not None:
            self.loss = loss
        if metrics is not None:
            self.metrics = metrics
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, **kwargs) # type: ignore

    def summary(self) -> None:
        if self.model is not None:
            self.model.summary()
        else:
            print("Model has not been built yet.")
    
    def reinitialize_weights(self):
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        for layer in self.model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
                print(f'[INFO] - Reinitialized kernel for layer {layer.name} with initializer {layer.kernel_initializer.__class__.__name__}')
            if hasattr(layer, 'bias_initializer') and layer.bias is not None:
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
                print(f'[INFO] - Reinitialized bias for layer {layer.name} with initializer {layer.bias_initializer.__class__.__name__}')

    # Custom training loop
    def train(self, X, Y, epochs=10, batch_size=32) -> dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        
        dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

        # Make sure model has been compiled
        self._compile()  # This will use the default optimizer and loss if not already set

        # Get optimizer and loss from the compiled model
        if self.model.optimizer is None or self.model.compiled_loss is None:
            raise ValueError("Model must be compiled with an optimizer and loss before training.")
        
        optimizer = self.model.optimizer
        loss_fn = self.model.compiled_loss

        # Init history object
        history = []

        for epoch in range(epochs):
            if self.verbose: print(f"Epoch {epoch+1}/{epochs}")
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch, training=True)
                    loss_value = loss_fn(y_batch, predictions)
                grads = tape.gradient(loss_value, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
                if self.verbose and step % 10 == 0:
                    print(f'\rEpoch {epoch+1}/{epochs} ... Loss: {loss_value.numpy():.4f}', end='')
                history.append(loss_value.numpy())
            print()  # New line after each epoch
        
        # Convert to dict and numpy array for easier plotting
        history_dict = {"loss": np.array(history)}
        return history_dict

    def train_instrumented(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.05,
        shuffle: bool = True,
        loss_mode: str = "half_mse",
        curvature_ema_rho: float = 0.05,
        chi: float = 1.5,
        eps: float = 1e-12,
        use_controller: bool = False,
        compute_analytic_hessian: bool = True,
        reference_A: Optional[np.ndarray] = None,
    ) -> FitHistory:
        """Custom instrumented training loop for ENABOL ablations.

        Parameters
        ----------
        X, Y : np.ndarray
            Dataset arrays.
        epochs : int
            Number of epochs.
        batch_size : int
            Mini-batch size.
        learning_rate : float
            Base SGD learning rate.
        shuffle : bool
            Whether to shuffle batches each epoch.
        loss_mode : {"half_mse", "keras_mse"}
            Loss scaling.
        curvature_ema_rho : float
            EMA update rate for curvature proxy.
        chi : float
            Stability margin target. Usually chi < 2.
        eps : float
            Numerical stability constant.
        use_controller : bool
            If True, apply alpha_t to the update.
            If False, only log what alpha_t would have been.
        compute_analytic_hessian : bool
            If True, compute analytic one-layer Hessian for current batch.
        reference_A : np.ndarray, optional
            Teacher matrix for logging ||W - A||_F in the one-layer case.

        Returns
        -------
        dict[str, np.ndarray]
            History dictionary.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        # Use explicit SGD. Do not use model.compile/apply_gradients here.
        eta = float(learning_rate)

        if loss_mode == "half_mse":
            loss_fn = half_mse_batch_loss
            keras_mse_scaling = False
        elif loss_mode == "keras_mse":
            loss_fn = lambda y_true, y_pred: tf.reduce_mean(tf.square(y_pred - y_true))
            keras_mse_scaling = True
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        # Dataset.
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
        ds = ds.batch(batch_size, drop_remainder=False)

        # History.
        history: Dict[str, list] = {
            "loss": [],
            "rmse": [],
            "theta_norm": [],
            "grad_norm": [],
            "raw_update_norm": [],
            "actual_update_norm": [],
            "update_cosine": [],
            "curvature_proxy": [],
            "curvature_ema": [],
            "alpha": [],
            "alpha_would": [],
            'curvature_for_control': [],
            "eta_eff": [],
            "forward_gain_spectral": [],
            "hessian_lambda_max": [],
            "hessian_lambda_min": [],
            "hessian_spectral_norm": [],
            "stability_margin_lambda_raw": [],
            "stability_margin_lambda_ctrl": [],
            "stability_margin_norm_raw": [],
            "stability_margin_norm_ctrl": [],
            "spectral_radius_raw": [],
            "spectral_radius_ctrl": [],
            "weight_error_fro": [],
            'loss_is_finite': [],
            'grad_is_finite': [],
            'theta_is_finite_before': [],
            'theta_is_finite_after': [],
            'diverged': []
        }

        prev_theta: Optional[tf.Tensor] = None
        prev_grad: Optional[tf.Tensor] = None
        curvature_ema = 0.0

        global_step = 0

        # Print main message about training configuration.
        if self.verbose:
            print(f"") # new line
            print(f'============================================================')
            print(f'Training {self.name} for {epochs} epochs with config:')
            print(f'  Learning rate (eta): {eta}')
            print(f'  Loss mode: {loss_mode}')
            print(f'  Curvature EMA rho: {curvature_ema_rho}')
            print(f'  Stability margin target (chi): {chi}')
            print(f'  Use controller: {use_controller}')
            print(f'  Compute analytic Hessian: {compute_analytic_hessian}')
            print(f'---------------------------------------------------------------')

        for epoch in range(epochs):
            for x_batch, y_batch in ds:
                # Snapshot theta before update.
                trainable_vars = self.model.trainable_variables
                theta_before = flatten_tensors(trainable_vars)

                with tf.GradientTape() as tape:
                    y_pred = self.model(x_batch, training=True)
                    loss_value = loss_fn(y_batch, y_pred)

                grads = tape.gradient(loss_value, trainable_vars)

                # Replace None gradients with zeros, if any.
                grads = [
                    tf.zeros_like(v) if g is None else g
                    for g, v in zip(grads, trainable_vars)
                ]

                grad_flat = flatten_tensors(grads)  # type: ignore

                loss_is_finite = bool(np.isfinite(float(loss_value.numpy())))
                grad_is_finite = bool(np.all(np.isfinite(grad_flat.numpy())))
                theta_is_finite_before = bool(np.all(np.isfinite(theta_before.numpy())))

                # Raw update for SGD.
                raw_update_flat = -eta * grad_flat

                # Curvature proxy.
                if prev_theta is None or prev_grad is None:
                    curvature_proxy = 0.0
                else:
                    dG = grad_flat - prev_grad
                    dtheta = theta_before - prev_theta
                    curvature_proxy = float(
                        tensor_l2_norm(dG, eps).numpy()
                        / (tensor_l2_norm(dtheta, eps).numpy() + eps)
                    )

                curvature_ema = (
                    (1.0 - curvature_ema_rho) * curvature_ema
                    + curvature_ema_rho * curvature_proxy
                )

                # Would-be controller.
                curvature_for_control = max(curvature_proxy, curvature_ema)

                if curvature_for_control <= eps:
                    alpha_would = 1.0
                else:
                    alpha_would = min(1.0, chi / (eta * (curvature_for_control + eps)))

                alpha = alpha_would if use_controller else 1.0
                eta_eff = alpha * eta

                # Apply actual update manually.
                for var, grad in zip(trainable_vars, grads):
                    var.assign_sub(eta_eff * grad)  # type: ignore

                theta_after = flatten_tensors(trainable_vars)
                actual_update_flat = theta_after - theta_before
                theta_is_finite_after = bool(np.all(np.isfinite(theta_after.numpy())))

                # Metrics.
                residual = y_pred - y_batch
                rmse = tf.sqrt(tf.reduce_mean(tf.square(residual)))

                theta_norm = tensor_l2_norm(theta_before, eps)
                grad_norm = tensor_l2_norm(grad_flat, eps)
                raw_update_norm = tensor_l2_norm(raw_update_flat, eps)
                actual_update_norm = tensor_l2_norm(actual_update_flat, eps)
                update_cosine = safe_cosine(actual_update_flat, raw_update_flat, eps)

                diverged = not (
                    loss_is_finite
                    and grad_is_finite
                    and theta_is_finite_before
                    and theta_is_finite_after
                )

                # Forward gain for one-layer Dense model.
                # For now: if first trainable var is a Dense kernel with shape (din, dout),
                # Keras stores W as (din, dout), while our math uses W_math as (dout, din).
                kernel_np = None
                forward_gain = np.nan
                weight_error = np.nan

                if len(trainable_vars) >= 1:
                    first = trainable_vars[0].numpy()
                    if first.ndim == 2:
                        # Keras Dense kernel: (din, dout).
                        # Input-output matrix in math convention is W_math = first.T.
                        W_math = first.T
                        kernel_np = W_math
                        forward_gain = spectral_norm_np(W_math)

                        if reference_A is not None and reference_A.shape == W_math.shape:
                            weight_error = float(np.linalg.norm(W_math - reference_A, ord="fro"))

                # Hessian metrics for current batch, one-layer no-bias.
                h_lam_max = np.nan
                h_lam_min = np.nan
                h_norm = np.nan
                margin_raw_lambda = np.nan
                margin_ctrl_lambda = np.nan
                margin_raw_norm = np.nan
                margin_ctrl_norm = np.nan
                rho_raw = np.nan
                rho_ctrl = np.nan

                if compute_analytic_hessian and kernel_np is not None:
                    H = analytic_single_dense_hessian(
                        x_batch.numpy(),
                        d_out=Y.shape[1],
                        keras_mse_scaling=keras_mse_scaling,
                    )

                    hm = hessian_metrics_np(H)
                    h_lam_max = hm["hessian_lambda_max"]
                    h_lam_min = hm["hessian_lambda_min"]
                    h_norm = hm["hessian_spectral_norm"]

                    sm_raw = stability_metrics_from_hessian(H, eta=eta, alpha=1.0)
                    sm_ctrl = stability_metrics_from_hessian(H, eta=eta, alpha=alpha_would)

                    margin_raw_lambda = sm_raw["stability_margin_lambda"]
                    margin_raw_norm = sm_raw["stability_margin_norm"]
                    rho_raw = sm_raw["spectral_radius_update_map"]

                    margin_ctrl_lambda = sm_ctrl["stability_margin_lambda"]
                    margin_ctrl_norm = sm_ctrl["stability_margin_norm"]
                    rho_ctrl = sm_ctrl["spectral_radius_update_map"]

                # Append logs.
                history["loss"].append(float(loss_value.numpy()))
                history["rmse"].append(float(rmse.numpy()))
                history["theta_norm"].append(float(theta_norm.numpy()))
                history["grad_norm"].append(float(grad_norm.numpy()))
                history["raw_update_norm"].append(float(raw_update_norm.numpy()))
                history["actual_update_norm"].append(float(actual_update_norm.numpy()))
                history["update_cosine"].append(float(update_cosine.numpy()))
                history["curvature_proxy"].append(float(curvature_proxy))
                history["curvature_ema"].append(float(curvature_ema))
                history["alpha"].append(float(alpha))
                history["alpha_would"].append(float(alpha_would))
                history["eta_eff"].append(float(eta_eff))
                history["forward_gain_spectral"].append(float(forward_gain))
                history["hessian_lambda_max"].append(float(h_lam_max))
                history["hessian_lambda_min"].append(float(h_lam_min))
                history["hessian_spectral_norm"].append(float(h_norm))
                history["stability_margin_lambda_raw"].append(float(margin_raw_lambda))
                history["stability_margin_lambda_ctrl"].append(float(margin_ctrl_lambda))
                history["stability_margin_norm_raw"].append(float(margin_raw_norm))
                history["stability_margin_norm_ctrl"].append(float(margin_ctrl_norm))
                history["spectral_radius_raw"].append(float(rho_raw))
                history["spectral_radius_ctrl"].append(float(rho_ctrl))
                history["weight_error_fro"].append(float(weight_error))
                history["loss_is_finite"].append(float(loss_is_finite))
                history["grad_is_finite"].append(float(grad_is_finite))
                history["theta_is_finite_before"].append(float(theta_is_finite_before))
                history["theta_is_finite_after"].append(float(theta_is_finite_after))
                history["curvature_for_control"].append(float(curvature_for_control))
                history["diverged"].append(float(diverged))

                if diverged:
                    print(
                        f"[DIVERGED] step={global_step}, "
                        f"loss={float(loss_value.numpy())}, "
                        f"theta_norm={float(theta_norm.numpy())}, "
                        f"grad_norm={float(grad_norm.numpy())}"
                    )
                    if self.verbose:
                        print(f"Aborting training due to divergence.")
                        print(f'============================================================')
                    return FitHistory(**{k: np.asarray(v) for k, v in history.items()})
                    

                # Update previous state for next curvature estimate.
                prev_theta = tf.identity(theta_before)
                prev_grad = tf.identity(grad_flat)

                global_step += 1

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"loss={history['loss'][-1]:.6f}, "
                    f"grad={history['grad_norm'][-1]:.3e}, "
                    f"C={history['curvature_proxy'][-1]:.3e}, "
                    f"alpha_would={history['alpha_would'][-1]:.3f}"
                )

        # Done 
        if self.verbose:
            print(f'============================================================')

        return FitHistory(**{k: np.asarray(v) for k, v in history.items()})

@dataclass
class LinearBlockModel(BaseModel):
    # blocks are defined as: [Dense] -> (Optional) [Activation] -> (Optional) [BatchNorm])
    num_hidden: list = field(default_factory=lambda: [64, 64])
    activation: Optional[Union[str, tf.keras.layers.Activation]] = None
    use_batchnorm: bool = False
    use_bias: bool = True
    name: str = "LinearBlockModel"

    def __post_init__(self):
        super().__post_init__()
        self.model = self._build_model(self.input_shape, self.output_shape, verbose=self.verbose)

    def _build_model(self, input_shape, output_shape, verbose=True) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape)
        
        if verbose: print(f'[INFO] - Building model with input shape {input_shape} and output shape {output_shape}')
        
        x = inputs
        for units in self.num_hidden:
            x = tf.keras.layers.Dense(units, use_bias=self.use_bias)(x)
            if verbose: print(f'[INFO] - Added Dense layer with {units} units')
            
            if self.activation is not None:
                x = tf.keras.layers.Activation(self.activation)(x)
                if verbose: print(f'[INFO] - Added Activation {self.activation if isinstance(self.activation, str) else self.activation.__class__.__name__} layer with activation ')
            
            if self.use_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)
                if verbose: print(f'[INFO] - Added BatchNormalization layer')

        # Check if output_shape is compatible with the last hidden layer
        if x.shape[-1] != output_shape[0]:
            # add a final Dense layer to match the output shape
            x = tf.keras.layers.Dense(output_shape[0])(x)
        
        if verbose: print(f'[INFO] - Added final Dense layer with {output_shape[0]} units for output')
        model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)
        return model
