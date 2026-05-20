from abc import ABC
import tensorflow as tf
from typing import Any, Literal, Mapping, Union, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from ..dataset import BaseDataset
from ..history import FitHistory
from ..precision import PrecisionDict, ensure_precision_dict
from ..quantization import quantize_tensor, rail_stats
from .controller import BaseController, make_controller
from .telemetry import MetricsConfig
from .training import InstrumentationConfig, InstrumentedTrainer

LossBound = float | Literal["auto"] | None

@dataclass(eq=False, repr=False)
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

    def describe(self) -> str:
        lines = [
            f"{self.name}(",
            f"  input_shape: {self.input_shape}",
            f"  output_shape: {self.output_shape}",
            f"  loss: {self.loss}",
            f"  optimizer: {self.optimizer}",
            f"  metrics: {self.metrics}",
            f"  seed: {self.seed}",
        ]
        if self.model is None:
            lines.append("  built: False")
        else:
            trainable_params = int(
                sum(tf.size(var).numpy() for var in self.model.trainable_variables)
            )
            lines.append("  built: True")
            lines.append(f"  trainable_params: {trainable_params}")
            lines.append("  layers:")
            for layer in self.model.layers:
                layer_output = getattr(layer, "output", None)
                output_shape = None if layer_output is None else layer_output.shape
                lines.append(
                    f"    {layer.name}: {layer.__class__.__name__}, "
                    f"output_shape={output_shape}"
                )
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()
    
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
        controller: Optional[str | BaseController] = None,
        metrics: Optional[MetricsConfig] = None,
        precision_dict: Optional[PrecisionDict | Mapping[str, Mapping[str, Any]]] = None,
        verbose: Optional[bool | str] = None,
        progress_interval: Optional[int] = None,
        stop_loss: LossBound = None,
        stop_patience: int = 1,
        stop_min_steps: int = 0,
        diverge_loss: LossBound = None,
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
        controller : BaseController or str, optional
            Controller object or registry ID. ``None`` means no controller.
            Controller-specific settings such as ``chi``, ``eps``, and
            ``curvature_ema_rho`` live on the controller object.
        metrics : MetricsConfig, optional
            Metric profiles and metric-specific inputs. ``None`` preserves the
            legacy full telemetry list.
        precision_dict : PrecisionDict or mapping, optional
            Layer-indexed precision configuration. If None, the loop uses full
            floating-point behavior and remains backward-compatible with the
            original Experiment 000 notebooks.
        verbose : bool or {"progressbar"}, optional
            Per-call verbosity override. ``None`` uses ``self.verbose``.
            ``"progressbar"`` prints one updating progress line.
        progress_interval : int, optional
            Number of logged steps between progressbar redraws. Defaults to
            roughly 5 percent of the total run length.
        stop_loss : float or {"auto"}, optional
            Stop early once the logged loss is at or below this value.
            ``"auto"`` uses the loss precision quantum when available.
        stop_patience : int
            Number of consecutive logged steps that must satisfy
            ``loss <= stop_loss`` before stopping.
        stop_min_steps : int
            Minimum number of logged steps before early stopping is allowed.
        diverge_loss : float or {"auto"}, optional
            Stop and mark the run as diverged once the logged loss is at or
            above this finite blow-up threshold. ``"auto"`` uses the upper
            loss precision rail when available.

        Returns
        -------
        dict[str, np.ndarray]
            History dictionary.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        precision = ensure_precision_dict(precision_dict)
        if precision is not None:
            precision.validate_model(self.model, allow_missing=True)

        active_controller = make_controller(controller)
        verbose_mode = self.verbose if verbose is None else verbose

        trainer = InstrumentedTrainer(
            model_host=self,
            controller=active_controller,
            precision=precision,
            learning_rate=learning_rate,
            config=InstrumentationConfig(
                loss_mode=loss_mode,
                verbose=verbose_mode,
                progress_interval=progress_interval,
                stop_loss=stop_loss,
                stop_patience=stop_patience,
                stop_min_steps=stop_min_steps,
                diverge_loss=diverge_loss,
            ),
            metrics=metrics,
        )

        if verbose_mode is True:
            print("")
            print("============================================================")
            print(f"Training {self.name}")
            print("[Loop]")
            print(f"  epochs={epochs} | batch_size={batch_size} | shuffle={shuffle}")
            print(f"  loss={loss_mode} | η={learning_rate}")
            print("[Controller]")
            for line in active_controller.summary(learning_rate=learning_rate).splitlines():
                print(f"  {line}")
            print("[Telemetry]")
            for line in trainer.metrics.summary().splitlines():
                print(f"  {line}")
            print("[Precision]")
            if precision is None:
                print("  PrecisionDict: disabled")
            else:
                for line in precision.summary().splitlines():
                    print(f"  {line}")
            print("---------------------------------------------------------------")

        history = trainer.fit(
            X,
            Y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if verbose_mode is True:
            print("============================================================")

        return history

    def _forward_with_precision(
        self,
        x: tf.Tensor,
        precision: PrecisionDict,
        *,
        training: bool,
    ) -> tf.Tensor:
        if self.model is None:
            raise ValueError("Model has not been built yet.")

        z = quantize_tensor(x, precision.dtype("input", "value"), ste=True)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            if isinstance(layer, tf.keras.layers.Dense):
                kernel = quantize_tensor(layer.kernel, precision.dtype(layer.name, "weight"), ste=True)
                z = tf.linalg.matmul(z, kernel)
                z = quantize_tensor(z, precision.dtype(layer.name, "accumulator"), ste=True)
                if layer.use_bias and layer.bias is not None:
                    bias = quantize_tensor(layer.bias, precision.dtype(layer.name, "bias"), ste=True)
                    z = z + bias
                z = quantize_tensor(z, precision.dtype(layer.name, "activation"), ste=True)
                continue

            try:
                z = layer(z, training=training)
            except TypeError:
                z = layer(z)
            z = quantize_tensor(z, precision.dtype(layer.name, "activation"), ste=True)

        return z

    def _quantize_gradients(
        self,
        grads: list[tf.Tensor],
        trainable_vars: list[tf.Variable],
        precision: PrecisionDict,
    ) -> list[tf.Tensor]:
        out = []
        for grad, var in zip(grads, trainable_vars):
            layer_name, _ = self._layer_and_field_for_variable(var)
            grad_dtype = precision.dtype(layer_name, "gradient")
            out.append(quantize_tensor(grad, grad_dtype, ste=False))
        return out

    def _quantize_loss(self, loss_value: tf.Tensor, precision: PrecisionDict) -> tf.Tensor:
        return quantize_tensor(
            loss_value,
            precision.dtype("loss", "value"),
            ste=True,
        )

    def _quantize_variable_storage(self, var: tf.Variable, precision: PrecisionDict) -> None:
        layer_name, field_name = self._layer_and_field_for_variable(var)
        dtype = precision.dtype(layer_name, field_name)
        if dtype is not None:
            var.assign(quantize_tensor(var, dtype, ste=False))

    def _layer_and_field_for_variable(self, var: tf.Variable) -> tuple[str, str]:
        if self.model is None:
            raise ValueError("Model has not been built yet.")
        for layer in self.model.layers:
            if hasattr(layer, "kernel") and self._same_variable(var, layer.kernel):
                return layer.name, "weight"
            if hasattr(layer, "bias") and layer.bias is not None and self._same_variable(var, layer.bias):
                return layer.name, "bias"
            for layer_var in layer.trainable_variables:
                if self._same_variable(var, layer_var):
                    return layer.name, "value"
        return "unknown", "value"

    @staticmethod
    def _same_variable(a: tf.Variable, b: tf.Variable) -> bool:
        if a is b:
            return True
        a_path = getattr(a, "path", None)
        b_path = getattr(b, "path", None)
        if a_path is not None and b_path is not None:
            return a_path == b_path
        return getattr(a, "name", None) == getattr(b, "name", None)

    def _rail_max_for_variables(
        self,
        vars_: list[tf.Variable],
        precision: Optional[PrecisionDict],
        *,
        fields: tuple[str, ...],
    ) -> tuple[float, float]:
        if precision is None:
            return 0.0, 0.0
        sat_max = 0.0
        near_max = 0.0
        for var in vars_:
            layer_name, field = self._layer_and_field_for_variable(var)
            if field not in fields:
                continue
            stats = rail_stats(var.numpy(), precision.dtype(layer_name, field))
            sat_max = max(sat_max, stats.saturation_fraction)
            near_max = max(near_max, stats.near_rail_fraction)
        return sat_max, near_max

    def _rail_max_for_tensors(
        self,
        tensors: list[tf.Tensor],
        trainable_vars: list[tf.Variable],
        precision: Optional[PrecisionDict],
        *,
        field: str,
    ) -> tuple[float, float]:
        if precision is None:
            return 0.0, 0.0
        sat_max = 0.0
        near_max = 0.0
        for tensor, var in zip(tensors, trainable_vars):
            layer_name, _ = self._layer_and_field_for_variable(var)
            stats = rail_stats(tensor.numpy(), precision.dtype(layer_name, field))
            sat_max = max(sat_max, stats.saturation_fraction)
            near_max = max(near_max, stats.near_rail_fraction)
        return sat_max, near_max

    def _max_update_quantum(
        self,
        vars_: list[tf.Variable],
        precision: Optional[PrecisionDict],
    ) -> float:
        if precision is None:
            return 0.0
        quantum = 0.0
        for var in vars_:
            layer_name, _ = self._layer_and_field_for_variable(var)
            dtype = precision.dtype(layer_name, "update")
            q = getattr(dtype, "quantum", None)
            if q is not None:
                quantum = max(quantum, float(q))
        return quantum

@dataclass(repr=False)
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
        inputs = tf.keras.Input(shape=input_shape, name="model_input")
        
        if verbose: print(f'[INFO] - Building model with input shape {input_shape} and output shape {output_shape}')
        
        x = inputs
        dense_idx = 0
        activation_idx = 0
        batchnorm_idx = 0
        for units in self.num_hidden:
            layer_name = f"dense{dense_idx}"
            x = tf.keras.layers.Dense(units, use_bias=self.use_bias, name=layer_name)(x)
            if verbose: print(f'[INFO] - Added Dense layer {layer_name} with {units} units')
            dense_idx += 1
            
            if self.activation is not None:
                layer_name = f"activation{activation_idx}"
                x = tf.keras.layers.Activation(self.activation, name=layer_name)(x)
                if verbose: print(f'[INFO] - Added Activation layer {layer_name}')
                activation_idx += 1
            
            if self.use_batchnorm:
                layer_name = f"batchnorm{batchnorm_idx}"
                x = tf.keras.layers.BatchNormalization(name=layer_name)(x)
                if verbose: print(f'[INFO] - Added BatchNormalization layer {layer_name}')
                batchnorm_idx += 1

        # Check if output_shape is compatible with the last hidden layer
        if x.shape[-1] != output_shape[0]:
            # add a final Dense layer to match the output shape
            layer_name = f"dense{dense_idx}"
            x = tf.keras.layers.Dense(output_shape[0], use_bias=self.use_bias, name=layer_name)(x)
            if verbose: print(f'[INFO] - Added final Dense layer {layer_name} with {output_shape[0]} units for output')
        
        model = tf.keras.Model(inputs=inputs, outputs=x, name=self.name)
        return model
