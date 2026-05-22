"""ENABOL to hls4ml compilation bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf

from .dataset import BaseDataset
from .dtypes import HLSDataType, ap_fixed
from .nn.controller import BaseController, make_controller
from .nn.models import BaseModel
from .precision import PrecisionDict, ensure_precision_dict
from .toolchain import toolchain_environment


TrainableSpec = bool | int | Sequence[str | int]

_STANDARD_LAYER_PRECISION_FIELDS = {
    'weight': 'weight',
    'bias': 'bias',
    'activation': 'result',
    'result': 'result',
    'accumulator': 'accum',
    'accum': 'accum',
}

_TRAINABLE_PRECISION_FIELDS = {
    'loss': 'loss',
    'loss_grad': 'loss_grad',
    'gradient': 'grad_out',
    'grad_in': 'grad_in',
    'grad_out': 'grad_out',
    'weight_grad': 'weight_grad',
    'bias_grad': 'bias_grad',
    'gradient_accum': 'gradient_accum',
    'raw_update': 'raw_update',
    'update': 'update',
    'optimizer_state': 'optimizer_state',
    'controller_metric': 'controller_metric',
    'alpha': 'alpha',
}


def _dtype_to_hls(dtype: HLSDataType | str | None) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, HLSDataType):
        return repr(dtype)
    return str(dtype)


def _normalize_learning_rate(learning_rate: float | None) -> float | None:
    if learning_rate is None:
        return None
    return float(learning_rate)


def _controller_config(controller: str | BaseController | None, *, learning_rate: float | None) -> dict[str, Any]:
    active_controller = make_controller(controller)
    return {
        'Kind': active_controller.controller_id,
        'LearningRate': _normalize_learning_rate(learning_rate),
        'Chi': active_controller.chi,
        'Epsilon': active_controller.eps,
        'CurvatureEmaRho': active_controller.curvature_ema_rho,
        'AlphaMin': active_controller.alpha_min,
        'AlphaMax': active_controller.alpha_max,
        'UseEmaMax': active_controller.use_ema_max,
        'SafetyBudget': {'Enabled': False},
    }


def _optimizer_config(optimizer: str, learning_rate: float | None, learning_rate_input: str | None) -> dict[str, Any]:
    return {
        'Kind': optimizer,
        'LearningRate': _normalize_learning_rate(learning_rate),
        'LearningRateInput': learning_rate_input,
    }


def _resolve_trainable_layers(layer_names: Sequence[str], trainable: TrainableSpec) -> dict[str, bool]:
    if isinstance(trainable, bool):
        return {name: trainable for name in layer_names}

    if isinstance(trainable, int):
        if trainable == 0:
            return {name: False for name in layer_names}
        if abs(trainable) > len(layer_names):
            raise ValueError(f'trainable={trainable} exceeds the number of hls4ml layers ({len(layer_names)}).')
        selected = set(layer_names[-trainable:] if trainable > 0 else layer_names[: abs(trainable)])
        return {name: name in selected for name in layer_names}

    selected_names: set[str] = set()
    selected_indices: set[int] = set()
    for item in trainable:
        if isinstance(item, str):
            selected_names.add(item)
        elif isinstance(item, int):
            selected_indices.add(item)
        else:
            raise TypeError(f'trainable entries must be strings or integers, got {type(item).__name__}.')

    unknown = selected_names - set(layer_names)
    if unknown:
        raise ValueError(f'trainable contains unknown hls4ml layer names: {sorted(unknown)}')

    selected_names.update(layer_names[index] for index in selected_indices)
    return {name: name in selected_names for name in layer_names}


def _set_standard_precision(layer_config: dict[str, Any], field: str, dtype: HLSDataType | None) -> None:
    hls_field = _STANDARD_LAYER_PRECISION_FIELDS.get(field)
    hls_dtype = _dtype_to_hls(dtype)
    if hls_field is None or hls_dtype is None:
        return
    layer_config.setdefault('Precision', {})[hls_field] = hls_dtype


def _set_trainable_precision(training_config: dict[str, Any], field: str, dtype: HLSDataType | None) -> None:
    hls_field = _TRAINABLE_PRECISION_FIELDS.get(field)
    hls_dtype = _dtype_to_hls(dtype)
    if hls_field is None or hls_dtype is None:
        return
    training_config.setdefault('Precision', {})[hls_field] = hls_dtype


def _apply_precision_config(hls_config: dict[str, Any], precision: PrecisionDict | None) -> None:
    if precision is None:
        return

    model_config = hls_config.setdefault('Model', {})
    model_training = model_config.setdefault('Training', {})

    default_fields = precision.get('__default__', {})
    for field, dtype in default_fields.items():
        hls_dtype = _dtype_to_hls(dtype)
        if hls_dtype is None:
            continue
        if field in _STANDARD_LAYER_PRECISION_FIELDS:
            model_config.setdefault('Precision', {})[_STANDARD_LAYER_PRECISION_FIELDS[field]] = hls_dtype
        if field in _TRAINABLE_PRECISION_FIELDS:
            model_training.setdefault('Precision', {})[_TRAINABLE_PRECISION_FIELDS[field]] = hls_dtype

    for layer_name, fields in precision.items():
        if layer_name in {'__default__', 'input'}:
            continue
        if layer_name == 'loss':
            for field, dtype in fields.items():
                _set_trainable_precision(model_training, field, dtype)
            continue

        layer_config = hls_config.setdefault('LayerName', {}).setdefault(layer_name, {})
        layer_training = layer_config.setdefault('Training', {})
        for field, dtype in fields.items():
            _set_standard_precision(layer_config, field, dtype)
            _set_trainable_precision(layer_training, field, dtype)


def build_hls_config(
    model: BaseModel | tf.keras.Model,
    *,
    backend: str = 'Vitis',
    granularity: Literal['model', 'type', 'name'] = 'name',
    default_precision: HLSDataType | str = ap_fixed(16, 6),
    reuse_factor: int = 1,
    strategy: Literal['Latency', 'Resource'] = 'Latency',
    trainable: TrainableSpec = True,
    optimizer: str = 'sgd',
    learning_rate: float | None = 0.01,
    learning_rate_input: str | None = None,
    batch_size: int = 1,
    loss: str = 'half_mse',
    controller: str | BaseController | None = None,
    precision: PrecisionDict | Mapping[str, Mapping[str, Any]] | None = None,
    trace: bool = False,
) -> dict[str, Any]:
    """Build the hls4ml config dictionary with ENABOL training metadata."""

    import hls4ml

    keras_model = model.model if isinstance(model, BaseModel) else model
    if keras_model is None:
        raise ValueError('ENABOL model has not been built yet.')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    if reuse_factor <= 0:
        raise ValueError('reuse_factor must be positive.')

    precision_dict = ensure_precision_dict(precision)
    if precision_dict is not None:
        precision_dict.validate_model(keras_model, allow_missing=True)

    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model,
        granularity=granularity,
        backend=backend,
        default_precision=_dtype_to_hls(default_precision),
        default_reuse_factor=reuse_factor,
    )

    model_config = hls_config.setdefault('Model', {})
    model_config['Strategy'] = strategy
    model_config['ReuseFactor'] = reuse_factor
    model_config['TraceOutput'] = trace
    model_config['Training'] = {
        'Trainable': bool(trainable),
        'BatchSize': int(batch_size),
        'Loss': {
            'Kind': loss,
        },
        'Optimizer': _optimizer_config(optimizer, learning_rate, learning_rate_input),
        'Controller': _controller_config(controller, learning_rate=learning_rate),
        'Precision': {},
    }

    layer_name_config = hls_config.setdefault('LayerName', {})
    trainable_layers = _resolve_trainable_layers(list(layer_name_config.keys()), trainable)
    model_config['Training']['Trainable'] = any(trainable_layers.values())

    for layer_name, is_trainable in trainable_layers.items():
        layer_config = layer_name_config.setdefault(layer_name, {})
        layer_config['Trace'] = trace
        layer_config.setdefault('Training', {})['Trainable'] = is_trainable

    _apply_precision_config(hls_config, precision_dict)
    return hls_config


def _write_dataset_for_testbench(dataset: BaseDataset | None, output_dir: str | os.PathLike[str]) -> tuple[str | None, str | None]:
    if dataset is None:
        return None, None

    tb_dir = Path(output_dir) / 'tb_data'
    dataset.to_dat(prefix=str(tb_dir))
    return str(tb_dir / 'tb_input_features.dat'), str(tb_dir / 'tb_output_predictions.dat')


def compile(
    model: BaseModel | tf.keras.Model,
    dataset: BaseDataset | None = None,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    project_name: str | None = None,
    backend: str = 'Vitis',
    toolchain: str | None = 'auto',
    toolchain_config: str | os.PathLike[str] | None = None,
    part: str = 'xcku035-fbva676-2-e',
    io_type: Literal['io_parallel', 'io_stream'] = 'io_parallel',
    strategy: Literal['Latency', 'Resource'] = 'Latency',
    reuse_factor: int = 1,
    granularity: Literal['model', 'type', 'name'] = 'name',
    trainable: TrainableSpec = True,
    optimizer: str = 'sgd',
    learning_rate: float | None = 0.01,
    learning_rate_input: str | None = None,
    batch_size: int = 1,
    loss: str = 'half_mse',
    controller: str | BaseController | None = None,
    precision: PrecisionDict | Mapping[str, Mapping[str, Any]] | None = None,
    default_precision: HLSDataType | str = ap_fixed(16, 6),
    trace: bool = False,
    write: bool = True,
    compile_cpp: bool = False,
    build: bool = False,
    csim: bool = True,
    synth: bool = False,
    cosim: bool = False,
    validation: bool = False,
    export: bool = False,
    vsynth: bool = False,
    overwrite: bool = False,
    verbose: bool = True,
    **convert_kwargs: Any,
):
    """Convert an ENABOL/Keras model to hls4ml and optionally write/build it.

    ``toolchain`` is applied inside this function. Use ``toolchain="auto"`` on
    Kona to select the configured default profile, and ``toolchain=None`` or
    ``toolchain="none"`` for pure local conversion/write checks.
    """

    import hls4ml

    keras_model = model.model if isinstance(model, BaseModel) else model
    if keras_model is None:
        raise ValueError('ENABOL model has not been built yet.')

    if project_name is None:
        project_name = getattr(model, 'name', None) or getattr(keras_model, 'name', 'enabol_model')
        project_name = str(project_name).replace('-', '_').replace(' ', '_')

    if output_dir is None:
        output_dir = Path.cwd() / f'{project_name}_hls'
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f'Output directory "{output_dir}" already exists. Use overwrite=True to reuse it.')

    input_data_tb, output_data_tb = _write_dataset_for_testbench(dataset, output_dir)
    hls_config = build_hls_config(
        model,
        backend=backend,
        granularity=granularity,
        default_precision=default_precision,
        reuse_factor=reuse_factor,
        strategy=strategy,
        trainable=trainable,
        optimizer=optimizer,
        learning_rate=learning_rate,
        learning_rate_input=learning_rate_input,
        batch_size=batch_size,
        loss=loss,
        controller=controller,
        precision=precision,
        trace=trace,
    )

    if verbose:
        print(f'[INFO] - Converting model to hls4ml backend={backend}, output_dir={output_dir}')

    active_toolchain = toolchain if build else None
    with toolchain_environment(active_toolchain, backend=backend, config_path=toolchain_config, require_commands=build):
        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=hls_config,
            io_type=io_type,
            backend=backend,
            output_dir=str(output_dir),
            project_name=project_name,
            part=part,
            input_data_tb=input_data_tb,
            output_data_tb=output_data_tb,
            **convert_kwargs,
        )

        if write and not compile_cpp and not build:
            if verbose:
                print('[INFO] - Writing hls4ml project.')
            hls_model.write()

        if compile_cpp:
            if verbose:
                print('[INFO] - Compiling generated C++ shared library.')
            hls_model.compile()

        if build:
            if verbose:
                print('[INFO] - Running hls4ml HLS build.')
            hls_model.build(
                csim=csim,
                synth=synth,
                cosim=cosim,
                validation=validation,
                export=export,
                vsynth=vsynth,
            )

    return hls_model, hls_config
