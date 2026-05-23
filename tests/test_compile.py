import tensorflow as tf

from enabol.compile import (
    _power_of_two_batch_size,
    _resolve_trainable_layer_configs,
    _resolve_trainable_layers,
    _write_dataset_for_testbench,
    build_hls_config,
)
from enabol.dataset import AffineDataset
from enabol.dtypes import ap_fixed


def make_dense_model():
    inputs = tf.keras.layers.Input(shape=(1,), name='features')
    outputs = tf.keras.layers.Dense(1, name='dense')(inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='tiny_dense')


def test_resolve_trainable_layers_from_bool():
    assert _resolve_trainable_layers(['a', 'b'], True) == {'a': True, 'b': True}
    assert _resolve_trainable_layers(['a', 'b'], False) == {'a': False, 'b': False}


def test_resolve_trainable_layers_from_last_n():
    assert _resolve_trainable_layers(['a', 'b', 'c'], 2) == {'a': False, 'b': True, 'c': True}


def test_resolve_trainable_layer_configs_skips_input_layers_by_default():
    layer_configs = {
        'model_input': {'Precision': {'result': 'auto'}},
        'dense': {'Precision': {'weight': 'auto', 'bias': 'auto', 'result': 'auto'}},
    }

    assert _resolve_trainable_layer_configs(layer_configs, True) == {'model_input': False, 'dense': True}


def test_power_of_two_batch_size_rounds_up():
    assert _power_of_two_batch_size(1) == (1, 0)
    assert _power_of_two_batch_size(4) == (4, 2)
    assert _power_of_two_batch_size(5) == (8, 3)


def test_build_hls_config_emits_training_schema():
    config = build_hls_config(
        make_dense_model(),
        backend='Vitis',
        trainable=True,
        learning_rate=0.025,
        batch_size=4,
        epochs=3,
        shuffle=False,
        shuffle_seed=21,
        log_every=2,
        controller='ctrl-gt-order-0',
        precision={
            '__default__': {'gradient': ap_fixed(18, 8), 'alpha': ap_fixed(16, 4)},
            'dense': {'weight': ap_fixed(12, 3), 'gradient': ap_fixed(20, 8), 'update': ap_fixed(20, 10)},
            'loss': {'value': ap_fixed(24, 10), 'loss_grad': ap_fixed(18, 8)},
        },
    )

    training = config['Model']['Training']

    assert training['Trainable'] is True
    assert training['BatchSize'] == 4
    assert training['BatchSizeRequested'] == 4
    assert training['BatchSizeLog2'] == 2
    assert training['Epochs'] == 3
    assert training['Shuffle'] is False
    assert training['ShuffleSeed'] == 21
    assert training['LogEvery'] == 2
    assert training['Trace'] == {'Loss': True, 'Alpha': True}
    assert training['Metadata']['GeneratedBy'] == 'enabol+hls4ml-trainable'
    assert training['Metadata']['EnabolVersion'] != 'unknown'
    assert training['Metadata']['Hls4mlTrainableVersion'] == '0.0.0a'
    assert training['Loss']['Kind'] == 'half_mse'
    assert training['Optimizer']['Kind'] == 'sgd'
    assert training['Optimizer']['LearningRate'] == 0.025
    assert training['Controller']['Kind'] == 'CTRL-GT-ORDER-0'
    assert training['Precision']['grad_out'] == 'ap_fixed<18,8,AP_TRN,AP_WRAP>'
    assert training['Precision']['alpha'] == 'ap_fixed<16,4,AP_TRN,AP_WRAP>'
    assert training['Precision']['loss'] == 'ap_fixed<24,10,AP_TRN,AP_WRAP>'
    assert training['Precision']['loss_grad'] == 'ap_fixed<18,8,AP_TRN,AP_WRAP>'
    assert training['Precision']['gradient_accum'] == 'ap_fixed<28,14,AP_TRN,AP_WRAP>'

    assert config['LayerName']['dense']['Training']['Trainable'] is True
    assert config['LayerName']['features']['Training']['Trainable'] is False
    assert config['LayerName']['dense']['Precision']['weight'] == 'ap_fixed<12,3,AP_TRN,AP_WRAP>'
    assert config['LayerName']['dense']['Training']['Precision']['grad_out'] == 'ap_fixed<20,8,AP_TRN,AP_WRAP>'
    assert config['LayerName']['dense']['Training']['Precision']['weight_grad'] == 'ap_fixed<20,8,AP_TRN,AP_WRAP>'
    assert config['LayerName']['dense']['Training']['Precision']['raw_update'] == 'ap_fixed<20,10,AP_TRN,AP_WRAP>'
    assert config['LayerName']['dense']['Training']['Precision']['optimizer_state'] == 'ap_fixed<20,10,AP_TRN,AP_WRAP>'


def test_build_hls_config_emits_default_trainable_precision_without_precision_dict():
    config = build_hls_config(make_dense_model(), backend='Vitis', trainable=True)

    precision = config['Model']['Training']['Precision']

    assert precision['loss'] == 'ap_fixed<32,16,AP_TRN,AP_WRAP>'
    assert precision['grad_in'] == 'ap_fixed<20,8,AP_TRN,AP_WRAP>'
    assert precision['raw_update'] == 'ap_fixed<20,6,AP_TRN,AP_WRAP>'
    assert precision['alpha'] == 'ap_fixed<16,4,AP_TRN,AP_WRAP>'


def test_build_hls_config_rounds_batch_size_up_to_power_of_two():
    config = build_hls_config(make_dense_model(), backend='Vitis', trainable=True, batch_size=5)

    training = config['Model']['Training']

    assert training['BatchSize'] == 8
    assert training['BatchSizeRequested'] == 5
    assert training['BatchSizeLog2'] == 3


def test_dataset_testbench_export_uses_staging_directory(tmp_path):
    dataset = AffineDataset(num_samples=2)
    input_path, output_path = _write_dataset_for_testbench(dataset, tmp_path)

    assert input_path == str(tmp_path / 'enabol_tb_data' / 'tb_input_features.dat')
    assert output_path == str(tmp_path / 'enabol_tb_data' / 'tb_output_predictions.dat')
    assert (tmp_path / 'enabol_tb_data' / 'tb_input_features.dat').exists()
    assert not (tmp_path / 'tb_data' / 'tb_input_features.dat').exists()
