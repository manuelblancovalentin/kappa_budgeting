from enabol.dtypes import ap_fixed
from enabol.precision import PrecisionDict, apply_hls_precision_config, dtype_to_hls


def test_dtype_to_hls_converts_enabol_dtype():
    assert dtype_to_hls(ap_fixed(16, 6)) == "ap_fixed<16,6,AP_TRN,AP_WRAP>"


def test_apply_hls_precision_config_fills_defaults_without_precision_dict():
    hls_config = {"Model": {"Training": {"Precision": {}}}}

    apply_hls_precision_config(hls_config, None)

    precision = hls_config["Model"]["Training"]["Precision"]
    assert precision["loss"] == "ap_fixed<32,16,AP_TRN,AP_WRAP>"
    assert precision["grad_in"] == "ap_fixed<20,8,AP_TRN,AP_WRAP>"
    assert precision["raw_update"] == "ap_fixed<20,6,AP_TRN,AP_WRAP>"
    assert precision["alpha"] == "ap_fixed<16,4,AP_TRN,AP_WRAP>"


def test_apply_hls_precision_config_expands_semantic_aliases():
    hls_config = {
        "Model": {"Training": {"Precision": {}}},
        "LayerName": {"dense": {"Precision": {}, "Training": {"Precision": {}}}},
    }
    precision = PrecisionDict(
        {
            "dense": {
                "weight": "ap_fixed<12,3>",
                "gradient": "ap_fixed<20,8>",
                "update": "ap_fixed<20,10>",
                "accumulator": "ap_fixed<28,14>",
            },
            "loss": {"value": "ap_fixed<24,10>"},
        }
    )

    apply_hls_precision_config(hls_config, precision)

    model_precision = hls_config["Model"]["Training"]["Precision"]
    dense_precision = hls_config["LayerName"]["dense"]["Training"]["Precision"]

    assert model_precision["loss"] == "ap_fixed<24,10,AP_TRN,AP_WRAP>"
    assert hls_config["LayerName"]["dense"]["Precision"]["weight"] == "ap_fixed<12,3,AP_TRN,AP_WRAP>"
    assert dense_precision["grad_in"] == "ap_fixed<20,8,AP_TRN,AP_WRAP>"
    assert dense_precision["grad_out"] == "ap_fixed<20,8,AP_TRN,AP_WRAP>"
    assert dense_precision["weight_grad"] == "ap_fixed<20,8,AP_TRN,AP_WRAP>"
    assert dense_precision["raw_update"] == "ap_fixed<20,10,AP_TRN,AP_WRAP>"
    assert dense_precision["optimizer_state"] == "ap_fixed<20,10,AP_TRN,AP_WRAP>"
    assert dense_precision["gradient_accum"] == "ap_fixed<28,14,AP_TRN,AP_WRAP>"
    assert dense_precision["controller_metric"] == "ap_fixed<28,14,AP_TRN,AP_WRAP>"
