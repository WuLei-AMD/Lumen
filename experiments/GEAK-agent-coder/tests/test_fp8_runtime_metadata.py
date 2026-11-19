from geak_agent_coder.export.exporter import _hf_quantization_config


def test_hf_fp8_metadata_uses_portable_format_and_preserved_modules():
    config = _hf_quantization_config(
        fp8_format="e4m3fn",
        block_size=128,
        preserved_modules={
            "model.layers.0.mlp.gate",
            "model.layers.0.input_layernorm",
            "lm_head",
        },
    )

    assert config == {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "lm_head",
            "model.layers.0.input_layernorm",
            "model.layers.0.mlp.gate",
        ],
    }
