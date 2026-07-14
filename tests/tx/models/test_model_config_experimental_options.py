from transformers import PretrainedConfig

from skyrl.tx.models.configs import Qwen3Config


class _NestedTextConfig(PretrainedConfig):
    model_type = "skyrl-nested-text-test"

    def __init__(self, **kwargs) -> None:
        text_config = kwargs.pop("text_config", None)
        super().__init__(**kwargs)
        self.text_config = text_config or PretrainedConfig(
            vocab_size=41,
            hidden_size=13,
        )


def test_tied_logprob_option_is_default_off() -> None:
    config = Qwen3Config(
        PretrainedConfig(vocab_size=37, hidden_size=11),
        max_lora_adapters=2,
        max_lora_rank=8,
        shard_attention_heads=True,
    )
    assert config.tied_logprob_vocab_superblock_size == 0
    assert config.qwen35_bf16_down_lora_residual is False
    assert config.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous is False


def test_tied_logprob_option_propagates_to_nested_text_config() -> None:
    config = Qwen3Config(
        _NestedTextConfig(tie_word_embeddings=True),
        max_lora_adapters=2,
        max_lora_rank=8,
        shard_attention_heads=True,
        loss_chunk_size=256,
        tied_logprob_vocab_superblock_size=4096,
        qwen35_bf16_down_lora_residual=True,
        qwen35_bf16_rms_gate_up_lora_swiglu_contiguous=True,
    )

    text_config = config.get_text_config()
    assert text_config.loss_chunk_size == 256
    assert text_config.tied_logprob_vocab_superblock_size == 4096
    assert text_config.qwen35_bf16_down_lora_residual is True
    assert text_config.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous is True
    assert text_config.vocab_size == 41
