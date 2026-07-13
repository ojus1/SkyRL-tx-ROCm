import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from skyrl.tinker.types import LoraConfig
from skyrl.tx.layers.lora import init_lora_adapter
from skyrl.tx.models.configs import Qwen3_5Config
from skyrl.tx.models.qwen3_5 import Qwen3_5ForCausalLM


def _tiny_tied_config() -> Qwen3_5Config:
    return Qwen3_5Config(
        {
            "vocab_size": 37,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rope_parameters": {
                "partial_rotary_factor": 0.5,
                "rope_theta": 10_000.0,
            },
            "attention_bias": False,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "layer_types": ["full_attention"],
            "rms_norm_eps": 1e-6,
            "output_hidden_states": False,
            "tie_word_embeddings": True,
        },
        max_lora_adapters=1,
        max_lora_rank=2,
        shard_attention_heads=True,
        loss_chunk_size=64,
        tied_logprob_vocab_superblock_size=13,
    )


def _tiny_mesh() -> jax.sharding.Mesh:
    return jax.make_mesh(
        (1, 1),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )


def test_tiny_tied_qwen_split_head_matches_dense_value_and_lora_gradients() -> None:
    config = _tiny_tied_config()
    input_ids = jnp.asarray([[1, 4, 7, 3, 9]], dtype=jnp.int32)
    target_ids = jnp.asarray([[4, 7, 3, 9, 2]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    adapter_indices = jnp.zeros((1,), dtype=jnp.int32)

    with jax.set_mesh(_tiny_mesh()):
        model = Qwen3_5ForCausalLM(
            config,
            dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        init_lora_adapter(
            model,
            adapter_index=0,
            lora_config=LoraConfig(rank=2, alpha=2, seed=11),
        )
        graphdef, lora_params, other_state = nnx.split(
            model,
            model.is_lora_param,
            ...,
        )

        def hidden_states(local_model):
            return local_model(
                input_ids,
                attention_mask=attention_mask,
                adapter_indices=adapter_indices,
                is_training=True,
            ).last_hidden_state

        def split_loss(params):
            local_model = nnx.merge(graphdef, params, other_state)
            logprobs = local_model.compute_logprobs(
                hidden_states(local_model),
                target_ids,
                adapter_indices,
            )
            return -jnp.mean(logprobs.astype(jnp.float32))

        def dense_loss(params):
            local_model = nnx.merge(graphdef, params, other_state)
            hidden = hidden_states(local_model)
            logits = local_model.compute_logits(hidden, adapter_indices)
            logprobs = local_model.logits_to_logprobs(logits, target_ids)
            return -jnp.mean(logprobs.astype(jnp.float32))

        split_value, split_gradients = jax.value_and_grad(split_loss)(lora_params)
        dense_value, dense_gradients = jax.value_and_grad(dense_loss)(lora_params)

    np.testing.assert_allclose(split_value, dense_value, rtol=2e-6, atol=2e-6)
    split_leaves = jax.tree.leaves_with_path(split_gradients)
    dense_leaves = jax.tree.leaves_with_path(dense_gradients)
    assert [jax.tree_util.keystr(path) for path, _ in split_leaves] == [
        jax.tree_util.keystr(path) for path, _ in dense_leaves
    ]
    for (_, split_gradient), (_, dense_gradient) in zip(
        split_leaves,
        dense_leaves,
        strict=True,
    ):
        np.testing.assert_allclose(
            split_gradient,
            dense_gradient,
            rtol=4e-6,
            atol=4e-6,
        )

    nonzero_lora_b_paths = {
        jax.tree_util.keystr(path)
        for path, gradient in split_leaves
        if "lora_B" in jax.tree_util.keystr(path)
        and bool(jnp.any(gradient != 0))
    }
    for module_name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
        assert any(module_name in path for path in nonzero_lora_b_paths)
