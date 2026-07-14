import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx
from transformers import AutoModelForCausalLM, AutoTokenizer

import skyrl.tx.models.qwen3_5 as qwen3_5_module
from skyrl.tinker.types import LoraConfig
from skyrl.tx.layers.lora import init_lora_adapter
from skyrl.tx.models.configs import Qwen3_5Config
from skyrl.tx.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)
from tests.tx.models.conftest import load_model


def _tiny_qwen3_5_config(*, gradient_checkpointing: bool) -> Qwen3_5Config:
    # One layer of each Qwen3.5 decoder type, with dimensions small enough for
    # CPU-only gradient tests. seq_len > 1 exercises the chunked GDN path.
    base_config = {
        "vocab_size": 32,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rope_parameters": {"partial_rotary_factor": 0.5, "rope_theta": 10_000.0},
        "attention_bias": False,
        "linear_num_value_heads": 2,
        "linear_num_key_heads": 1,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 2,
        "intermediate_size": 12,
        "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        "output_hidden_states": False,
        "tie_word_embeddings": False,
    }
    return Qwen3_5Config(
        base_config,
        max_lora_adapters=1,
        max_lora_rank=2,
        shard_attention_heads=True,
        gradient_checkpointing=gradient_checkpointing,
    )


def _tiny_mesh() -> jax.sharding.Mesh:
    return jax.make_mesh(
        (1, 1),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto),
    )


def _checkpointing_training_result(use_checkpointing: bool) -> dict:
    config = _tiny_qwen3_5_config(gradient_checkpointing=use_checkpointing)
    # 70 positions force two 64-token GDN chunks. The final five positions are
    # right padding, leaving one valid token in the second chunk.
    input_ids = (jnp.arange(70, dtype=jnp.int32) % config.vocab_size)[None, :]
    target_ids = jnp.roll(input_ids, -1, axis=1)
    attention_mask = (jnp.arange(input_ids.shape[1])[None, :] < 65).astype(jnp.int32)
    adapter_indices = jnp.zeros((input_ids.shape[0],), dtype=jnp.int32)

    with jax.set_mesh(_tiny_mesh()):
        model = Qwen3_5ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        init_lora_adapter(model, adapter_index=0, lora_config=LoraConfig(rank=2, alpha=2, seed=1))
        graphdef, lora_params, other_state = nnx.split(model, model.is_lora_param, ...)

        def loss_fn(params):
            local_model = nnx.merge(graphdef, params, other_state)
            outputs = local_model(
                input_ids,
                attention_mask=attention_mask,
                adapter_indices=adapter_indices,
                is_training=True,
            )
            logits = local_model.compute_logits(outputs.last_hidden_state, adapter_indices)
            token_logprobs = jax.nn.log_softmax(logits, axis=-1)
            selected_logprobs = jnp.take_along_axis(token_logprobs, target_ids[..., None], axis=-1)[..., 0]
            loss = -jnp.sum(selected_logprobs * attention_mask) / jnp.sum(attention_mask)
            return loss, outputs.last_hidden_state

        forward_jaxpr = jax.make_jaxpr(lambda params: loss_fn(params)[0])(lora_params)
        remat_count = sum(eqn.primitive.name == "remat2" for eqn in forward_jaxpr.jaxpr.eqns)
        (loss, hidden_states), gradients = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(lora_params)

        gradient_leaves = [
            (jax.tree_util.keystr(path), np.asarray(value)) for path, value in jax.tree.leaves_with_path(gradients)
        ]
        return {
            "loss": np.asarray(loss),
            "hidden_states": np.asarray(hidden_states),
            "gradients": gradient_leaves,
            "remat_count": remat_count,
            "num_hidden_layers": config.num_hidden_layers,
        }


@pytest.mark.parametrize("tp", [1, 2])
def test_qwen3_5(tp: int):
    if tp > 1 and os.getenv("CI"):
        pytest.skip("TP > 1 currently runs out of memory in the CI")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", attn_implementation="eager", use_safetensors=True, torch_dtype=torch.float32
    )

    inputs = ["The capital of France is", "The most popular programming language is"]
    batch = tokenizer(inputs, return_tensors="pt", padding=True)
    with torch.no_grad():
        hf_outputs = hf_model(
            batch.input_ids, attention_mask=batch.attention_mask, output_hidden_states=True, return_dict=True
        )
    del hf_model

    _, model = load_model(
        "Qwen/Qwen3.5-0.8B",
        Qwen3_5Config,
        Qwen3_5ForCausalLM,
        ("fsdp", "tp"),
        mesh_shape=(1, tp),
        max_lora_adapters=32,
        max_lora_rank=32,
    )

    outputs = model(batch.input_ids.numpy(), attention_mask=batch.attention_mask.numpy(), output_hidden_states=True)
    assert outputs.hidden_states is not None
    assert np.allclose(hf_outputs.hidden_states[0].numpy(), outputs.hidden_states[0], rtol=1e-6)
    assert np.allclose(hf_outputs.hidden_states[1].numpy(), outputs.hidden_states[1], rtol=1e-3, atol=1e-3)
    assert np.allclose(hf_outputs.hidden_states[-1].numpy(), outputs.hidden_states[-1], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64, 100])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_chunk_gated_delta_rule_matches_recurrent(batch_size: int, seq_len: int, chunk_size: int, dtype: jnp.dtype):
    """Test that chunk_gated_delta_rule produces the same results as recurrent_gated_delta_rule."""
    num_heads = 4
    k_head_dim = 16
    v_head_dim = 32

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Generate random inputs
    query = jax.random.normal(k1, (batch_size, seq_len, num_heads, k_head_dim), dtype=dtype)
    keys = jax.random.normal(k2, (batch_size, seq_len, num_heads, k_head_dim), dtype=dtype)
    value = jax.random.normal(k3, (batch_size, seq_len, num_heads, v_head_dim), dtype=dtype)
    # g should be negative (decay)
    g = -jax.random.uniform(k4, (batch_size, seq_len, num_heads), dtype=dtype, minval=0.01, maxval=0.5)
    # beta should be in [0, 1]
    beta = jax.random.uniform(k5, (batch_size, seq_len, num_heads), dtype=dtype)

    # Test without initial state
    recurrent_out, recurrent_state = recurrent_gated_delta_rule(query, keys, value, g, beta)
    chunk_out, chunk_state = chunk_gated_delta_rule(query, keys, value, g, beta, chunk_size=chunk_size)

    # Outputs are cast back to BF16 after the FP32 reference computation.
    rtol = 2e-2 if dtype == jnp.bfloat16 else 1e-4
    atol = 2e-2 if dtype == jnp.bfloat16 else 1e-4

    assert (
        recurrent_out.shape == chunk_out.shape
    ), f"Output shapes don't match: {recurrent_out.shape} vs {chunk_out.shape}"
    assert (
        recurrent_state.shape == chunk_state.shape
    ), f"State shapes don't match: {recurrent_state.shape} vs {chunk_state.shape}"

    np.testing.assert_allclose(
        np.array(recurrent_out),
        np.array(chunk_out),
        rtol=rtol,
        atol=atol,
        err_msg="Outputs don't match between recurrent and chunked implementations",
    )
    np.testing.assert_allclose(
        np.array(recurrent_state),
        np.array(chunk_state),
        rtol=rtol,
        atol=atol,
        err_msg="Final states don't match between recurrent and chunked implementations",
    )

    # Test with initial state
    initial_state = jax.random.normal(k6, (batch_size, num_heads, k_head_dim, v_head_dim), dtype=dtype)

    recurrent_out2, recurrent_state2 = recurrent_gated_delta_rule(
        query, keys, value, g, beta, initial_state=initial_state
    )
    chunk_out2, chunk_state2 = chunk_gated_delta_rule(
        query, keys, value, g, beta, chunk_size=chunk_size, initial_state=initial_state
    )

    np.testing.assert_allclose(
        np.array(recurrent_out2),
        np.array(chunk_out2),
        rtol=rtol,
        atol=atol,
        err_msg="Outputs don't match with initial state",
    )
    np.testing.assert_allclose(
        np.array(recurrent_state2),
        np.array(chunk_state2),
        rtol=rtol,
        atol=atol,
        err_msg="Final states don't match with initial state",
    )


@pytest.mark.parametrize("implementation", ["recurrent", "chunk"])
def test_gated_delta_rule_matches_transformers_fp32_reference(implementation: str):
    """BF16 model inputs must use the reference FP32 recurrent-state contract."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )

    rng = np.random.default_rng(7)
    batch_size, num_heads, key_dim, value_dim = 1, 2, 4, 3
    seq_len = 5 if implementation == "recurrent" else 65
    shapes = {
        "query": (batch_size, seq_len, num_heads, key_dim),
        "key": (batch_size, seq_len, num_heads, key_dim),
        "value": (batch_size, seq_len, num_heads, value_dim),
        "gate": (batch_size, seq_len, num_heads),
    }
    values = {name: rng.normal(size=shape).astype(np.float32) for name, shape in shapes.items()}
    values["gate"] = -np.abs(values["gate"]) * 0.03
    beta = rng.uniform(size=shapes["gate"]).astype(np.float32)
    initial_state = rng.normal(size=(batch_size, num_heads, key_dim, value_dim)).astype(np.float32) * 0.02

    jax_inputs = [jnp.asarray(values[name], dtype=jnp.bfloat16) for name in ("query", "key", "value", "gate")]
    jax_beta = jnp.asarray(beta, dtype=jnp.bfloat16)
    jax_initial_state = jnp.asarray(initial_state, dtype=jnp.float32)
    torch_inputs = [torch.tensor(values[name], dtype=torch.bfloat16) for name in ("query", "key", "value", "gate")]
    torch_beta = torch.tensor(beta, dtype=torch.bfloat16)
    torch_initial_state = torch.tensor(initial_state, dtype=torch.float32)

    if implementation == "recurrent":
        actual_output, actual_state = recurrent_gated_delta_rule(
            *jax_inputs,
            jax_beta,
            initial_state=jax_initial_state,
        )
        expected_output, expected_state = torch_recurrent_gated_delta_rule(
            *torch_inputs,
            torch_beta,
            torch_initial_state,
            True,
            True,
        )
    else:
        actual_output, actual_state = chunk_gated_delta_rule(
            *jax_inputs,
            jax_beta,
            chunk_size=64,
            initial_state=jax_initial_state,
        )
        expected_output, expected_state = torch_chunk_gated_delta_rule(
            *torch_inputs,
            torch_beta,
            64,
            torch_initial_state,
            True,
            True,
        )

    actual_output_f32 = np.asarray(actual_output, dtype=np.float32)
    expected_output_f32 = expected_output.float().numpy()
    actual_state_f32 = np.asarray(actual_state, dtype=np.float32)
    expected_state_f32 = expected_state.float().numpy()
    assert actual_output.dtype == jnp.bfloat16
    assert actual_state.dtype == jnp.float32
    assert np.max(np.abs(actual_output_f32 - expected_output_f32)) <= 0.004
    assert np.linalg.norm(actual_output_f32 - expected_output_f32) / np.linalg.norm(expected_output_f32) < 0.006
    assert np.max(np.abs(actual_state_f32 - expected_state_f32)) < 0.006
    assert np.linalg.norm(actual_state_f32 - expected_state_f32) / np.linalg.norm(expected_state_f32) < 0.004


def test_rms_norm_matches_transformers_bf16_reference():
    """Ordinary Qwen3.5 RMSNorm must retain the reference FP32 math contract."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm as HFRMSNorm

    rng = np.random.default_rng(19)
    dim = 256
    eps = 1e-6
    input_values = rng.normal(size=(2, 7, dim)).astype(np.float32)
    weight_values = (rng.normal(size=(dim,)) * 0.2).astype(np.float32)
    x = jnp.asarray(input_values, dtype=jnp.bfloat16)
    weight = jnp.asarray(weight_values, dtype=jnp.bfloat16)

    with jax.set_mesh(_tiny_mesh()):
        actual_norm = qwen3_5_module.Qwen3_5RMSNorm(
            dim,
            eps=eps,
            dtype=jnp.bfloat16,
            rngs=nnx.Rngs(0),
        )
        actual_norm.weight[...] = weight
        actual = actual_norm(x)

    reference_norm = HFRMSNorm(dim, eps=eps)
    with torch.no_grad():
        reference_norm.weight.copy_(torch.from_numpy(np.asarray(weight, dtype=np.float32)))
        expected = reference_norm(torch.from_numpy(np.asarray(x, dtype=np.float32)).to(torch.bfloat16))

    actual_f32 = np.asarray(actual, dtype=np.float32)
    expected_f32 = expected.float().numpy()
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_array_equal(actual_f32, expected_f32)

    # Guard against regressing to the former all-BF16 variance and affine path.
    old_bf16 = x * jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    old_bf16 = old_bf16 * (1.0 + weight)
    assert np.count_nonzero(np.asarray(old_bf16, dtype=np.float32) != expected_f32) > 0


def test_gated_delta_net_preserves_checkpoint_fp32_parameters_in_bf16_model():
    config = _tiny_qwen3_5_config(gradient_checkpointing=False)
    with jax.set_mesh(_tiny_mesh()):
        layer = Qwen3_5GatedDeltaNet(config, layer_idx=0, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))

    assert layer.A_log[...].dtype == jnp.float32
    assert layer.norm.weight[...].dtype == jnp.float32
    assert layer.dt_bias[...].dtype == jnp.bfloat16


def test_qwen35_mlp_rejects_unqualified_up_down_fusion_composition():
    config = _tiny_qwen3_5_config(gradient_checkpointing=True)
    config.qwen35_bf16_down_lora_residual = True
    config.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous = True

    with jax.set_mesh(_tiny_mesh()), pytest.raises(ValueError, match="cannot be enabled together"):
        Qwen3_5MLP(config, dtype=jnp.bfloat16, rngs=nnx.Rngs(0))


def test_qwen35_bf16_down_fusion_selects_only_exact_t64_single_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeParam:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.value = jax.ShapeDtypeStruct(shape, dtype)

        def __getitem__(self, key):
            assert key is Ellipsis
            return self.value

    class FakeGateUp:
        def __call__(self, x, adapter_indices=None):
            del adapter_indices
            shape = (*x.shape[:-1], 9216)
            values = jnp.ones(shape, dtype=jnp.bfloat16)
            return values, values

    class FakeDown:
        def __init__(self):
            self.kernel = FakeParam((9216, 2560), jnp.bfloat16)
            self.max_lora_rank = 8
            self.lora_A = FakeParam((2, 9216, 8), jnp.bfloat16)
            self.lora_B = FakeParam((2, 8, 2560), jnp.bfloat16)
            self.lora_scaling = FakeParam((2,), jnp.bfloat16)
            self.fallback_calls = 0

        def _select_single_adapter_lora(self, adapter_index):
            assert adapter_index.shape == ()
            return (
                jax.ShapeDtypeStruct((9216, 8), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, 2560), jnp.bfloat16),
                jnp.asarray(2.0, dtype=jnp.bfloat16),
            )

        def __call__(self, product, adapter_indices=None):
            del adapter_indices
            self.fallback_calls += 1
            return jnp.zeros((*product.shape[:-1], 2560), dtype=product.dtype)

    fused_calls = []

    def fused_spy(product, weight, lora_a, lora_b, scaling, residual, **kwargs):
        fused_calls.append((product.shape, weight.shape, lora_a.shape, lora_b.shape, scaling, kwargs))
        return residual + jnp.ones_like(residual)

    monkeypatch.setattr(qwen3_5_module, "bf16_lora_residual", fused_spy)
    config = _tiny_qwen3_5_config(gradient_checkpointing=False)
    with jax.set_mesh(_tiny_mesh()):
        mlp = Qwen3_5MLP(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        mlp.qwen35_bf16_down_lora_residual = True
        mlp.gate_up_proj = FakeGateUp()
        fake_down = FakeDown()
        mlp.down_proj = fake_down

        x = jnp.zeros((1, 64, 2560), dtype=jnp.bfloat16)
        residual = jnp.zeros((1, 64, 2560), dtype=jnp.bfloat16)
        adapter_indices = jnp.asarray([0], dtype=jnp.int32)
        actual = mlp(x, adapter_indices=adapter_indices, residual=residual)

        np.testing.assert_array_equal(actual, jnp.ones_like(residual))
        assert fake_down.fallback_calls == 0
        assert len(fused_calls) == 1
        assert fused_calls[0][:4] == (
            (1, 64, 9216),
            (9216, 2560),
            (9216, 8),
            (8, 2560),
        )
        assert fused_calls[0][-1] == {
            "enabled": True,
            "block_m": 16,
            "block_n": 64,
            "block_k": 64,
            "row_superblock": 256,
        }

        short_x = x[:, :-1]
        short_residual = residual[:, :-1]
        fallback = mlp(
            short_x,
            adapter_indices=adapter_indices,
            residual=short_residual,
        )
        np.testing.assert_array_equal(fallback, short_residual)
        assert fake_down.fallback_calls == 1
        assert len(fused_calls) == 1

        fake_down.kernel.dtype = jnp.float32
        mixed_parameter_fallback = mlp(
            x,
            adapter_indices=adapter_indices,
            residual=residual,
        )
        np.testing.assert_array_equal(mixed_parameter_fallback, residual)
        assert fake_down.fallback_calls == 2
        assert len(fused_calls) == 1


def test_qwen35_bf16_contiguous_mlp_up_selects_only_exact_t64_single_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeParam:
        def __init__(self, shape, dtype, *, value=None):
            self.shape = shape
            self.dtype = dtype
            self.value = jax.ShapeDtypeStruct(shape, dtype) if value is None else value

        def __getitem__(self, key):
            assert key is Ellipsis
            return self.value

    class FakeGateUp:
        components = ("gate_proj", "up_proj")
        group_sizes = (1, 1)
        bias = None
        max_lora_adapters = 2
        max_lora_rank = 8

        def __init__(self):
            self.kernel = FakeParam((2560, 18432), jnp.bfloat16)
            self.lora_A = FakeParam((2, 2560, 8), jnp.bfloat16)
            self.lora_B = FakeParam((2, 8, 18432), jnp.bfloat16)
            self.lora_scaling = FakeParam((2,), jnp.bfloat16)
            self.fallback_calls = 0
            self.selected_adapter_indices = []

        def _select_single_adapter_lora(self, adapter_index):
            assert adapter_index.shape == ()
            self.selected_adapter_indices.append(np.asarray(adapter_index).item())
            return (
                jax.ShapeDtypeStruct((2560, 8), jnp.bfloat16),
                jax.ShapeDtypeStruct((8, 18432), jnp.bfloat16),
                jnp.asarray(2.0, dtype=jnp.bfloat16),
            )

        def __call__(self, x, adapter_indices=None):
            del adapter_indices
            self.fallback_calls += 1
            values = jnp.ones((*x.shape[:-1], 9216), dtype=jnp.bfloat16)
            return values, values

    class FakeDown:
        def __init__(self):
            self.calls = 0

        def __call__(self, product, adapter_indices=None):
            del adapter_indices
            self.calls += 1
            return product[..., :2560]

    class FakeRMS:
        eps = 1e-6

        def __init__(self):
            self.weight = FakeParam(
                (2560,),
                jnp.bfloat16,
                value=jnp.zeros((2560,), dtype=jnp.bfloat16),
            )
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            return x

    fused_calls = []

    def fused_spy(x, rms_delta, weight, lora_a, lora_b, scaling, **kwargs):
        fused_calls.append(
            (
                x.shape,
                rms_delta.shape,
                weight.shape,
                lora_a.shape,
                lora_b.shape,
                scaling,
                kwargs,
            )
        )
        return jnp.ones((1, 64, 9216), dtype=jnp.bfloat16)

    monkeypatch.setattr(
        qwen3_5_module,
        "bf16_rms_gate_up_lora_swiglu_contiguous",
        fused_spy,
    )
    config = _tiny_qwen3_5_config(gradient_checkpointing=False)
    with jax.set_mesh(_tiny_mesh()):
        mlp = Qwen3_5MLP(config, dtype=jnp.float32, rngs=nnx.Rngs(0))
        mlp.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous = True
        gate_up = FakeGateUp()
        down = FakeDown()
        norm = FakeRMS()
        mlp.gate_up_proj = gate_up
        mlp.down_proj = down

        x = jnp.zeros((1, 64, 2560), dtype=jnp.bfloat16)
        residual = jnp.zeros_like(x)
        adapter_indices = jnp.asarray([1], dtype=jnp.int32)
        actual = mlp.forward_with_post_attention_rms(
            x,
            norm,
            adapter_indices=adapter_indices,
            residual=residual,
        )

        np.testing.assert_array_equal(actual, jnp.ones_like(residual))
        assert norm.calls == 0
        assert gate_up.fallback_calls == 0
        assert gate_up.selected_adapter_indices == [1]
        assert down.calls == 1
        assert len(fused_calls) == 1
        assert fused_calls[0][:5] == (
            (1, 64, 2560),
            (2560,),
            (2560, 18432),
            (2560, 8),
            (8, 18432),
        )
        assert fused_calls[0][-1] == {
            "enabled": True,
            "eps": 1e-6,
            "block_m": 16,
            "block_physical_n": 64,
            "block_k": 32,
        }

        short = x[:, :-1]
        fallback = mlp.forward_with_post_attention_rms(
            short,
            norm,
            adapter_indices=adapter_indices,
            residual=residual[:, :-1],
        )
        np.testing.assert_array_equal(
            fallback,
            nnx.silu(jnp.ones_like(fallback)),
        )
        assert norm.calls == 1
        assert gate_up.fallback_calls == 1
        assert len(fused_calls) == 1

        mlp.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous = False
        disabled_fallback = mlp.forward_with_post_attention_rms(
            x,
            norm,
            adapter_indices=adapter_indices,
            residual=residual,
        )
        np.testing.assert_array_equal(
            disabled_fallback,
            nnx.silu(jnp.ones_like(disabled_fallback)),
        )
        assert norm.calls == 2
        assert gate_up.fallback_calls == 2
        assert len(fused_calls) == 1
        mlp.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous = True

        norm.eps = 1e-5
        epsilon_fallback = mlp.forward_with_post_attention_rms(
            x,
            norm,
            adapter_indices=adapter_indices,
            residual=residual,
        )
        np.testing.assert_array_equal(
            epsilon_fallback,
            nnx.silu(jnp.ones_like(epsilon_fallback)),
        )
        assert norm.calls == 3
        assert gate_up.fallback_calls == 3
        assert len(fused_calls) == 1
        norm.eps = 1e-6

        def can_use(
            *,
            x_arg=x,
            rms_delta=norm.weight[...],
            eps=norm.eps,
            indices=adapter_indices,
            residual_arg=residual,
        ):
            return mlp._can_use_bf16_rms_gate_up_lora_swiglu_contiguous(
                x_arg,
                rms_delta,
                eps,
                indices,
                residual_arg,
            )

        assert can_use()
        assert not can_use(indices=None)
        assert not can_use(indices=jnp.asarray([1], dtype=jnp.float32))
        assert not can_use(indices=jnp.asarray([0, 1], dtype=jnp.int32))
        assert not can_use(x_arg=jax.ShapeDtypeStruct(x.shape, jnp.float32))
        assert not can_use(rms_delta=jax.ShapeDtypeStruct((2560,), jnp.float32))
        assert not can_use(rms_delta=jax.ShapeDtypeStruct((2048,), jnp.bfloat16))
        assert not can_use(residual_arg=jax.ShapeDtypeStruct(residual.shape, jnp.float32))

        for attribute, invalid in (
            ("components", ("up_proj", "gate_proj")),
            ("group_sizes", (9216, 9216)),
            ("bias", FakeParam((18432,), jnp.bfloat16)),
            ("max_lora_rank", 4),
            ("max_lora_adapters", 0),
        ):
            original = getattr(gate_up, attribute)
            setattr(gate_up, attribute, invalid)
            assert not can_use(), attribute
            setattr(gate_up, attribute, original)

        gate_up.kernel.dtype = jnp.float32
        assert not can_use()
        gate_up.kernel.dtype = jnp.bfloat16
        gate_up.kernel.shape = (2560, 9216)
        assert not can_use()
        gate_up.kernel.shape = (2560, 18432)
        gate_up.lora_A.dtype = jnp.float32
        assert not can_use()
        gate_up.lora_A.dtype = jnp.bfloat16
        gate_up.lora_A.shape = (2, 2560, 4)
        assert not can_use()
        gate_up.lora_A.shape = (2, 2560, 8)
        gate_up.lora_B.shape = (2, 4, 18432)
        assert not can_use()
        gate_up.lora_B.shape = (2, 8, 18432)
        gate_up.lora_B.dtype = jnp.float32
        assert not can_use()
        gate_up.lora_B.dtype = jnp.bfloat16
        gate_up.lora_scaling.dtype = jnp.float32
        assert not can_use()
        gate_up.lora_scaling.dtype = jnp.bfloat16
        gate_up.lora_scaling = FakeParam((), jnp.bfloat16)
        assert not can_use()


def test_qwen35_decoder_allows_contiguous_mlp_up_only_in_training_wrapper():
    class IdentityNorm:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            return x

    class FakeAttention:
        def __call__(
            self,
            hidden_states,
            *,
            attention_mask,
            positions,
            adapter_indices,
            kv_cache,
        ):
            del attention_mask, positions, adapter_indices, kv_cache
            return jnp.ones_like(hidden_states), (hidden_states, hidden_states)

    class FakeMLP:
        def __init__(self):
            self.fused_calls = []
            self.ordinary_calls = []

        def __call__(self, x, adapter_indices=None, *, residual=None):
            self.ordinary_calls.append((x, adapter_indices, residual))
            assert residual is not None
            return residual + jnp.asarray(4.0, dtype=residual.dtype)

        def forward_with_post_attention_rms(
            self,
            x,
            rms_norm,
            adapter_indices=None,
            *,
            residual=None,
        ):
            self.fused_calls.append((x, rms_norm, adapter_indices, residual))
            assert residual is not None
            return residual + jnp.asarray(2.0, dtype=residual.dtype)

    config = _tiny_qwen3_5_config(gradient_checkpointing=False)
    config.qwen35_bf16_rms_gate_up_lora_swiglu_contiguous = True
    with jax.set_mesh(_tiny_mesh()):
        layer = Qwen3_5DecoderLayer(
            config,
            layer_idx=1,
            dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        input_norm = IdentityNorm()
        post_attention_norm = IdentityNorm()
        attention = FakeAttention()
        mlp = FakeMLP()
        layer.input_layernorm = input_norm
        layer.post_attention_layernorm = post_attention_norm
        layer.self_attn = attention
        layer.mlp = mlp

        hidden_states = jnp.zeros((1, 64, 2560), dtype=jnp.bfloat16)
        adapter_indices = jnp.asarray([0], dtype=jnp.int32)
        attention_mask = jnp.ones((1, 64), dtype=jnp.int32)
        positions = jnp.arange(64, dtype=jnp.int32)[None, :]
        inference_actual, updated_kv, _, _ = layer(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
        )
        training_actual = qwen3_5_module._run_training_decoder_layer(
            layer,
            hidden_states,
            attention_mask,
            positions,
            adapter_indices,
        )

    np.testing.assert_array_equal(inference_actual, jnp.full_like(hidden_states, 5.0))
    assert updated_kv is not None
    assert input_norm.calls == 2
    assert post_attention_norm.calls == 1
    assert len(mlp.ordinary_calls) == 1
    ordinary_x, ordinary_indices, ordinary_residual = mlp.ordinary_calls[0]
    np.testing.assert_array_equal(ordinary_x, jnp.ones_like(hidden_states))
    np.testing.assert_array_equal(ordinary_residual, ordinary_x)
    np.testing.assert_array_equal(ordinary_indices, adapter_indices)

    np.testing.assert_array_equal(training_actual, jnp.full_like(hidden_states, 3.0))
    assert len(mlp.fused_calls) == 1
    routed_x, routed_norm, routed_indices, routed_residual = mlp.fused_calls[0]
    np.testing.assert_array_equal(routed_x, jnp.ones_like(hidden_states))
    np.testing.assert_array_equal(routed_residual, routed_x)
    assert routed_norm is post_attention_norm
    np.testing.assert_array_equal(routed_indices, adapter_indices)


def test_qwen3_5_gradient_checkpointing_preserves_training_outputs_and_lora_gradients():
    """Per-layer remat must preserve a compiled hybrid-model training step."""
    without_checkpointing = _checkpointing_training_result(False)
    with_checkpointing = _checkpointing_training_result(True)

    assert without_checkpointing["remat_count"] == 0
    assert with_checkpointing["remat_count"] == with_checkpointing["num_hidden_layers"]
    np.testing.assert_allclose(with_checkpointing["loss"], without_checkpointing["loss"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        with_checkpointing["hidden_states"],
        without_checkpointing["hidden_states"],
        rtol=1e-6,
        atol=1e-6,
    )

    gradients_without = without_checkpointing["gradients"]
    gradients_with = with_checkpointing["gradients"]
    assert [path for path, _ in gradients_with] == [path for path, _ in gradients_without]
    for (path, gradient_with), (_, gradient_without) in zip(gradients_with, gradients_without):
        np.testing.assert_allclose(
            gradient_with,
            gradient_without,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Gradient mismatch at {path}",
        )

    # Both the GDN layer and full-attention layer must contribute trainable LoRA gradients.
    for layer_idx in range(with_checkpointing["num_hidden_layers"]):
        layer_lora_b_gradients = [
            gradient for path, gradient in gradients_with if f"['layers'][{layer_idx}]" in path and "['lora_B']" in path
        ]
        assert layer_lora_b_gradients
        assert any(np.any(gradient != 0) for gradient in layer_lora_b_gradients)


def test_qwen3_5_gradient_checkpointing_is_training_only(monkeypatch: pytest.MonkeyPatch):
    """The checkpoint flag must not rematerialize inference or change its KV cache path."""
    config = _tiny_qwen3_5_config(gradient_checkpointing=True)
    input_ids = jnp.arange(5, dtype=jnp.int32)[None, :]
    attention_mask = jnp.ones_like(input_ids)
    rematerialized_layer_types = []

    def remat_spy(layer, hidden_states, mask, positions, adapter_indices):
        rematerialized_layer_types.append(layer.layer_type)
        return qwen3_5_module._run_training_decoder_layer(layer, hidden_states, mask, positions, adapter_indices)

    monkeypatch.setattr(qwen3_5_module, "_remat_training_decoder_layer", remat_spy)

    with jax.set_mesh(_tiny_mesh()):
        model = Qwen3_5ForCausalLM(config, dtype=jnp.float32, rngs=nnx.Rngs(0))

        inference_outputs = model(input_ids, attention_mask=attention_mask)
        assert rematerialized_layer_types == []
        assert inference_outputs.kv_cache is not None
        assert len(inference_outputs.kv_cache.keys) == config.num_hidden_layers

        training_outputs = model(input_ids, attention_mask=attention_mask, is_training=True)
        assert rematerialized_layer_types == ["linear_attention", "full_attention"]
        assert training_outputs.kv_cache is None
        np.testing.assert_allclose(
            np.asarray(training_outputs.last_hidden_state),
            np.asarray(inference_outputs.last_hidden_state),
            rtol=1e-6,
            atol=1e-6,
        )

        config.gradient_checkpointing = False
        outputs_without_checkpointing = model(input_ids, attention_mask=attention_mask, is_training=True)
        assert rematerialized_layer_types == ["linear_attention", "full_attention"]
        np.testing.assert_allclose(
            np.asarray(outputs_without_checkpointing.last_hidden_state),
            np.asarray(training_outputs.last_hidden_state),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_gated_delta_net_right_padded_prefill_cache_matches_unpadded_decode(dtype: jnp.dtype):
    """Right padding must not change a sequence's GDN states or next-token output."""
    config = _tiny_qwen3_5_config(gradient_checkpointing=False)
    config.linear_conv_kernel_dim = 4
    sequence_lengths = (65, 37)
    padded_length = 70

    prefix_key, decode_key = jax.random.split(jax.random.key(42))
    padded_hidden_states = jax.random.normal(
        prefix_key,
        (len(sequence_lengths), padded_length, config.hidden_size),
        dtype=dtype,
    )
    next_hidden_states = jax.random.normal(
        decode_key,
        (len(sequence_lengths), 1, config.hidden_size),
        dtype=dtype,
    )
    attention_mask = jnp.arange(padded_length)[None, :] < jnp.asarray(sequence_lengths)[:, None]

    with jax.set_mesh(_tiny_mesh()):
        gated_delta_net = Qwen3_5GatedDeltaNet(config, layer_idx=0, dtype=dtype, rngs=nnx.Rngs(0))
        padded_output, padded_conv_state, padded_recurrent_state = gated_delta_net(
            padded_hidden_states,
            attention_mask=attention_mask,
        )
        padded_decode_output, padded_decode_conv_state, padded_decode_recurrent_state = gated_delta_net(
            next_hidden_states,
            attention_mask=None,
            conv_state=padded_conv_state,
            recurrent_state=padded_recurrent_state,
        )

        for batch_idx, sequence_length in enumerate(sequence_lengths):
            unpadded_hidden_states = padded_hidden_states[batch_idx : batch_idx + 1, :sequence_length]
            unpadded_output, unpadded_conv_state, unpadded_recurrent_state = gated_delta_net(
                unpadded_hidden_states,
                attention_mask=jnp.ones((1, sequence_length), dtype=jnp.bool_),
            )
            all_valid_output, all_valid_conv_state, all_valid_recurrent_state = gated_delta_net(
                unpadded_hidden_states,
                attention_mask=None,
            )
            unpadded_decode_output, unpadded_decode_conv_state, unpadded_decode_recurrent_state = gated_delta_net(
                next_hidden_states[batch_idx : batch_idx + 1],
                attention_mask=None,
                conv_state=unpadded_conv_state,
                recurrent_state=unpadded_recurrent_state,
            )

            np.testing.assert_allclose(
                np.asarray(padded_output[batch_idx, :sequence_length]),
                np.asarray(unpadded_output[0]),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(all_valid_output),
                np.asarray(unpadded_output),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(all_valid_conv_state),
                np.asarray(unpadded_conv_state),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(all_valid_recurrent_state),
                np.asarray(unpadded_recurrent_state),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(padded_conv_state[batch_idx]),
                np.asarray(unpadded_conv_state[0]),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(padded_recurrent_state[batch_idx]),
                np.asarray(unpadded_recurrent_state[0]),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(padded_decode_output[batch_idx]),
                np.asarray(unpadded_decode_output[0]),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(padded_decode_conv_state[batch_idx]),
                np.asarray(unpadded_decode_conv_state[0]),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(padded_decode_recurrent_state[batch_idx]),
                np.asarray(unpadded_decode_recurrent_state[0]),
                rtol=1e-5,
                atol=1e-5,
            )

        masked_decode_output, masked_decode_conv_state, masked_decode_recurrent_state = gated_delta_net(
            next_hidden_states,
            attention_mask=jnp.zeros((len(sequence_lengths), 1), dtype=jnp.bool_),
            conv_state=padded_conv_state,
            recurrent_state=padded_recurrent_state,
        )
        np.testing.assert_array_equal(np.asarray(masked_decode_output), 0)
        np.testing.assert_array_equal(
            np.asarray(masked_decode_conv_state),
            np.asarray(padded_conv_state),
        )
        np.testing.assert_allclose(
            np.asarray(masked_decode_recurrent_state),
            np.asarray(padded_recurrent_state),
            rtol=1e-6,
            atol=1e-6,
        )
