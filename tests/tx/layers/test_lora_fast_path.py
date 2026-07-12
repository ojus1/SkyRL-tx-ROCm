from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import skyrl.tx.layers.lora as lora_module
from skyrl.tx.layers.lora import LoRAEmbed, LoRALinear


def _mesh() -> jax.sharding.Mesh:
    return jax.make_mesh((1,), ("fsdp",), axis_types=(jax.sharding.AxisType.Auto,))


def _set_lora_state(module: LoRALinear | LoRAEmbed, dtype: jnp.dtype = jnp.float32) -> None:
    assert (
        module.lora_A is not None
        and module.lora_B is not None
        and module.lora_scaling is not None
        and module.lora_ranks is not None
    )
    lora_A = jnp.arange(module.lora_A[...].size, dtype=dtype).reshape(module.lora_A.shape) / 97.0
    lora_B = jnp.arange(module.lora_B[...].size, dtype=dtype).reshape(module.lora_B.shape) / 89.0
    module.lora_A[...] = lora_A
    module.lora_B[...] = lora_B
    module.lora_scaling[...] = jnp.asarray([0.25, 0.75, 1.25], dtype=dtype)
    module.lora_ranks[...] = jnp.full(module.lora_ranks.shape, module.max_lora_rank, dtype=jnp.int32)


def _linear(dtype: jnp.dtype = jnp.float32) -> LoRALinear:
    with jax.set_mesh(_mesh()):
        layer = LoRALinear(
            5,
            7,
            sharding=(None, None),
            max_lora_adapters=3,
            max_lora_rank=3,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.zeros_init(),
            rngs=nnx.Rngs(0),
        )
    layer.kernel[...] = jnp.arange(35, dtype=dtype).reshape(5, 7) / 53.0
    _set_lora_state(layer, dtype)
    return layer


def _embedding() -> LoRAEmbed:
    with jax.set_mesh(_mesh()):
        embedding = LoRAEmbed(
            num_embeddings=11,
            features=5,
            sharding=(None, None),
            max_lora_adapters=3,
            max_lora_rank=3,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            embedding_init=nnx.initializers.zeros_init(),
            rngs=nnx.Rngs(0),
        )
    embedding.embedding[...] = jnp.arange(55, dtype=jnp.float32).reshape(11, 5) / 47.0
    _set_lora_state(embedding)
    return embedding


def _assert_tree_allclose(actual, expected, *, rtol: float = 1e-5, atol: float = 1e-6) -> None:
    actual_paths = [jax.tree_util.keystr(path) for path, _ in jax.tree.leaves_with_path(actual)]
    expected_paths = [jax.tree_util.keystr(path) for path, _ in jax.tree.leaves_with_path(expected)]
    assert actual_paths == expected_paths
    for (path, actual_leaf), (_, expected_leaf) in zip(
        jax.tree.leaves_with_path(actual), jax.tree.leaves_with_path(expected), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
            err_msg=jax.tree_util.keystr(path),
        )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_single_adapter_linear_matches_direct_formula_for_dynamic_adapter_index(dtype: jnp.dtype):
    layer = _linear(dtype)
    x = jnp.arange(20, dtype=dtype).reshape(1, 4, 5) / 19.0

    @jax.jit
    def run(adapter_indices):
        return layer(x, adapter_indices=adapter_indices)

    for adapter_index in (0, 2):
        actual = run(jnp.asarray([adapter_index], dtype=jnp.int32))
        lora_A = layer.lora_A[...][adapter_index]
        lora_B = layer.lora_B[...][adapter_index]
        expected = x @ layer.kernel[...] + (x @ lora_A) @ lora_B * layer.lora_scaling[...][adapter_index]
        tolerance = 2e-2 if dtype == jnp.bfloat16 else 1e-6
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_single_adapter_linear_forward_and_gradients_match_generic_routing(dtype: jnp.dtype):
    layer = _linear(dtype)
    graphdef, params, other_state = nnx.split(layer, nnx.Param, ...)
    x = jnp.arange(20, dtype=dtype).reshape(1, 4, 5) / 17.0
    cotangent = jnp.arange(28, dtype=dtype).reshape(4, 7) / 31.0

    def loss_fn(params, x, *, duplicate_batch: bool):
        local_layer = nnx.merge(graphdef, params, other_state)
        if duplicate_batch:
            model_input = jnp.concatenate([x, jax.lax.stop_gradient(x * 0.5 + 0.25)], axis=0)
            adapter_indices = jnp.asarray([1, 1], dtype=jnp.int32)
        else:
            model_input = x
            adapter_indices = jnp.asarray([1], dtype=jnp.int32)
        selected_output = local_layer(model_input, adapter_indices=adapter_indices)[0]
        return jnp.sum(selected_output * cotangent), selected_output

    fast = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True), static_argnames="duplicate_batch")
    (fast_loss, fast_output), (fast_param_grads, fast_x_grad) = fast(
        params, x, duplicate_batch=False
    )
    (generic_loss, generic_output), (generic_param_grads, generic_x_grad) = fast(
        params, x, duplicate_batch=True
    )

    tolerance = 2e-2 if dtype == jnp.bfloat16 else 1e-6
    np.testing.assert_allclose(np.asarray(fast_loss), np.asarray(generic_loss), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(np.asarray(fast_output), np.asarray(generic_output), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(np.asarray(fast_x_grad), np.asarray(generic_x_grad), rtol=tolerance, atol=tolerance)
    _assert_tree_allclose(fast_param_grads, generic_param_grads, rtol=tolerance, atol=tolerance)


def test_single_adapter_embedding_forward_and_gradients_match_generic_routing():
    embedding = _embedding()
    graphdef, params, other_state = nnx.split(embedding, nnx.Param, ...)
    token_ids = jnp.asarray([[1, 7, 3, 9]], dtype=jnp.int32)
    cotangent = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 23.0

    def loss_fn(params, *, duplicate_batch: bool):
        local_embedding = nnx.merge(graphdef, params, other_state)
        if duplicate_batch:
            model_input = jnp.concatenate([token_ids, jnp.asarray([[2, 4, 6, 8]], dtype=jnp.int32)], axis=0)
            adapter_indices = jnp.asarray([2, 2], dtype=jnp.int32)
        else:
            model_input = token_ids
            adapter_indices = jnp.asarray([2], dtype=jnp.int32)
        selected_output = local_embedding(model_input, adapter_indices=adapter_indices)[0]
        return jnp.sum(selected_output * cotangent), selected_output

    evaluate = jax.jit(jax.value_and_grad(loss_fn, has_aux=True), static_argnames="duplicate_batch")
    (fast_loss, fast_output), fast_grads = evaluate(params, duplicate_batch=False)
    (generic_loss, generic_output), generic_grads = evaluate(params, duplicate_batch=True)

    adapter_index = 2
    direct_lora = (
        embedding.lora_A[...][adapter_index, token_ids[0]]
        @ embedding.lora_B[...][adapter_index]
        * embedding.lora_scaling[...][adapter_index]
    )
    direct = embedding.embedding[...][token_ids[0]] + direct_lora

    np.testing.assert_allclose(np.asarray(fast_output), np.asarray(direct), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fast_loss), np.asarray(generic_loss), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(fast_output), np.asarray(generic_output), rtol=1e-6, atol=1e-6)
    _assert_tree_allclose(fast_grads, generic_grads)


def test_single_adapter_rank_zero_embedding_is_base_only_with_zero_lora_gradients():
    embedding = _embedding()
    assert embedding.lora_A is not None
    assert embedding.lora_B is not None
    assert embedding.lora_scaling is not None
    assert embedding.lora_ranks is not None
    adapter_index = 1
    embedding.lora_A[...] = embedding.lora_A[...].at[adapter_index].set(0.0)
    embedding.lora_B[...] = embedding.lora_B[...].at[adapter_index].set(0.0)
    embedding.lora_scaling[...] = embedding.lora_scaling[...].at[adapter_index].set(0.0)
    embedding.lora_ranks[...] = embedding.lora_ranks[...].at[adapter_index].set(0)

    graphdef, params, other_state = nnx.split(embedding, nnx.Param, ...)
    token_ids = jnp.asarray([[0, 5, 10]], dtype=jnp.int32)

    def loss_fn(params):
        local_embedding = nnx.merge(graphdef, params, other_state)
        output = local_embedding(token_ids, adapter_indices=jnp.asarray([adapter_index], dtype=jnp.int32))
        return output.sum(), output

    (_, output), gradients = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(params)
    np.testing.assert_allclose(
        np.asarray(output),
        np.asarray(embedding.embedding[...][token_ids]),
        rtol=0.0,
        atol=0.0,
    )
    lora_gradient_leaves = [
        np.asarray(value)
        for path, value in jax.tree.leaves_with_path(gradients)
        if "lora_A" in jax.tree_util.keystr(path) or "lora_B" in jax.tree_util.keystr(path)
    ]
    assert lora_gradient_leaves
    assert all(np.count_nonzero(gradient) == 0 for gradient in lora_gradient_leaves)


def test_single_adapter_bypasses_prepare_routing_but_multi_adapter_keeps_it(monkeypatch: pytest.MonkeyPatch):
    layer = _linear()
    calls = []
    original_prepare_routing = lora_module.prepare_routing

    def prepare_routing_spy(*args, **kwargs):
        calls.append(args[1])
        return original_prepare_routing(*args, **kwargs)

    monkeypatch.setattr(lora_module, "prepare_routing", prepare_routing_spy)

    single_input = jnp.ones((1, 4, 5), dtype=jnp.float32)
    run = jax.jit(lambda values, indices: layer(values, adapter_indices=indices))
    run(single_input, jnp.asarray([1], dtype=jnp.int32))
    assert calls == []

    multi_input = jnp.arange(40, dtype=jnp.float32).reshape(2, 4, 5) / 37.0
    adapter_indices = jnp.asarray([0, 2], dtype=jnp.int32)
    multi_output = run(multi_input, adapter_indices)
    assert len(calls) == 1

    expected = []
    for batch_index, adapter_index in enumerate((0, 2)):
        sample = multi_input[batch_index]
        sample_output = sample @ layer.kernel[...]
        sample_output += (
            (sample @ layer.lora_A[...][adapter_index])
            @ layer.lora_B[...][adapter_index]
            * layer.lora_scaling[...][adapter_index]
        )
        expected.append(sample_output)
    np.testing.assert_allclose(np.asarray(multi_output), np.asarray(jnp.stack(expected)), rtol=1e-6, atol=1e-6)
