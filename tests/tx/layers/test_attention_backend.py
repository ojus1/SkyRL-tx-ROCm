from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.layers import attention


def _array(shape, dtype):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype)


def _valid_pallas_inputs():
    q = _array((1, 512, 16, 256), jnp.bfloat16)
    k = _array((1, 512, 4, 256), jnp.bfloat16)
    v = _array((1, 512, 4, 256), jnp.bfloat16)
    mask = _array((1, 512), jnp.int32)
    return q, k, v, mask


def test_has_cuda_backend_for_cuda(monkeypatch):
    monkeypatch.setattr(attention.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        attention.jax_backend, "get_backend", lambda: SimpleNamespace(platform_version="CUDA 12.8")
    )

    assert attention._has_cuda_backend()


def test_has_cuda_backend_rejects_rocm(monkeypatch):
    monkeypatch.setattr(attention.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        attention.jax_backend,
        "get_backend",
        lambda: SimpleNamespace(platform_version="PJRT C API\nrocm 70200"),
    )

    assert not attention._has_cuda_backend()


def test_has_cuda_backend_rejects_non_gpu(monkeypatch):
    monkeypatch.setattr(attention.jax, "default_backend", lambda: "cpu")

    assert not attention._has_cuda_backend()


def test_has_rocm_backend(monkeypatch):
    monkeypatch.setattr(attention.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        attention.jax_backend,
        "get_backend",
        lambda: SimpleNamespace(platform_version="PJRT C API\nROCm 7.2.0"),
    )

    assert attention._has_rocm_backend()


def test_has_rocm_backend_rejects_cuda(monkeypatch):
    monkeypatch.setattr(attention.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        attention.jax_backend, "get_backend", lambda: SimpleNamespace(platform_version="CUDA 12.8")
    )

    assert not attention._has_rocm_backend()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_rocm_pallas_attention_opt_in_values(monkeypatch, value):
    monkeypatch.setenv(attention._ROCM_PALLAS_ATTENTION_ENV, value)

    assert attention._rocm_pallas_attention_enabled()


@pytest.mark.parametrize("value", [None, "", "0", "false", "FALSE", "no", "off", " Off "])
def test_rocm_pallas_attention_is_disabled_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(attention._ROCM_PALLAS_ATTENTION_ENV, raising=False)
    else:
        monkeypatch.setenv(attention._ROCM_PALLAS_ATTENTION_ENV, value)

    assert not attention._rocm_pallas_attention_enabled()


@pytest.mark.parametrize("value", ["anything-else", "truthy", "2"])
def test_rocm_pallas_attention_rejects_invalid_opt_in_values(monkeypatch, value):
    monkeypatch.setenv(attention._ROCM_PALLAS_ATTENTION_ENV, value)

    with pytest.raises(ValueError, match="Invalid SKYRL_ROCM_PALLAS_ATTENTION"):
        attention._rocm_pallas_attention_enabled()


def test_pallas_selector_accepts_causal_rocm_gqa(monkeypatch):
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: True)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    q, k, v, mask = _valid_pallas_inputs()

    assert attention._can_use_rocm_pallas_attention(q, k, v, mask, True, 256)


def test_pallas_selector_requires_opt_in(monkeypatch):
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: False)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    q, k, v, mask = _valid_pallas_inputs()

    assert not attention._can_use_rocm_pallas_attention(q, k, v, mask, True, 256)


def test_pallas_selector_requires_rocm(monkeypatch):
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: True)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: False)
    q, k, v, mask = _valid_pallas_inputs()

    assert not attention._can_use_rocm_pallas_attention(q, k, v, mask, True, 256)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape", "mask_shape", "dtype", "mask_dtype", "causal", "head_dim"),
    [
        ((1, 128, 16, 256), (1, 128, 4, 256), (1, 128, 4, 256), (1, 128), jnp.bfloat16, jnp.int32, False, 256),
        ((1, 96, 16, 256), (1, 96, 4, 256), (1, 96, 4, 256), (1, 96), jnp.bfloat16, jnp.int32, True, 256),
        ((1, 1, 16, 256), (1, 128, 4, 256), (1, 128, 4, 256), (1, 128), jnp.bfloat16, jnp.int32, True, 256),
        ((1, 128, 16, 256), (1, 128, 4, 256), (1, 128, 4, 256), (1, 128), jnp.float32, jnp.int32, True, 256),
        ((1, 32768, 16, 256), (1, 32768, 4, 256), (1, 32768, 4, 256), (1, 32768), jnp.bfloat16, jnp.int32, True, 256),
        ((1, 128, 16, 32), (1, 128, 4, 32), (1, 128, 4, 32), (1, 128), jnp.bfloat16, jnp.int32, True, 32),
        ((1, 128, 10, 256), (1, 128, 4, 256), (1, 128, 4, 256), (1, 128), jnp.bfloat16, jnp.int32, True, 256),
        ((1, 128, 16, 256), (1, 128, 4, 256), (1, 128, 4, 256), (1, 128), jnp.bfloat16, jnp.float32, True, 256),
        ((2, 128, 16, 256), (2, 128, 4, 256), (2, 128, 4, 256), (2, 128), jnp.bfloat16, jnp.int32, True, 256),
        ((1, 128, 8, 256), (1, 128, 2, 256), (1, 128, 2, 256), (1, 128), jnp.bfloat16, jnp.int32, True, 256),
    ],
)
def test_pallas_selector_rejects_incompatible_inputs(
    monkeypatch,
    q_shape,
    k_shape,
    v_shape,
    mask_shape,
    dtype,
    mask_dtype,
    causal,
    head_dim,
):
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: True)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    q = _array(q_shape, dtype)
    k = _array(k_shape, dtype)
    v = _array(v_shape, dtype)
    mask = _array(mask_shape, mask_dtype)

    assert not attention._can_use_rocm_pallas_attention(q, k, v, mask, causal, head_dim)


def test_pallas_selector_rejects_mixed_qkv_dtypes(monkeypatch):
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: True)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    q, k, v, mask = _valid_pallas_inputs()
    k.dtype = jnp.float16

    assert not attention._can_use_rocm_pallas_attention(q, k, v, mask, True, 256)


def test_dot_product_attention_preserves_cuda_cudnn_precedence(monkeypatch):
    q = np.zeros((1, 8, 2, 4), dtype=np.float16)
    k = np.zeros((1, 8, 2, 4), dtype=np.float16)
    v = np.zeros((1, 8, 2, 4), dtype=np.float16)
    mask = np.ones((1, 8), dtype=np.int32)
    expected = object()
    calls = []

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: True)
    monkeypatch.setattr(
        attention,
        "_can_use_rocm_pallas_attention",
        lambda *args, **kwargs: pytest.fail("ROCm selector must not run after a supported CUDA dispatch"),
    )

    def fake_dot_product_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(attention.jax.nn, "dot_product_attention", fake_dot_product_attention)

    result = attention.dot_product_attention(q, k, v, mask, True, 4)

    assert result is expected
    assert len(calls) == 1
    assert calls[0][1]["implementation"] == "cudnn"


def test_dot_product_attention_dispatches_eligible_rocm_input_to_pallas(monkeypatch):
    q, k, v, mask = _valid_pallas_inputs()
    expected = object()
    calls = []

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: False)
    monkeypatch.setattr(attention, "_can_use_rocm_pallas_attention", lambda *args: True)

    def fake_rocm_pallas_attention(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(attention, "_rocm_pallas_attention", fake_rocm_pallas_attention)

    result = attention.dot_product_attention(q, k, v, mask, True, 256)

    assert result is expected
    assert calls == [(q, k, v, mask, 1 / 16)]


def test_dot_product_attention_uses_generic_fallback_for_ineligible_input(monkeypatch):
    q = np.zeros((1, 8, 2, 4), dtype=np.float32)
    k = np.zeros((1, 8, 2, 4), dtype=np.float32)
    v = np.zeros((1, 8, 2, 4), dtype=np.float32)
    mask = np.asarray([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=np.int32)
    expected = object()
    calls = []

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: False)
    monkeypatch.setattr(attention, "_can_use_rocm_pallas_attention", lambda *args: False)

    def fake_dot_product_attention(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(attention.jax.nn, "dot_product_attention", fake_dot_product_attention)

    result = attention.dot_product_attention(q, k, v, mask, False, 4)

    assert result is expected
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][1]["mask"], mask[:, None, None, :].astype(bool))
    assert calls[0][1]["is_causal"] is False
    assert "implementation" not in calls[0][1]


def test_explicit_pallas_opt_in_refuses_long_quadratic_fallback(monkeypatch):
    q = _array((1, 512, 16, 256), jnp.float32)
    k = _array((1, 512, 4, 256), jnp.float32)
    v = _array((1, 512, 4, 256), jnp.float32)
    mask = _array((1, 512), jnp.int32)

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: False)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: True)
    monkeypatch.setattr(attention, "_can_use_rocm_pallas_attention", lambda *args: False)
    monkeypatch.setattr(
        attention.jax.nn,
        "dot_product_attention",
        lambda *args, **kwargs: pytest.fail("long opt-in input must not reach quadratic XLA attention"),
    )

    with pytest.raises(ValueError, match="Refusing the quadratic XLA fallback"):
        attention.dot_product_attention(q, k, v, mask, True, 256)


def test_default_off_rocm_refuses_long_quadratic_fallback(monkeypatch):
    q = _array((1, 512, 16, 256), jnp.bfloat16)
    k = _array((1, 512, 4, 256), jnp.bfloat16)
    v = _array((1, 512, 4, 256), jnp.bfloat16)
    mask = _array((1, 512), jnp.int32)

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: False)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: False)
    monkeypatch.setattr(attention, "_can_use_rocm_pallas_attention", lambda *args: False)
    monkeypatch.setattr(
        attention.jax.nn,
        "dot_product_attention",
        lambda *args, **kwargs: pytest.fail("long default-off input must not reach quadratic XLA attention"),
    )

    with pytest.raises(ValueError, match="SKYRL_ROCM_PALLAS_ATTENTION=1"):
        attention.dot_product_attention(q, k, v, mask, True, 256)


def test_explicit_pallas_opt_in_refuses_sequence_above_gfx1100_safety_limit(monkeypatch):
    q = _array((1, 32768, 16, 256), jnp.bfloat16)
    k = _array((1, 32768, 4, 256), jnp.bfloat16)
    v = _array((1, 32768, 4, 256), jnp.bfloat16)
    mask = _array((1, 32768), jnp.int32)

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: False)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: True)

    with pytest.raises(ValueError, match="exceeds the validated 16384-token safety limit"):
        attention.dot_product_attention(q, k, v, mask, True, 256)


def test_rocm_safety_limit_applies_even_when_pallas_is_disabled(monkeypatch):
    q = _array((1, 32768, 16, 256), jnp.bfloat16)
    k = _array((1, 32768, 4, 256), jnp.bfloat16)
    v = _array((1, 32768, 4, 256), jnp.bfloat16)
    mask = _array((1, 32768), jnp.int32)

    monkeypatch.setattr(attention, "_has_cuda_backend", lambda: False)
    monkeypatch.setattr(attention, "_has_rocm_backend", lambda: True)
    monkeypatch.setattr(attention, "_rocm_pallas_attention_enabled", lambda: False)
    monkeypatch.setattr(
        attention.jax.nn,
        "dot_product_attention",
        lambda *args, **kwargs: pytest.fail("unsafe 32K ROCm input must not reach the generic fallback"),
    )

    with pytest.raises(ValueError, match="exceeds the validated 16384-token safety limit"):
        attention.dot_product_attention(q, k, v, mask, True, 256)


def test_rocm_pallas_attention_expands_gqa_and_uses_mask_as_segment_ids(monkeypatch):
    captured = {}
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        q = jnp.zeros((1, 64, 4, 64), dtype=jnp.float16)
        k = jnp.broadcast_to(jnp.asarray([10, 20], dtype=jnp.float16)[None, None, :, None], (1, 64, 2, 64))
        v = jnp.broadcast_to(jnp.asarray([30, 40], dtype=jnp.float16)[None, None, :, None], (1, 64, 2, 64))
        mask = jnp.asarray([[1] * 24 + [2] * 24 + [0] * 16], dtype=jnp.int32)

        def fake_pallas_mha(q_arg, k_arg, v_arg, segment_ids_arg, scale_arg):
            captured.update(q=q_arg, k=k_arg, v=v_arg, segment_ids=segment_ids_arg, scale=scale_arg)
            return q_arg

        monkeypatch.setattr(attention, "_pallas_mha", fake_pallas_mha)
        result = attention._rocm_pallas_attention(q, k, v, mask, 0.125)

    assert result is q
    assert captured["k"].shape == q.shape
    assert captured["v"].shape == q.shape
    np.testing.assert_array_equal(np.asarray(captured["k"][0, 0, :, 0]), [10, 10, 20, 20])
    np.testing.assert_array_equal(np.asarray(captured["v"][0, 0, :, 0]), [30, 30, 40, 40])
    np.testing.assert_array_equal(np.asarray(captured["segment_ids"]), np.asarray(mask != 0))
    assert captured["segment_ids"].dtype == jnp.int32
    assert captured["scale"] == 0.125


def test_pallas_mha_uses_validated_rocm_compiler_contract(monkeypatch):
    from jax.experimental.pallas.ops.gpu import attention as pallas_attention

    captured = {}
    expected = object()

    def fake_mha(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return expected

    monkeypatch.setattr(pallas_attention, "mha", fake_mha)
    # The API guard examines the callable signature, so preserve the production
    # keyword surface on the test double.
    fake_mha.__signature__ = __import__("inspect").signature(
        lambda q, k, v, segment_ids, *, block_sizes, backward_pass_impl, num_warps, num_stages, **kwargs: None
    )

    q, k, v, segment_ids = object(), object(), object(), object()
    result = attention._pallas_mha(q, k, v, segment_ids, 0.0625)

    assert result is expected
    block_sizes = captured["kwargs"]["block_sizes"]
    assert block_sizes.block_q == 64
    assert block_sizes.block_k == 64
    assert block_sizes.block_q_dkv == 32
    assert block_sizes.block_kv_dkv == 32
    assert block_sizes.block_q_dq == 32
    assert block_sizes.block_kv_dq == 32
    assert captured["kwargs"]["backward_pass_impl"] == "triton"
    assert captured["kwargs"]["num_warps"] == 4
    assert captured["kwargs"]["num_stages"] == 1
    assert captured["kwargs"]["causal"] is True
    assert captured["kwargs"]["sm_scale"] == 0.0625


def test_pallas_mha_interpret_forward_and_gradients_match_reference():
    from jax.experimental.pallas.ops.gpu.attention import BlockSizes, mha, mha_reference

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        key = jax.random.key(0)
        q, k, v = [
            jax.random.normal(subkey, (1, 8, 2, 16), dtype=jnp.float32)
            for subkey in jax.random.split(key, 3)
        ]
        segment_ids = jnp.asarray([[1, 1, 1, 1, 1, 1, 0, 0]], dtype=jnp.int32)
        block_sizes = BlockSizes(8, 8, 4, 4, 4, 4)

        def actual(q_arg, k_arg, v_arg):
            return mha(
                q_arg,
                k_arg,
                v_arg,
                segment_ids,
                sm_scale=0.25,
                causal=True,
                block_sizes=block_sizes,
                interpret=True,
            )

        def reference(q_arg, k_arg, v_arg):
            return mha_reference(
                q_arg,
                k_arg,
                v_arg,
                segment_ids,
                sm_scale=0.25,
                causal=True,
            )

        def generic(q_arg, k_arg, v_arg):
            return jax.nn.dot_product_attention(
                q_arg,
                k_arg,
                v_arg,
                scale=0.25,
                mask=segment_ids[:, None, None, :].astype(bool),
                is_causal=True,
            )

        actual_out = actual(q, k, v)
        reference_out = reference(q, k, v)
        generic_out = generic(q, k, v)
        actual_grads = jax.grad(lambda *args: jnp.sum(actual(*args) ** 2), argnums=(0, 1, 2))(q, k, v)
        reference_grads = jax.grad(lambda *args: jnp.sum(reference(*args) ** 2), argnums=(0, 1, 2))(q, k, v)
        valid_query_mask = segment_ids[:, :, None, None].astype(jnp.float32)
        actual_masked_grads = jax.grad(
            lambda *args: jnp.sum(actual(*args) ** 2 * valid_query_mask), argnums=(0, 1, 2)
        )(q, k, v)
        generic_masked_grads = jax.grad(
            lambda *args: jnp.sum(generic(*args) ** 2 * valid_query_mask), argnums=(0, 1, 2)
        )(q, k, v)

    np.testing.assert_allclose(np.asarray(actual_out), np.asarray(reference_out), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(actual_out[:, :6]), np.asarray(generic_out[:, :6]), rtol=2e-5, atol=2e-5)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(reference_grad), rtol=3e-5, atol=3e-5)
    for actual_grad, generic_grad in zip(actual_masked_grads, generic_masked_grads, strict=True):
        np.testing.assert_allclose(np.asarray(actual_grad), np.asarray(generic_grad), rtol=3e-5, atol=3e-5)
