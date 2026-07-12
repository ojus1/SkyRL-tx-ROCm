import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


_PROBE_PATH = Path(__file__).parents[3] / "rocm" / "probe_pallas_attention.py"
_SPEC = importlib.util.spec_from_file_location("probe_pallas_attention", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


@pytest.mark.parametrize("valid_length", [37, 64])
def test_chunked_reference_matches_generic_gqa(valid_length):
    sequence_length = 64
    query_heads = 4
    kv_heads = 2
    head_dim = 8
    q_key, k_key, v_key, cotangent_key = jax.random.split(jax.random.key(9), 4)
    q = jax.random.normal(q_key, (1, sequence_length, query_heads, head_dim))
    k = jax.random.normal(k_key, (1, sequence_length, kv_heads, head_dim))
    v = jax.random.normal(v_key, (1, sequence_length, kv_heads, head_dim))
    mask = (jnp.arange(sequence_length)[None, :] < valid_length).astype(jnp.int32)
    cotangent = jax.random.normal(cotangent_key, q.shape)
    scale = head_dim**-0.5

    def chunked(*items):
        return _PROBE._chunked_reference_attention(jax, jnp, *items, mask, scale)

    def generic(q_arg, k_arg, v_arg):
        return jax.nn.dot_product_attention(
            q_arg,
            k_arg,
            v_arg,
            mask=mask[:, None, None, :].astype(bool),
            scale=scale,
            is_causal=True,
            implementation="xla",
        )

    chunked_output, chunked_pullback = jax.vjp(chunked, q, k, v)
    generic_output, generic_pullback = jax.vjp(generic, q, k, v)
    np.testing.assert_allclose(chunked_output, generic_output, rtol=2e-6, atol=2e-6)
    for actual, expected in zip(
        chunked_pullback(cotangent), generic_pullback(cotangent), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
