"""Unit tests for LogitsProcessorMixin chunked logprobs computation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skyrl.tx.utils.logits_processor import LogitsProcessorMixin
from tests.tx.utils.test_generator import DummyModel


class _TiedLogprobModel(LogitsProcessorMixin):
    def __init__(
        self,
        embedding: jax.Array | None,
        *,
        loss_chunk_size: int,
        vocab_superblock_size: int,
    ) -> None:
        self.embedding = embedding
        self.config = SimpleNamespace(
            loss_chunk_size=loss_chunk_size,
            tied_logprob_vocab_superblock_size=vocab_superblock_size,
            gradient_checkpointing=True,
        )

    def get_model_config(self):
        return self.config

    def get_lm_head(self):
        raise AssertionError("the split tied path must not call the dense lm_head")

    def get_frozen_tied_embedding(self):
        return self.embedding


def assert_chunked_matches_nonchunked(
    hidden_states: jnp.ndarray,
    target_ids: jnp.ndarray,
    chunk_size: int,
    adapter_indices: jnp.ndarray | None = None,
    vocab_size: int = 16,
):
    """Assert chunked and non-chunked paths produce identical results."""
    model_chunked = DummyModel(vocab_size=vocab_size, loss_chunk_size=chunk_size)
    model_nonchunked = DummyModel(vocab_size=vocab_size, loss_chunk_size=0)

    logprobs_chunked = model_chunked.compute_logprobs(hidden_states, target_ids, adapter_indices)
    logprobs_nonchunked = model_nonchunked.compute_logprobs(hidden_states, target_ids, adapter_indices)

    B, T = target_ids.shape
    assert logprobs_chunked.shape == (B, T)
    assert logprobs_nonchunked.shape == (B, T)

    np.testing.assert_allclose(
        np.asarray(logprobs_chunked),
        np.asarray(logprobs_nonchunked),
        rtol=1e-5,
        atol=1e-5,
    )


class TestChunkedLogprobs:
    """Tests for chunked vs non-chunked logprobs computation."""

    @pytest.mark.parametrize(
        "B,T,chunk_size",
        [
            (2, 4, 3),  # chunk doesn't divide evenly, needs padding
            (2, 4, 8),  # chunk equals B*T exactly
            (2, 4, 16),  # chunk larger than B*T
            (1, 8, 3),  # single batch element
            (4, 1, 2),  # single token per sequence
            (1, 1, 1),  # minimal case
        ],
    )
    def test_chunk_boundary_cases(self, B, T, chunk_size):
        """Test various chunk size vs total token relationships."""
        V = 16  # vocab_size = hidden_size for identity lm_head
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        assert_chunked_matches_nonchunked(hidden_states, target_ids, chunk_size, vocab_size=V)

    @pytest.mark.parametrize(
        "B,T,chunk_size,adapter_indices",
        [
            (2, 4, 3, None),  # no adapters
            (2, 4, 3, "arange"),  # different adapter per batch, chunk spans boundary
            (3, 4, 5, "arange"),  # chunk spans multiple batches
            (4, 2, 3, "zeros"),  # all same adapter
        ],
    )
    def test_adapter_indices_handling(self, B, T, chunk_size, adapter_indices):
        """Test adapter indices are correctly mapped across chunk boundaries."""
        V = 16
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        if adapter_indices == "arange":
            adapter_indices = jnp.arange(B, dtype=jnp.int32)
        elif adapter_indices == "zeros":
            adapter_indices = jnp.zeros(B, dtype=jnp.int32)

        assert_chunked_matches_nonchunked(hidden_states, target_ids, chunk_size, adapter_indices, vocab_size=V)

    def test_gradient_checkpointing_flag(self):
        """Gradient checkpointing should not affect forward pass results."""
        B, T, V, chunk_size = 2, 4, 16, 3
        hidden_states = jnp.arange(B * T * V, dtype=jnp.float32).reshape(B, T, V) / (B * T * V)
        target_ids = jnp.arange(B * T, dtype=jnp.int32).reshape(B, T) % V

        model_no_ckpt = DummyModel(vocab_size=V, loss_chunk_size=chunk_size)
        model_no_ckpt.config.gradient_checkpointing = False

        model_ckpt = DummyModel(vocab_size=V, loss_chunk_size=chunk_size)
        model_ckpt.config.gradient_checkpointing = True

        logprobs_no_ckpt = model_no_ckpt.compute_logprobs(hidden_states, target_ids)
        logprobs_ckpt = model_ckpt.compute_logprobs(hidden_states, target_ids)

        np.testing.assert_allclose(
            np.asarray(logprobs_no_ckpt),
            np.asarray(logprobs_ckpt),
            rtol=1e-5,
            atol=1e-5,
        )


def test_default_off_split_tied_path_preserves_existing_dense_chunking() -> None:
    model = DummyModel(vocab_size=16, loss_chunk_size=8)
    hidden = jnp.arange(2 * 7 * 16, dtype=jnp.float32).reshape(2, 7, 16) / 100
    targets = jnp.arange(14, dtype=jnp.int32).reshape(2, 7) % 16

    assert model.config.tied_logprob_vocab_superblock_size == 0
    actual = model.compute_logprobs(hidden, targets)
    expected = model.logits_to_logprobs(hidden, targets)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_opt_in_split_tied_path_matches_dense_forward_and_dhidden() -> None:
    hidden = jax.random.normal(jax.random.key(901), (2, 67, 11)) * 0.2
    embedding = jax.random.normal(jax.random.key(902), (37, 11)) * 0.2
    targets = jax.random.randint(
        jax.random.key(903), (2, 67), 0, embedding.shape[0], dtype=jnp.int32
    )
    adapter_indices = jnp.asarray([3, 7], dtype=jnp.int32)
    model = _TiedLogprobModel(embedding, loss_chunk_size=64, vocab_superblock_size=13)

    actual = model.compute_logprobs(hidden, targets, adapter_indices)
    dense = model.logits_to_logprobs(hidden @ embedding.T, targets)
    np.testing.assert_allclose(actual, dense, rtol=2e-6, atol=2e-6)

    cotangent = jax.random.normal(jax.random.key(904), targets.shape)
    def split_objective(hidden_value, embedding_value):
        split_model = _TiedLogprobModel(
            embedding_value,
            loss_chunk_size=64,
            vocab_superblock_size=13,
        )
        return jnp.sum(split_model.compute_logprobs(hidden_value, targets, adapter_indices) * cotangent)

    actual_dhidden, actual_dembedding = jax.grad(split_objective, argnums=(0, 1))(hidden, embedding)
    dense_dhidden, dense_dembedding = jax.grad(
        lambda hidden_value, embedding_value: jnp.sum(
            model.logits_to_logprobs(hidden_value @ embedding_value.T, targets) * cotangent
        ),
        argnums=(0, 1),
    )(hidden, embedding)
    np.testing.assert_allclose(actual_dhidden, dense_dhidden, rtol=3e-6, atol=3e-6)
    assert float(jnp.linalg.norm(actual_dhidden)) > 0
    np.testing.assert_array_equal(actual_dembedding, jnp.zeros_like(embedding))
    assert float(jnp.linalg.norm(dense_dembedding)) > 0


def test_opt_in_split_tied_path_fails_closed_without_exact_tied_weight() -> None:
    model = _TiedLogprobModel(None, loss_chunk_size=64, vocab_superblock_size=8)
    hidden = jnp.ones((1, 3, 5), dtype=jnp.float32)
    targets = jnp.zeros((1, 3), dtype=jnp.int32)

    with pytest.raises(ValueError, match="exact frozen tied embedding"):
        model.compute_logprobs(hidden, targets)


def test_opt_in_split_tied_path_rejects_dense_only_chunk_geometry() -> None:
    model = _TiedLogprobModel(jnp.eye(5), loss_chunk_size=1024, vocab_superblock_size=8)
    hidden = jnp.ones((1, 3, 5), dtype=jnp.float32)
    targets = jnp.zeros((1, 3), dtype=jnp.int32)

    with pytest.raises(ValueError, match="require loss_chunk_size"):
        model.compute_logprobs(hidden, targets)


def test_negative_split_tied_superblock_does_not_silently_disable() -> None:
    model = _TiedLogprobModel(jnp.eye(5), loss_chunk_size=64, vocab_superblock_size=-1)
    hidden = jnp.ones((1, 3, 5), dtype=jnp.float32)
    targets = jnp.zeros((1, 3), dtype=jnp.int32)

    with pytest.raises(ValueError, match="must be nonnegative"):
        model.compute_logprobs(hidden, targets)
