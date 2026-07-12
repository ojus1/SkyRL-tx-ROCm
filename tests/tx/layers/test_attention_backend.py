from types import SimpleNamespace

from skyrl.tx.layers import attention


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
