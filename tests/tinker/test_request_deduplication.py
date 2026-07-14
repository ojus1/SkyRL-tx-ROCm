from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.db_models import (
    CheckpointDB,
    CheckpointStatus,
    FutureDB,
    ModelDB,
    RequestDedupDB,
    SamplingSessionDB,
    SessionDB,
    enable_sqlite_wal,
)


def _run(coroutine):
    return asyncio.run(coroutine)


async def _database(tmp_path) -> AsyncEngine:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'dedup.db'}")
    enable_sqlite_wal(engine.sync_engine)
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _seed_session_and_model(engine: AsyncEngine) -> None:
    async with AsyncSession(engine) as session:
        session.add(
            SessionDB(
                session_id="session_test",
                tags=[],
                user_metadata={},
                sdk_version="0.22.4",
            )
        )
        await session.flush()
        session.add(
            ModelDB(
                model_id="model_test",
                base_model="Qwen/Qwen3.5-4B",
                lora_config=types.LoraConfig(
                    rank=8,
                    alpha=32.0,
                    seed=7,
                    train_mlp=True,
                    train_attn=True,
                    train_unembed=False,
                ).model_dump(mode="json"),
                status="created",
                request_id=0,
                session_id="session_test",
            )
        )
        await session.commit()


def _datum(token: int = 1) -> api.Datum:
    return api.Datum(
        model_input=api.ModelInput(
            chunks=[api.EncodedTextChunk(tokens=[token, token + 1])]
        ),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[token + 1, token + 2]),
            "weights": api.TensorData(data=[1.0, 1.0]),
        },
    )


def _forward_request(*, seq_id: int, token: int = 1) -> api.ForwardBackwardRequest:
    return api.ForwardBackwardRequest(
        model_id="model_test",
        seq_id=seq_id,
        forward_backward_input=api.ForwardBackwardInput(
            data=[_datum(token)],
            loss_fn="cross_entropy",
        ),
    )


def _optim_request(*, seq_id: int, learning_rate: float = 1e-5):
    return api.OptimStepRequest(
        model_id="model_test",
        seq_id=seq_id,
        adam_params=api.AdamParams(learning_rate=learning_rate),
    )


async def _future_rows(engine: AsyncEngine) -> list[FutureDB]:
    async with AsyncSession(engine) as session:
        return list((await session.exec(select(FutureDB))).all())


def test_create_model_commit_loss_replay_returns_one_model_and_future(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            async with AsyncSession(engine) as session:
                session.add(
                    SessionDB(
                        session_id="session_test",
                        tags=[],
                        user_metadata={},
                        sdk_version="0.22.4",
                    )
                )
                await session.commit()
            request = api.CreateModelRequest(
                session_id="session_test",
                model_seq_id=0,
                base_model="Qwen/Qwen3.5-4B",
                lora_config=api.LoRAConfig(
                    rank=8,
                    seed=7,
                    train_mlp=True,
                    train_attn=True,
                    train_unembed=False,
                ),
            )
            async with AsyncSession(engine) as session:
                first = await api.create_model(request, session)
            # Simulate commit followed by a lost HTTP response: a new request
            # session submits the same SDK sequence identity.
            async with AsyncSession(engine) as session:
                replay = await api.create_model(request, session)
            assert replay == first
            async with AsyncSession(engine) as session:
                models = list((await session.exec(select(ModelDB))).all())
                futures = list((await session.exec(select(FutureDB))).all())
                receipts = list((await session.exec(select(RequestDedupDB))).all())
            assert len(models) == len(futures) == len(receipts) == 1
            stored = types.CreateModelInput.model_validate(futures[0].request_data)
            assert stored.lora_config.train_mlp is True
            assert stored.lora_config.train_attn is True
            assert stored.lora_config.train_unembed is False
        finally:
            await engine.dispose()

    _run(scenario())


def test_training_sequence_replay_is_exact_and_cross_operation_reuse_conflicts(
    tmp_path,
):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            forward = _forward_request(seq_id=1)
            async with AsyncSession(engine) as session:
                first_forward = await api.forward_backward(forward, session)
            async with AsyncSession(engine) as session:
                replay_forward = await api.forward_backward(forward, session)
            assert replay_forward == first_forward

            optim = _optim_request(seq_id=2)
            async with AsyncSession(engine) as session:
                first_optim = await api.optim_step(optim, session)
            async with AsyncSession(engine) as session:
                replay_optim = await api.optim_step(optim, session)
            assert replay_optim == first_optim
            assert len(await _future_rows(engine)) == 2

            with pytest.raises(HTTPException) as conflict:
                async with AsyncSession(engine) as session:
                    await api.optim_step(_optim_request(seq_id=1), session)
            assert conflict.value.status_code == 400

            with pytest.raises(HTTPException) as conflict:
                async with AsyncSession(engine) as session:
                    await api.forward_backward(
                        _forward_request(seq_id=1, token=9), session
                    )
            assert conflict.value.status_code == 400
            assert len(await _future_rows(engine)) == 2
        finally:
            await engine.dispose()

    _run(scenario())


def test_concurrent_duplicate_optimizer_submissions_create_one_future(
    tmp_path, monkeypatch
):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            request = _optim_request(seq_id=1)
            owner_reserved = asyncio.Event()
            release_owner = asyncio.Event()

            async def owner():
                request_type = types.RequestType.OPTIM_STEP.value
                request_key = api._sequence_request_identity(
                    "training:model_test", 1
                )
                payload_sha256 = api._request_payload_sha256(request)
                async with AsyncSession(engine) as session:
                    assert (
                        await api._reserve_deduplicated_request(
                            session,
                            request_key=request_key,
                            request_type=request_type,
                            payload_sha256=payload_sha256,
                        )
                        is None
                    )
                    owner_reserved.set()
                    await release_owner.wait()
                    request_id = await api.create_future(
                        session=session,
                        request_type=types.RequestType.OPTIM_STEP,
                        model_id="model_test",
                        request_data=types.OptimStepInput(
                            adam_params=request.adam_params.to_types()
                        ),
                    )
                    response = api.FutureResponse(
                        future_id=str(request_id),
                        request_id=str(request_id),
                    )
                    committed, _ = await api._commit_deduplicated_response(
                        session,
                        request_key=request_key,
                        request_type=request_type,
                        payload_sha256=payload_sha256,
                        response_data=response.model_dump(mode="json"),
                    )
                    return api.FutureResponse.model_validate(committed)

            async def loser():
                await owner_reserved.wait()
                async with AsyncSession(engine) as session:
                    return await api.optim_step(request, session)

            owner_task = asyncio.create_task(owner())
            await owner_reserved.wait()
            loser_entered_reservation = asyncio.Event()
            reserve_request = api._reserve_deduplicated_request

            async def observed_reservation(*args, **kwargs):
                loser_entered_reservation.set()
                return await reserve_request(*args, **kwargs)

            monkeypatch.setattr(
                api, "_reserve_deduplicated_request", observed_reservation
            )
            loser_task = asyncio.create_task(loser())
            await asyncio.wait_for(loser_entered_reservation.wait(), timeout=1)
            assert loser_task.done() is False
            release_owner.set()
            first, second = await asyncio.wait_for(
                asyncio.gather(owner_task, loser_task), timeout=5
            )
            assert first == second
            assert len(await _future_rows(engine)) == 1
        finally:
            await engine.dispose()

    _run(scenario())


def test_concurrent_conflicting_optimizer_payload_has_one_winner(
    tmp_path, monkeypatch
):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            winner = _optim_request(seq_id=1, learning_rate=1e-5)
            conflict = _optim_request(seq_id=1, learning_rate=2e-5)
            owner_reserved = asyncio.Event()
            release_owner = asyncio.Event()

            async def owner():
                request_type = types.RequestType.OPTIM_STEP.value
                request_key = api._sequence_request_identity(
                    "training:model_test", 1
                )
                payload_sha256 = api._request_payload_sha256(winner)
                async with AsyncSession(engine) as session:
                    assert (
                        await api._reserve_deduplicated_request(
                            session,
                            request_key=request_key,
                            request_type=request_type,
                            payload_sha256=payload_sha256,
                        )
                        is None
                    )
                    owner_reserved.set()
                    await release_owner.wait()
                    request_id = await api.create_future(
                        session=session,
                        request_type=types.RequestType.OPTIM_STEP,
                        model_id="model_test",
                        request_data=types.OptimStepInput(
                            adam_params=winner.adam_params.to_types()
                        ),
                    )
                    response = api.FutureResponse(
                        future_id=str(request_id),
                        request_id=str(request_id),
                    )
                    await api._commit_deduplicated_response(
                        session,
                        request_key=request_key,
                        request_type=request_type,
                        payload_sha256=payload_sha256,
                        response_data=response.model_dump(mode="json"),
                    )
                    return response

            async def loser():
                await owner_reserved.wait()
                async with AsyncSession(engine) as session:
                    return await api.optim_step(conflict, session)

            owner_task = asyncio.create_task(owner())
            await owner_reserved.wait()
            loser_entered_reservation = asyncio.Event()
            reserve_request = api._reserve_deduplicated_request

            async def observed_reservation(*args, **kwargs):
                loser_entered_reservation.set()
                return await reserve_request(*args, **kwargs)

            monkeypatch.setattr(
                api, "_reserve_deduplicated_request", observed_reservation
            )
            loser_task = asyncio.create_task(loser())
            await asyncio.wait_for(loser_entered_reservation.wait(), timeout=1)
            assert loser_task.done() is False
            release_owner.set()
            results = await asyncio.wait_for(
                asyncio.gather(owner_task, loser_task, return_exceptions=True),
                timeout=5,
            )
            assert sum(isinstance(result, api.FutureResponse) for result in results) == 1
            conflicts = [
                result for result in results if isinstance(result, HTTPException)
            ]
            assert len(conflicts) == 1
            assert conflicts[0].status_code == 400
            assert len(await _future_rows(engine)) == 1
        finally:
            await engine.dispose()

    _run(scenario())


def test_committed_receipt_replays_after_sqlite_reopen(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        await _seed_session_and_model(engine)
        request = _optim_request(seq_id=1)
        async with AsyncSession(engine) as session:
            first = await api.optim_step(request, session)
        await engine.dispose()

        reopened = await _database(tmp_path)
        try:
            async with AsyncSession(reopened) as session:
                replay = await api.optim_step(request, session)
            assert replay == first
            assert len(await _future_rows(reopened)) == 1
        finally:
            await reopened.dispose()

    _run(scenario())


def test_rolled_back_reservation_can_be_owned_by_retry(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            request = _optim_request(seq_id=1)
            request_type = types.RequestType.OPTIM_STEP.value
            request_key = api._sequence_request_identity("training:model_test", 1)
            payload_sha256 = api._request_payload_sha256(request)
            async with AsyncSession(engine) as session:
                replay = await api._reserve_deduplicated_request(
                    session,
                    request_key=request_key,
                    request_type=request_type,
                    payload_sha256=payload_sha256,
                )
                assert replay is None
                await session.rollback()

            async with AsyncSession(engine) as session:
                result = await api.optim_step(request, session)
            assert result.status == "pending"
            assert len(await _future_rows(engine)) == 1
        finally:
            await engine.dispose()

    _run(scenario())


def test_persisted_incomplete_receipt_fails_closed_without_future(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            request = _optim_request(seq_id=1)
            async with AsyncSession(engine) as session:
                session.add(
                    RequestDedupDB(
                        request_key=api._sequence_request_identity(
                            "training:model_test", 1
                        ),
                        request_type=types.RequestType.OPTIM_STEP.value,
                        payload_sha256=api._request_payload_sha256(request),
                        response_data=None,
                    )
                )
                await session.commit()

            with pytest.raises(HTTPException) as failed:
                async with AsyncSession(engine) as session:
                    await api.optim_step(request, session)
            assert failed.value.status_code == 503
            assert await _future_rows(engine) == []
        finally:
            await engine.dispose()

    _run(scenario())


def test_sqlite_foreign_keys_reject_orphan_checkpoint(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            async with engine.connect() as connection:
                assert (
                    await connection.exec_driver_sql("PRAGMA foreign_keys")
                ).scalar_one() == 1
            with pytest.raises(IntegrityError):
                async with AsyncSession(engine) as session:
                    session.add(
                        CheckpointDB(
                            model_id="missing-model",
                            checkpoint_id="orphan",
                            checkpoint_type=types.CheckpointType.TRAINING,
                            status=CheckpointStatus.PENDING,
                        )
                    )
                    await session.commit()
        finally:
            await engine.dispose()

    _run(scenario())


def test_ephemeral_snapshot_replay_ignores_only_retry_allocated_session_counter(
    tmp_path,
):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            first_request = api.SaveWeightsForSamplerRequest(
                model_id="model_test",
                sampling_session_seq_id=0,
                seq_id=1,
            )
            retried_request = first_request.model_copy(
                update={"sampling_session_seq_id": 1}
            )
            async with AsyncSession(engine) as session:
                first = await api.save_weights_for_sampler(first_request, session)
            async with AsyncSession(engine) as session:
                replay = await api.save_weights_for_sampler(retried_request, session)
            assert replay == first
            async with AsyncSession(engine) as session:
                checkpoints = list(
                    (await session.exec(select(CheckpointDB))).all()
                )
                sampling_sessions = list(
                    (await session.exec(select(SamplingSessionDB))).all()
                )
            assert len(checkpoints) == len(sampling_sessions) == 1
            assert len(await _future_rows(engine)) == 1
        finally:
            await engine.dispose()

    _run(scenario())


def test_sample_commit_loss_replay_creates_one_generator_future(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            await _seed_session_and_model(engine)
            async with AsyncSession(engine) as session:
                session.add(
                    SamplingSessionDB(
                        sampling_session_id="sampling_test",
                        session_id="session_test",
                        sampling_session_seq_id=0,
                        base_model="Qwen/Qwen3.5-4B",
                    )
                )
                await session.commit()
            request = api.SampleRequest(
                sampling_session_id="sampling_test",
                seq_id=0,
                num_samples=2,
                prompt=api.ModelInput(
                    chunks=[api.EncodedTextChunk(tokens=[1, 2, 3])]
                ),
                sampling_params=api.SamplingParams(max_tokens=2, seed=7),
            )
            http_request = SimpleNamespace(
                app=SimpleNamespace(
                    state=SimpleNamespace(external_inference_client=None)
                )
            )
            async with AsyncSession(engine) as session:
                first = await api.asample(request, http_request, session)
            async with AsyncSession(engine) as session:
                replay = await api.asample(request, http_request, session)
            assert replay == first
            assert len(await _future_rows(engine)) == 1
        finally:
            await engine.dispose()

    _run(scenario())


def test_create_session_uses_generated_client_idempotency_header(tmp_path):
    async def scenario():
        engine = await _database(tmp_path)
        try:
            request = api.CreateSessionRequest(
                tags=["gate"], user_metadata={}, sdk_version="0.22.4"
            )
            http_request = Request(
                {
                    "type": "http",
                    "headers": [(b"x-idempotency-key", b"stable-key")],
                }
            )
            async with AsyncSession(engine) as session:
                first = await api.create_session(request, http_request, session)
            async with AsyncSession(engine) as session:
                replay = await api.create_session(request, http_request, session)
            assert replay == first
            async with AsyncSession(engine) as session:
                sessions = list((await session.exec(select(SessionDB))).all())
            assert len(sessions) == 1

            changed = request.model_copy(update={"tags": ["changed"]})
            with pytest.raises(HTTPException) as conflict:
                async with AsyncSession(engine) as session:
                    await api.create_session(changed, http_request, session)
            assert conflict.value.status_code == 400
        finally:
            await engine.dispose()

    _run(scenario())


def test_server_disables_whole_sample_operation_retries():
    assert _run(api.client_config()).sample_no_retries is True
