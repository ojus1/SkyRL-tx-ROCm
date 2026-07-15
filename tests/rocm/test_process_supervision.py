from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rocm import process_supervision as supervision


class _FakeProcess:
    def __init__(self, events: list[str], *, pid: int = 424_242, returncode: int = 0):
        self.pid = pid
        self._events = events
        self._returncode = returncode
        self.wait_calls = 0
        self.kill_calls = 0

    def poll(self):
        raise AssertionError("Popen.poll must never be used")

    def wait(self, *, timeout: float):
        assert timeout > 0
        self.wait_calls += 1
        self._events.append("reap")
        return self._returncode

    def kill(self):
        self.kill_calls += 1
        self._events.append("direct_kill")


class _FakeScope:
    def __init__(self, events: list[str], *, populated: bool = True):
        self.path = Path("/fake/current/strict-child")
        self.membership = "/fake/current/strict-child"
        self.procs_fd = 91
        self.kill_writes = 0
        self.removed = False
        self.live = populated
        self._events = events
        self.fail_kill = False

    @property
    def resources_closed(self) -> bool:
        return self.removed

    def populated(self) -> bool:
        self._events.append("populated")
        return self.live

    def hard_kill(self) -> None:
        self._events.append("cgroup_kill")
        if self.fail_kill:
            raise OSError("cgroup.kill failed")
        self.kill_writes += 1
        self.live = False

    def cleanup(self):
        self._events.append("cleanup")
        self.removed = True
        return ()


def _wait_result(pid: int, returncode: int = 0):
    if returncode >= 0:
        return SimpleNamespace(
            si_pid=pid,
            si_code=os.CLD_EXITED,
            si_status=returncode,
        )
    return SimpleNamespace(
        si_pid=pid,
        si_code=os.CLD_KILLED,
        si_status=-returncode,
    )


def _fake_supervisor(
    monkeypatch,
    *,
    live: bool = True,
    returncode: int = 0,
    pidfd: int = -1,
):
    events: list[str] = []
    process = _FakeProcess(events, returncode=returncode)
    scope = _FakeScope(events, populated=live)
    monkeypatch.setattr(
        supervision, "_read_pid_start_time_ticks", lambda _pid, _root: 123
    )
    monkeypatch.setattr(
        supervision, "_read_pid_cgroup", lambda _pid, _root: scope.membership
    )

    def waitid(_kind, _pidfd, _options):
        events.append("waitid")
        return None if scope.live else _wait_result(process.pid, returncode)

    monkeypatch.setattr(supervision.os, "waitid", waitid)
    wrapped = supervision.WrappedProcessSupervisor(
        process=process,  # type: ignore[arg-type]
        pidfd=pidfd,
        scope=scope,  # type: ignore[arg-type]
        proc_root=Path("/fake/proc"),
    )
    return wrapped, process, scope, events


def test_default_containment_hard_kills_before_exactly_one_reap(monkeypatch):
    wrapped, process, scope, events = _fake_supervisor(monkeypatch)

    report = wrapped.contain(0.2)

    assert report.ok
    assert report.cgroup_kill_writes == 1
    assert report.process_group_term_sent is False
    assert events.index("cgroup_kill") < events.index("reap")
    assert events.index("reap") < events.index("cleanup")
    assert process.wait_calls == 1
    assert scope.removed


def test_explicit_graceful_term_orders_term_then_kill_then_reap(monkeypatch):
    wrapped, process, _scope, events = _fake_supervisor(monkeypatch)
    monkeypatch.setattr(supervision.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(supervision.os, "getsid", lambda _pid: process.pid)
    monkeypatch.setattr(supervision.os, "getpgrp", lambda: process.pid + 1)

    def killpg(_pgid, signum):
        assert signum == signal.SIGTERM
        events.append("group_term")

    monkeypatch.setattr(supervision.os, "killpg", killpg)

    report = wrapped.contain(0.2, graceful_term_seconds=1e-9)

    assert report.ok
    assert report.process_group_term_sent
    assert events.index("group_term") < events.index("cgroup_kill")
    assert events.index("cgroup_kill") < events.index("reap")


def test_terminal_retry_never_revisits_pid_pgid_or_signals(monkeypatch):
    wrapped, process, _scope, _events = _fake_supervisor(monkeypatch)
    first = wrapped.contain(0.2)
    assert first.ok

    def forbidden(*_args, **_kwargs):
        raise AssertionError("terminal retry revisited a numeric process identity")

    monkeypatch.setattr(supervision, "_read_pid_start_time_ticks", forbidden)
    monkeypatch.setattr(supervision, "_read_pid_cgroup", forbidden)
    monkeypatch.setattr(supervision.os, "waitid", forbidden)
    monkeypatch.setattr(supervision.os, "getpgid", forbidden)
    monkeypatch.setattr(supervision.os, "getsid", forbidden)
    monkeypatch.setattr(supervision.os, "killpg", forbidden)

    second = wrapped.contain(0.2)

    assert second == first
    assert process.wait_calls == 1


def test_cleanup_retry_after_reap_never_revisits_reused_pid(monkeypatch):
    wrapped, process, scope, _events = _fake_supervisor(monkeypatch)
    cleanup_calls = 0

    def flaky_cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            return (
                supervision.SupervisionIssue(
                    phase="cgroup_cleanup",
                    operation="remove_scope",
                    error_type="OSError",
                    message="busy",
                ),
            )
        scope.removed = True
        return ()

    monkeypatch.setattr(scope, "cleanup", flaky_cleanup)

    first = wrapped.contain(0.2)

    assert first.leader_reaped
    assert not first.terminal
    assert process.wait_calls == 1

    def forbidden(*_args, **_kwargs):
        raise AssertionError("cleanup retry touched a potentially reused PID/PGID")

    monkeypatch.setattr(supervision, "_read_pid_start_time_ticks", forbidden)
    monkeypatch.setattr(supervision, "_read_pid_cgroup", forbidden)
    monkeypatch.setattr(supervision.os, "waitid", forbidden)
    monkeypatch.setattr(supervision.os, "getpgid", forbidden)
    monkeypatch.setattr(supervision.os, "getsid", forbidden)
    monkeypatch.setattr(supervision.os, "killpg", forbidden)

    second = wrapped.contain(0.2)

    assert second.terminal
    assert not second.ok  # The first cleanup failure remains in the audit trail.
    assert cleanup_calls == 2
    assert process.wait_calls == 1


def test_ambiguous_pidfd_close_never_closes_reused_descriptor(monkeypatch):
    owned_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    wrapped, _process, _scope, _events = _fake_supervisor(
        monkeypatch, pidfd=owned_fd
    )
    real_close = os.close
    first_close = True

    def close_then_raise(fd: int) -> None:
        nonlocal first_close
        if fd == owned_fd and first_close:
            first_close = False
            real_close(fd)
            raise OSError("ambiguous close")
        real_close(fd)

    monkeypatch.setattr(supervision.os, "close", close_then_raise)

    first = wrapped.contain(0.2)

    assert first.terminal
    assert not first.ok
    reused_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    assert reused_fd == owned_fd
    try:
        wrapped.contain(0.2)
        os.fstat(reused_fd)
    finally:
        real_close(reused_fd)


def test_scope_ambiguous_close_invalidates_before_error(monkeypatch):
    owned_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    scope = supervision._PrivateCgroup.__new__(supervision._PrivateCgroup)
    scope.procs_fd = owned_fd
    real_close = os.close

    def close_then_raise(fd: int) -> None:
        real_close(fd)
        raise OSError("ambiguous close")

    monkeypatch.setattr(supervision.os, "close", close_then_raise)

    with pytest.raises(OSError, match="ambiguous"):
        scope._close_fd("procs_fd")

    assert scope.procs_fd == -1
    reused_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    assert reused_fd == owned_fd
    try:
        scope._close_fd("procs_fd")
        os.fstat(reused_fd)
    finally:
        real_close(reused_fd)


def test_failed_launch_ambiguous_pidfd_close_is_not_retried(monkeypatch):
    events: list[str] = []
    process = _FakeProcess(events)
    scope = _FakeScope(events, populated=False)
    owned_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    failed = supervision._FailedLaunchSupervisor(
        process=process,  # type: ignore[arg-type]
        pidfd=owned_fd,
        scope=scope,  # type: ignore[arg-type]
    )
    real_close = os.close
    first_close = True
    monkeypatch.setattr(
        supervision.signal, "pidfd_send_signal", lambda *_args: None
    )

    def close_then_raise(fd: int) -> None:
        nonlocal first_close
        if fd == owned_fd and first_close:
            first_close = False
            real_close(fd)
            raise OSError("ambiguous close")
        real_close(fd)

    monkeypatch.setattr(supervision.os, "close", close_then_raise)

    report = failed.contain(0.2)

    assert report.terminal
    assert not report.ok
    reused_fd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    assert reused_fd == owned_fd
    try:
        failed.contain(0.2)
        os.fstat(reused_fd)
    finally:
        real_close(reused_fd)

def test_echild_disables_group_signal_and_cannot_claim_terminal(monkeypatch):
    wrapped, process, scope, _events = _fake_supervisor(monkeypatch)
    group_calls = 0

    def no_child(*_args):
        raise ChildProcessError("already reaped")

    def killpg(*_args):
        nonlocal group_calls
        group_calls += 1

    monkeypatch.setattr(supervision.os, "waitid", no_child)
    monkeypatch.setattr(supervision.os, "killpg", killpg)

    report = wrapped.contain(0.2, graceful_term_seconds=0.01)

    assert not report.ok
    assert not report.terminal
    assert report.cgroup_empty
    assert not report.leader_reaped
    assert group_calls == 0
    assert process.wait_calls == 0
    assert not scope.removed
    assert any(issue.operation == "waitid_wnowait" for issue in report.issues)


def test_generic_waitid_failure_disables_group_fallback(monkeypatch):
    wrapped, _process, scope, _events = _fake_supervisor(monkeypatch)
    scope.fail_kill = True
    group_calls = 0

    monkeypatch.setattr(
        supervision.os,
        "waitid",
        lambda *_args: (_ for _ in ()).throw(OSError("waitid failed")),
    )

    def killpg(*_args):
        nonlocal group_calls
        group_calls += 1

    monkeypatch.setattr(supervision.os, "killpg", killpg)

    report = wrapped.contain(0.002)

    assert not report.ok
    assert not report.terminal
    assert group_calls == 0
    assert any(issue.operation == "cgroup_kill" for issue in report.issues)


def test_waitid_failure_still_uses_retained_pidfd_not_numeric_pid(monkeypatch):
    wrapped, _process, scope, _events = _fake_supervisor(monkeypatch, pidfd=88)
    scope.fail_kill = True
    pidfd_signals: list[tuple[int, int]] = []
    group_calls = 0

    monkeypatch.setattr(
        supervision.os,
        "waitid",
        lambda *_args: (_ for _ in ()).throw(OSError("waitid failed")),
    )
    monkeypatch.setattr(
        supervision.signal,
        "pidfd_send_signal",
        lambda fd, signum: pidfd_signals.append((fd, signum)),
    )

    def killpg(*_args):
        nonlocal group_calls
        group_calls += 1

    monkeypatch.setattr(supervision.os, "killpg", killpg)

    report = wrapped.contain(0.002)

    assert not report.terminal
    assert pidfd_signals
    assert set(pidfd_signals) == {(88, signal.SIGKILL)}
    assert group_calls == 0


def test_final_leader_binding_drift_is_terminal_but_unprovable(monkeypatch):
    wrapped, _process, _scope, _events = _fake_supervisor(monkeypatch)
    monkeypatch.setattr(
        supervision, "_read_pid_cgroup", lambda _pid, _root: "/escaped"
    )

    report = wrapped.contain(0.2)

    assert report.terminal
    assert not report.ok
    assert report.leader_reaped
    assert any(issue.operation == "leader_cgroup_drift" for issue in report.issues)


def test_stuck_populated_scope_is_nonterminal_and_retryable(monkeypatch):
    wrapped, process, scope, _events = _fake_supervisor(monkeypatch)

    def lagging_kill():
        scope.kill_writes += 1

    monkeypatch.setattr(scope, "hard_kill", lagging_kill)

    first = wrapped.contain(0.002)

    assert not first.terminal
    assert not first.cgroup_empty
    assert not first.leader_reaped
    assert not scope.removed
    assert process.wait_calls == 0

    scope.live = False
    second = wrapped.contain(0.2)

    assert second.terminal
    assert second.leader_reaped
    assert not second.ok  # The bounded first-attempt timeout remains evidence.
    assert process.wait_calls == 1


def test_failed_cgroup_kill_uses_only_freshly_proven_group_fallback(monkeypatch):
    wrapped, process, scope, events = _fake_supervisor(monkeypatch)
    scope.fail_kill = True
    monkeypatch.setattr(supervision.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(supervision.os, "getsid", lambda _pid: process.pid)
    monkeypatch.setattr(supervision.os, "getpgrp", lambda: process.pid + 1)

    def killpg(_pgid, signum):
        assert signum == signal.SIGKILL
        events.append("group_kill")
        scope.live = False

    monkeypatch.setattr(supervision.os, "killpg", killpg)

    report = wrapped.contain(0.2)

    assert not report.ok  # The failed primary cgroup kill remains evidence.
    assert report.terminal
    assert events.index("cgroup_kill") < events.index("group_kill")
    assert events.index("group_kill") < events.index("reap")


def test_scope_kill_rejects_owner_membership_drift_before_write(monkeypatch):
    scope = supervision._PrivateCgroup.__new__(supervision._PrivateCgroup)
    scope.removed = False
    scope.identity = (1, 2)
    scope.parent_identity = (3, 4)
    scope.parent_fd = 10
    scope.kill_fd = 11
    scope.owner_pid = 123
    scope.parent_membership = "/safe/parent"
    scope.proc_root = Path("/fake/proc")
    scope.name = "strict-child"
    scope.kill_writes = 0
    writes: list[bytes] = []

    monkeypatch.setattr(supervision.os, "getpid", lambda: 123)
    monkeypatch.setattr(
        supervision.os,
        "stat",
        lambda *_args, **_kwargs: SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=2
        ),
    )
    monkeypatch.setattr(
        supervision.os,
        "fstat",
        lambda _fd: SimpleNamespace(st_dev=3, st_ino=4),
    )
    monkeypatch.setattr(
        supervision,
        "_read_pid_cgroup",
        lambda _pid, _root: "/safe/parent/strict-child",
    )
    monkeypatch.setattr(
        supervision.os, "write", lambda _fd, value: writes.append(value) or len(value)
    )

    with pytest.raises(RuntimeError, match="supervisor moved"):
        scope.hard_kill()

    assert writes == []


def test_duplicate_populated_records_are_rejected(monkeypatch):
    scope = supervision._PrivateCgroup.__new__(supervision._PrivateCgroup)
    scope.events_fd = 22
    monkeypatch.setattr(scope, "verify_identity", lambda: None)
    monkeypatch.setattr(
        supervision.os,
        "pread",
        lambda _fd, _size, _offset: b"populated 0\npopulated 1\n",
    )

    with pytest.raises(ValueError, match="exact populated"):
        scope.populated()


@pytest.mark.parametrize(
    "pass_fds",
    ((True,), (1,), (999_999,), (3, 3)),
)
def test_pass_fd_validation_rejects_bool_stdio_closed_and_duplicate(
    pass_fds, monkeypatch
):
    if pass_fds == (3, 3):
        monkeypatch.setattr(supervision.os, "fstat", lambda _fd: object())
    with pytest.raises(ValueError):
        supervision._validated_pass_fds(pass_fds)


def test_launch_attestation_failure_retains_retryable_supervisor(monkeypatch):
    events: list[str] = []
    scope = _FakeScope(events)
    process = _FakeProcess(events)
    fake_pidfd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    pidfd_signals: list[tuple[int, int]] = []
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(runtime, "_probe_sigchld_waitability", lambda: None)
    monkeypatch.setattr(
        supervision._PrivateCgroup, "create", lambda **_kwargs: scope
    )
    monkeypatch.setattr(supervision.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(supervision.os, "pidfd_open", lambda _pid, _flags: fake_pidfd)
    monkeypatch.setattr(
        supervision, "_read_pid_start_time_ticks", lambda _pid, _root: 123
    )
    monkeypatch.setattr(supervision, "_read_pid_parent", lambda _pid, _root: -1)
    monkeypatch.setattr(
        supervision.signal,
        "pidfd_send_signal",
        lambda fd, signum: pidfd_signals.append((fd, signum)),
    )

    with pytest.raises(supervision.ProcessSupervisionError) as raised:
        runtime.launch(["attestation-fails"])

    assert raised.value.containment is not None
    assert raised.value.containment.terminal
    assert raised.value.supervisor.terminal
    assert process.wait_calls == 1
    assert process.kill_calls == 0
    assert pidfd_signals == [(fake_pidfd, signal.SIGKILL)]


def test_process_present_failed_launch_is_retained_until_retry(monkeypatch):
    events: list[str] = []
    scope = _FakeScope(events)
    process = _FakeProcess(events)
    fake_pidfd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    runtime._saved_sigchld = signal.SIG_DFL
    first_read = True

    def populated():
        nonlocal first_read
        if first_read:
            first_read = False
            raise OSError("events unreadable")
        return scope.live

    def kill_without_settling():
        scope.kill_writes += 1

    monkeypatch.setattr(scope, "populated", populated)
    monkeypatch.setattr(scope, "hard_kill", kill_without_settling)
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(runtime, "_probe_sigchld_waitability", lambda: None)
    monkeypatch.setattr(
        supervision._PrivateCgroup, "create", lambda **_kwargs: scope
    )
    monkeypatch.setattr(supervision.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(supervision.os, "pidfd_open", lambda _pid, _flags: fake_pidfd)
    monkeypatch.setattr(
        supervision, "_read_pid_start_time_ticks", lambda _pid, _root: 123
    )
    monkeypatch.setattr(supervision, "_read_pid_parent", lambda _pid, _root: -1)
    monkeypatch.setattr(supervision.signal, "pidfd_send_signal", lambda *_args: None)

    with pytest.raises(supervision.ProcessSupervisionError) as raised:
        runtime.launch(["attestation-fails-nonterminal"])

    assert not raised.value.containment.terminal
    assert raised.value.supervisor in runtime._supervisors
    assert process.wait_calls == 0
    assert not scope.removed

    scope.live = False
    report = runtime.restore()

    assert report.containment[-1].terminal
    assert process.wait_calls == 1
    assert scope.removed


def test_popen_failure_scope_is_retained_until_exact_empty_retry(monkeypatch):
    events: list[str] = []
    scope = _FakeScope(events)
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    runtime._saved_sigchld = signal.SIG_DFL
    first_read = True

    def populated():
        nonlocal first_read
        if first_read:
            first_read = False
            raise OSError("events unreadable")
        return scope.live

    def kill_without_settling():
        scope.kill_writes += 1

    monkeypatch.setattr(scope, "populated", populated)
    monkeypatch.setattr(scope, "hard_kill", kill_without_settling)
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(runtime, "_probe_sigchld_waitability", lambda: None)
    monkeypatch.setattr(
        supervision._PrivateCgroup, "create", lambda **_kwargs: scope
    )
    monkeypatch.setattr(
        supervision.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.SubprocessError("post-migration preexec failed")
        ),
    )

    with pytest.raises(supervision.ProcessSupervisionError) as raised:
        runtime.launch(["popen-fails-nonterminal"])

    assert not raised.value.containment.terminal
    assert raised.value.supervisor in runtime._supervisors
    assert not scope.removed
    assert "cleanup" not in events

    scope.live = False
    report = runtime.restore()

    assert report.containment[-1].terminal
    assert scope.removed


def test_pre_spawn_check_is_last_and_rejection_never_launches(monkeypatch):
    events: list[str] = []
    scope = _FakeScope(events, populated=False)
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(runtime, "_probe_sigchld_waitability", lambda: None)
    monkeypatch.setattr(
        supervision._PrivateCgroup, "create", lambda **_kwargs: scope
    )

    def reject():
        events.append("pre_spawn_check")
        raise RuntimeError("safety sample stale")

    monkeypatch.setattr(
        supervision.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Popen called after rejected pre-spawn check")
        ),
    )

    with pytest.raises(supervision.ProcessSupervisionError) as raised:
        runtime.launch(["never"], pre_spawn_check=reject)

    assert events[0] == "pre_spawn_check"
    assert raised.value.containment.ok
    assert scope.removed


def test_launch_identity_rejects_waitid_echild_nonchild(monkeypatch):
    wrapped, process, scope, _events = _fake_supervisor(monkeypatch)
    monkeypatch.setattr(
        supervision, "_read_pid_parent", lambda _pid, _root: os.getpid()
    )
    monkeypatch.setattr(supervision.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(supervision.os, "getsid", lambda _pid: process.pid)
    monkeypatch.setattr(
        supervision.os,
        "waitid",
        lambda *_args: (_ for _ in ()).throw(ChildProcessError("ECHILD")),
    )

    with pytest.raises(ChildProcessError, match="ECHILD"):
        wrapped.verify_launch_identity()

    assert not scope.removed


@pytest.mark.parametrize("prefix", ("../escape", "bad/name", ".", "..", "x" * 65))
def test_scope_prefix_is_one_bounded_safe_component(prefix: str):
    with pytest.raises(ValueError, match="invalid private cgroup prefix"):
        supervision._PrivateCgroup(
            parent=Path("/never-opened"),
            parent_membership="/safe",
            prefix=prefix,
            proc_root=Path("/fake/proc"),
        )


def test_short_preexec_cgroup_move_is_an_error_and_closes_fd(monkeypatch):
    closes: list[int] = []
    monkeypatch.setattr(supervision.os, "write", lambda _fd, _value: 0)
    monkeypatch.setattr(supervision.os, "close", closes.append)

    with pytest.raises(OSError, match="short write"):
        supervision._minimal_move_to_cgroup(19)

    assert closes == [19]


def test_popen_failure_proves_empty_scope_and_cleans_it(monkeypatch):
    events: list[str] = []
    scope = _FakeScope(events, populated=False)
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(runtime, "_probe_sigchld_waitability", lambda: None)
    monkeypatch.setattr(
        supervision._PrivateCgroup,
        "create",
        lambda **_kwargs: scope,
    )
    monkeypatch.setattr(
        supervision.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            subprocess.SubprocessError("preexec failed")
        ),
    )

    with pytest.raises(supervision.ProcessSupervisionError) as raised:
        runtime.launch(["never-executed"])

    assert raised.value.containment is not None
    assert raised.value.containment.ok
    assert raised.value.supervisor.terminal
    assert scope.removed
    assert "cleanup" in events


def test_runtime_reports_sigchld_drift_without_mutating_it(monkeypatch):
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    runtime._saved_sigchld = signal.SIG_DFL
    signal_writes = 0

    monkeypatch.setattr(supervision.signal, "getsignal", lambda _signum: signal.SIG_IGN)

    def forbidden_signal(*_args):
        nonlocal signal_writes
        signal_writes += 1

    monkeypatch.setattr(supervision.signal, "signal", forbidden_signal)

    report = runtime.restore()

    assert not report.ok
    assert not report.sigchld_restored
    assert signal_writes == 0
    assert report.issues[-1].operation == "verify_sigchld_unchanged"


def test_runtime_rejects_nondefault_sigchld_before_probe(monkeypatch):
    runtime = supervision.ChildProcessRuntime()
    probe_calls = 0
    monkeypatch.setattr(supervision, "_require_linux_primitives", lambda: None)
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(supervision.signal, "getsignal", lambda _signum: signal.SIG_IGN)

    def probe():
        nonlocal probe_calls
        probe_calls += 1

    monkeypatch.setattr(runtime, "_probe_sigchld_waitability", probe)

    with pytest.raises(supervision.ProcessSupervisionError, match="exact default"):
        runtime.start()

    assert probe_calls == 0


def test_launch_reattests_sigchld_before_creating_scope(monkeypatch):
    runtime = supervision.ChildProcessRuntime()
    runtime._started = True
    monkeypatch.setattr(supervision, "_require_single_thread", lambda: None)
    monkeypatch.setattr(supervision.signal, "getsignal", lambda _signum: signal.SIG_DFL)
    monkeypatch.setattr(
        runtime,
        "_probe_sigchld_waitability",
        lambda: (_ for _ in ()).throw(
            supervision.ProcessSupervisionError("effective SIGCHLD drift")
        ),
    )
    monkeypatch.setattr(
        supervision._PrivateCgroup,
        "create",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("scope created before SIGCHLD re-attestation")
        ),
    )

    with pytest.raises(supervision.ProcessSupervisionError, match="effective"):
        runtime.launch(["never-launched"])


def test_effective_sigchld_probe_rejects_echild(monkeypatch):
    runtime = supervision.ChildProcessRuntime()
    fake_pidfd = os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)

    class _ReadyPoll:
        def register(self, _fd, _events):
            pass

        def poll(self, _timeout):
            return [(fake_pidfd, supervision.select.POLLIN)]

    monkeypatch.setattr(supervision.os, "fork", lambda: 123_456)
    monkeypatch.setattr(
        supervision.os, "pidfd_open", lambda _pid, _flags: fake_pidfd
    )
    monkeypatch.setattr(
        supervision.os,
        "waitid",
        lambda *_args: (_ for _ in ()).throw(ChildProcessError("ECHILD")),
    )
    monkeypatch.setattr(supervision.os, "waitpid", lambda _pid, _flags: (123_456, 0))
    monkeypatch.setattr(supervision.signal, "pidfd_send_signal", lambda *_args: None)
    monkeypatch.setattr(supervision.select, "poll", _ReadyPoll)

    with pytest.raises(
        supervision.ProcessSupervisionError, match="not reliably waitable"
    ):
        runtime._probe_sigchld_waitability()


def test_probe_without_pidfd_raw_kills_before_releasing_gate(monkeypatch):
    runtime = supervision.ChildProcessRuntime()
    events: list[str] = []
    monkeypatch.setattr(supervision.os, "pipe2", lambda _flags: (10, 11))
    monkeypatch.setattr(supervision.os, "fork", lambda: 123_456)
    monkeypatch.setattr(
        supervision.os,
        "pidfd_open",
        lambda *_args: (_ for _ in ()).throw(OSError("pidfd")),
    )
    monkeypatch.setattr(
        supervision.os, "close", lambda fd: events.append(f"close:{fd}")
    )
    monkeypatch.setattr(
        supervision.os,
        "kill",
        lambda pid, signum: events.append(f"kill:{pid}:{signum}"),
    )
    monkeypatch.setattr(
        supervision.os,
        "waitpid",
        lambda pid, _flags: (events.append(f"wait:{pid}") or (pid, 0)),
    )

    with pytest.raises(supervision.ProcessSupervisionError):
        runtime._probe_sigchld_waitability()

    assert events.index(f"kill:123456:{signal.SIGKILL}") < events.index("close:11")


def test_probe_with_pidfd_never_raw_signals_after_gate_release(monkeypatch):
    runtime = supervision.ChildProcessRuntime()
    events: list[str] = []
    monkeypatch.setattr(supervision.os, "pipe2", lambda _flags: (10, 11))
    monkeypatch.setattr(supervision.os, "fork", lambda: 123_456)
    monkeypatch.setattr(supervision.os, "pidfd_open", lambda *_args: 77)
    monkeypatch.setattr(supervision.os, "write", lambda _fd, _value: 1)
    monkeypatch.setattr(
        supervision.os, "close", lambda fd: events.append(f"close:{fd}")
    )
    monkeypatch.setattr(
        supervision.os,
        "waitid",
        lambda *_args: (_ for _ in ()).throw(OSError("waitid")),
    )
    monkeypatch.setattr(
        supervision.signal,
        "pidfd_send_signal",
        lambda *_args: (_ for _ in ()).throw(OSError("pidfd signal")),
    )
    monkeypatch.setattr(
        supervision.os,
        "kill",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("raw signal after gate release")
        ),
    )
    monkeypatch.setattr(supervision.os, "waitpid", lambda pid, _flags: (pid, 0))

    with pytest.raises(supervision.ProcessSupervisionError):
        runtime._probe_sigchld_waitability()

    assert "close:11" in events


@pytest.mark.parametrize("failure", ("pidfd_open", "gate_and_pidfd_signal"))
def test_sigchld_probe_failure_cleanup_is_bounded_in_subprocess(failure: str):
    source = """
import os, sys
from rocm import process_supervision as supervision
mode = sys.argv[1]
runtime = supervision.ChildProcessRuntime()
if mode == 'pidfd_open':
    supervision.os.pidfd_open = lambda *_args: (_ for _ in ()).throw(OSError('pidfd'))
else:
    original_write = supervision.os.write
    def fail_gate(fd, value):
        if value == b'1':
            raise OSError('gate')
        return original_write(fd, value)
    supervision.os.write = fail_gate
    supervision.signal.pidfd_send_signal = lambda *_args: (_ for _ in ()).throw(OSError('signal'))
try:
    runtime._probe_sigchld_waitability()
except supervision.ProcessSupervisionError:
    raise SystemExit(0)
raise SystemExit(2)
"""
    repo = Path(__file__).parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    result = subprocess.run(
        [sys.executable, "-c", source, failure],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_context_exit_flattens_containment_issues(monkeypatch):
    containment_issue = supervision.SupervisionIssue(
        phase="contain",
        operation="proof",
        error_type="InvariantError",
        message="not proven",
    )
    containment = supervision.ContainmentReport(
        pid=99,
        observed_returncode=None,
        reaped_returncode=None,
        cgroup_path="/fake/scope",
        cgroup_kill_writes=1,
        process_group_term_sent=False,
        cgroup_empty=False,
        leader_reaped=False,
        terminal=False,
        issues=(containment_issue,),
    )

    class _Supervisor:
        def contain(self, _timeout):
            return containment

    runtime = supervision.ChildProcessRuntime()
    runtime._supervisors = [_Supervisor()]  # type: ignore[list-item]
    monkeypatch.setattr(supervision.signal, "getsignal", lambda _signum: signal.SIG_DFL)

    with pytest.raises(supervision.ProcessSupervisionError) as raised:
        runtime.__exit__(None, None, None)

    assert containment_issue in raised.value.issues


def test_context_exit_groups_original_and_containment_failures(monkeypatch):
    containment_issue = supervision.SupervisionIssue(
        phase="contain",
        operation="proof",
        error_type="InvariantError",
        message="not proven",
    )
    containment = supervision.ContainmentReport(
        pid=100,
        observed_returncode=None,
        reaped_returncode=None,
        cgroup_path="/fake/scope",
        cgroup_kill_writes=1,
        process_group_term_sent=False,
        cgroup_empty=False,
        leader_reaped=False,
        terminal=False,
        issues=(containment_issue,),
    )

    class _Supervisor:
        def contain(self, _timeout):
            return containment

    runtime = supervision.ChildProcessRuntime()
    runtime._supervisors = [_Supervisor()]  # type: ignore[list-item]
    monkeypatch.setattr(supervision.signal, "getsignal", lambda _signum: signal.SIG_DFL)
    original = KeyboardInterrupt("stop")

    with pytest.raises(BaseExceptionGroup) as raised:
        runtime.__exit__(KeyboardInterrupt, original, None)

    assert original in raised.value.exceptions
    supervision_errors = [
        error
        for error in raised.value.exceptions
        if isinstance(error, supervision.ProcessSupervisionError)
    ]
    assert len(supervision_errors) == 1
    assert containment_issue in supervision_errors[0].issues


_REAL_CGROUP_TESTS = os.environ.get("SKYRL_RUN_REAL_CGROUP_TESTS") == "1"


@pytest.mark.skipif(
    not _REAL_CGROUP_TESTS,
    reason="requires explicit approval for real private-cgroup containment",
)
def test_real_immediate_exit_is_observed_wnowait_then_reaped_once():
    runtime = supervision.ChildProcessRuntime(cgroup_prefix="skyrl-test-immediate")
    runtime.start()
    try:
        wrapped = runtime.launch([sys.executable, "-c", "raise SystemExit(17)"])
        deadline = time.monotonic() + 3
        while wrapped.observe_returncode() is None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert wrapped.returncode == 17

        report = wrapped.contain(2)

        assert report.ok
        assert report.observed_returncode == 17
        assert report.reaped_returncode == 17
        assert report.cgroup_kill_writes == 1
        assert not wrapped.cgroup_path.exists()
    finally:
        restored = runtime.restore()
        assert restored.ok


@pytest.mark.skipif(
    not _REAL_CGROUP_TESTS,
    reason="requires explicit approval for real private-cgroup containment",
)
def test_real_scope_kills_same_group_and_setsid_double_fork(tmp_path: Path):
    pid_file = tmp_path / "descendants.json"
    child_source = """
import json, os, subprocess, sys, time
path = sys.argv[1]
same = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
records = [
    line.split('::', 1)[1]
    for line in open('/proc/self/cgroup')
    if line.startswith('0::')
]
membership = records[0].strip()
nested = '/sys/fs/cgroup' + membership + '/nested-child'
os.mkdir(nested)
nested_fd = os.open(nested + '/cgroup.procs', os.O_WRONLY)
first = os.fork()
if first == 0:
    os.write(nested_fd, b'0')
    os.close(nested_fd)
    os.setsid()
    leaf = os.fork()
    if leaf == 0:
        with open(path + '.leaf.tmp', 'w') as stream:
            stream.write(str(os.getpid()))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path + '.leaf.tmp', path + '.leaf')
        time.sleep(30)
        os._exit(0)
    os._exit(0)
os.close(nested_fd)
os.waitpid(first, 0)
deadline = time.monotonic() + 2
while not os.path.exists(path + '.leaf') and time.monotonic() < deadline:
    time.sleep(0.001)
leaf = int(open(path + '.leaf').read())
with open(path + '.tmp', 'w') as stream:
    json.dump({'same': same.pid, 'leaf': leaf}, stream)
    stream.flush()
    os.fsync(stream.fileno())
os.replace(path + '.tmp', path)
"""
    runtime = supervision.ChildProcessRuntime(cgroup_prefix="skyrl-test-tree")
    runtime.start()
    descendant_pidfds: list[int] = []
    try:
        wrapped = runtime.launch(
            [sys.executable, "-c", child_source, str(pid_file)]
        )
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.001)
        descendants = json.loads(pid_file.read_text())
        descendant_pidfds = [
            os.pidfd_open(int(descendants[name]), 0) for name in ("same", "leaf")
        ]
        while wrapped.observe_returncode() is None and time.monotonic() < deadline:
            time.sleep(0.001)
        assert wrapped.returncode == 0

        report = wrapped.contain(2)

        assert report.ok
        assert report.cgroup_kill_writes == 1
        for pidfd in descendant_pidfds:
            with pytest.raises(ProcessLookupError):
                signal.pidfd_send_signal(pidfd, 0)
    finally:
        for pidfd in descendant_pidfds:
            os.close(pidfd)
        restored = runtime.restore()
        assert restored.ok


@pytest.mark.skipif(
    not _REAL_CGROUP_TESTS,
    reason="requires explicit approval for real private-cgroup containment",
)
def test_real_preexec_failure_leaves_no_private_scope(monkeypatch):
    runtime = supervision.ChildProcessRuntime(cgroup_prefix="skyrl-test-preexec")
    runtime.start()
    parent_membership = supervision._read_unified_cgroup(Path("/proc/self/cgroup"))
    parent = Path("/sys/fs/cgroup") / parent_membership.lstrip("/")
    before = set(parent.glob("skyrl-test-preexec-*"))

    def fail_move(_fd: int) -> None:
        raise OSError("injected preexec failure")

    monkeypatch.setattr(supervision, "_minimal_move_to_cgroup", fail_move)
    try:
        with pytest.raises(supervision.ProcessSupervisionError):
            runtime.launch([sys.executable, "-c", "raise SystemExit(0)"])
        assert set(parent.glob("skyrl-test-preexec-*")) == before
    finally:
        restored = runtime.restore()
        assert restored.ok


@pytest.mark.skipif(
    not _REAL_CGROUP_TESTS,
    reason="requires explicit approval for real private-cgroup containment",
)
def test_real_post_migration_preexec_failure_is_contained(monkeypatch):
    runtime = supervision.ChildProcessRuntime(
        cgroup_prefix="skyrl-test-post-migration"
    )
    runtime.start()
    parent_membership = supervision._read_unified_cgroup(Path("/proc/self/cgroup"))
    parent = Path("/sys/fs/cgroup") / parent_membership.lstrip("/")
    before = set(parent.glob("skyrl-test-post-migration-*"))
    original_move = supervision._minimal_move_to_cgroup

    def move_then_fail(fd: int) -> None:
        original_move(fd)
        raise OSError("injected failure after migration")

    monkeypatch.setattr(supervision, "_minimal_move_to_cgroup", move_then_fail)
    try:
        with pytest.raises(supervision.ProcessSupervisionError) as raised:
            runtime.launch([sys.executable, "-c", "raise SystemExit(0)"])
        assert raised.value.containment is not None
        assert raised.value.containment.cgroup_empty
        assert raised.value.containment.leader_reaped
        assert set(parent.glob("skyrl-test-post-migration-*")) == before
    finally:
        restored = runtime.restore()
        assert restored.ok


@pytest.mark.skipif(
    not _REAL_CGROUP_TESTS,
    reason="requires explicit approval for real private-cgroup containment",
)
def test_real_pass_fd_survives_after_preexec_cgroup_move(tmp_path: Path):
    output = tmp_path / "pass-fd.json"
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    os.write(write_fd, b"bound-token")
    os.close(write_fd)
    runtime = supervision.ChildProcessRuntime(cgroup_prefix="skyrl-test-passfd")
    runtime.start()
    try:
        source = """
import json, os, sys
fd = int(sys.argv[1])
record = {
    'payload': os.read(fd, 64).decode(),
    'cgroup': open('/proc/self/cgroup').read().strip(),
}
with open(sys.argv[2], 'w') as stream:
    json.dump(record, stream)
    stream.flush()
    os.fsync(stream.fileno())
"""
        wrapped = runtime.launch(
            [sys.executable, "-c", source, str(read_fd), str(output)],
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        read_fd = -1
        deadline = time.monotonic() + 3
        while wrapped.observe_returncode() is None and time.monotonic() < deadline:
            time.sleep(0.001)
        record = json.loads(output.read_text())

        report = wrapped.contain(2)

        assert report.ok
        assert record["payload"] == "bound-token"
        expected_membership = str(wrapped.cgroup_path).removeprefix(
            "/sys/fs/cgroup"
        )
        assert record["cgroup"] == f"0::{expected_membership}"
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        restored = runtime.restore()
        assert restored.ok
