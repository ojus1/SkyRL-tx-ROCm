"""Linux process supervision with unreaped PID identity and cgroup ownership.

The safety contract is deliberately narrow and explicit:

* the caller is a single-threaded Linux process with the default ``SIGCHLD``
  disposition and working pidfd/waitid support;
* the wrapped command is trusted not to move itself out of its private cgroup;
* command completion is observed with ``waitid(..., WNOWAIT)`` and never with
  ``Popen.poll()``;
* containment hard-kills the private cgroup before reaping its leader; and
* after the one reap, retries perform no numeric PID or process-group action.

The dedicated cgroup contains ordinary forks, new sessions, double forks, and
nested child cgroups.  It is not a hostile sandbox: a same-UID process with
write access to an ancestor ``cgroup.procs`` can deliberately migrate out.
Leader binding is therefore checked both after launch and immediately before
reap, and drift is reported as an unprovable containment failure.
"""

from __future__ import annotations

import os
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

_CGROUP_MOUNT = Path("/sys/fs/cgroup")
_SELF_CGROUP = Path("/proc/self/cgroup")
_PROC_ROOT = Path("/proc")
_PROBE_EXIT_STATUS = 73


@dataclass(frozen=True)
class SupervisionIssue:
    phase: str
    operation: str
    error_type: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "operation": self.operation,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class ContainmentReport:
    pid: int
    observed_returncode: int | None
    reaped_returncode: int | None
    cgroup_path: str
    cgroup_kill_writes: int
    process_group_term_sent: bool
    cgroup_empty: bool
    leader_reaped: bool
    terminal: bool
    issues: tuple[SupervisionIssue, ...]

    @property
    def ok(self) -> bool:
        return (
            self.terminal
            and self.cgroup_empty
            and self.leader_reaped
            and not self.issues
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "observed_returncode": self.observed_returncode,
            "reaped_returncode": self.reaped_returncode,
            "cgroup_path": self.cgroup_path,
            "cgroup_kill_writes": self.cgroup_kill_writes,
            "process_group_term_sent": self.process_group_term_sent,
            "cgroup_empty": self.cgroup_empty,
            "leader_reaped": self.leader_reaped,
            "terminal": self.terminal,
            "issues": [issue.as_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class RuntimeReport:
    containment: tuple[ContainmentReport, ...]
    sigchld_restored: bool
    issues: tuple[SupervisionIssue, ...]

    @property
    def ok(self) -> bool:
        return (
            self.sigchld_restored
            and not self.issues
            and all(report.ok for report in self.containment)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "containment": [report.as_dict() for report in self.containment],
            "sigchld_restored": self.sigchld_restored,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class ProcessSupervisionError(RuntimeError):
    """Raised when launch/runtime setup cannot preserve supervision invariants."""

    def __init__(
        self,
        message: str,
        *,
        issues: Sequence[SupervisionIssue] = (),
        containment: ContainmentReport | None = None,
        supervisor: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.issues = tuple(issues)
        self.containment = containment
        self.supervisor = supervisor


def _issue(phase: str, operation: str, error: BaseException | str) -> SupervisionIssue:
    if isinstance(error, BaseException):
        return SupervisionIssue(
            phase=phase,
            operation=operation,
            error_type=type(error).__name__,
            message=str(error),
        )
    return SupervisionIssue(
        phase=phase,
        operation=operation,
        error_type="InvariantError",
        message=error,
    )


def _require_linux_primitives() -> None:
    missing: list[str] = []
    for owner, names in (
        (
            os,
            (
                "fork",
                "pidfd_open",
                "waitid",
                "P_PIDFD",
                "WEXITED",
                "WNOHANG",
                "WNOWAIT",
                "CLD_EXITED",
                "CLD_KILLED",
                "CLD_DUMPED",
            ),
        ),
        (signal, ("pidfd_send_signal",)),
    ):
        for name in names:
            if not hasattr(owner, name):
                missing.append(name)
    if sys.platform != "linux" or missing:
        detail = ", ".join(missing) if missing else sys.platform
        raise ProcessSupervisionError(
            f"Linux pidfd/waitid supervision primitives unavailable: {detail}"
        )


def _require_single_thread() -> None:
    if threading.current_thread() is not threading.main_thread():
        raise ProcessSupervisionError("process supervision must run on the main thread")
    try:
        tasks = [entry for entry in Path("/proc/self/task").iterdir() if entry.name.isdigit()]
    except OSError as error:
        raise ProcessSupervisionError(
            "cannot attest single-threaded preexec safety",
            issues=(_issue("runtime", "enumerate_tasks", error),),
        ) from error
    if len(tasks) != 1:
        raise ProcessSupervisionError(
            f"minimal preexec requires exactly one OS thread, observed {len(tasks)}"
        )


def _decode_waitid(result: Any) -> int:
    if result.si_code == os.CLD_EXITED:
        return int(result.si_status)
    if result.si_code in {os.CLD_KILLED, os.CLD_DUMPED}:
        return -int(result.si_status)
    raise ValueError(f"unexpected waitid si_code {result.si_code!r}")


def _read_unified_cgroup(path: Path) -> str:
    records = path.read_text().splitlines()
    unified = [line.split("::", 1)[1] for line in records if line.startswith("0::")]
    if len(unified) != 1:
        raise ValueError("expected exactly one unified cgroup-v2 membership record")
    member = PurePosixPath(unified[0])
    if not member.is_absolute() or ".." in member.parts:
        raise ValueError("invalid unified cgroup path")
    return member.as_posix()


def _read_pid_cgroup(pid: int, proc_root: Path) -> str:
    return _read_unified_cgroup(proc_root / str(pid) / "cgroup")


def _read_pid_start_time_ticks(pid: int, proc_root: Path) -> int:
    raw = (proc_root / str(pid) / "stat").read_text()
    _prefix, separator, suffix = raw.rpartition(")")
    fields = suffix.split()
    if separator != ")" or len(fields) <= 19:
        raise ValueError("malformed /proc PID stat record")
    return int(fields[19])


def _read_pid_parent(pid: int, proc_root: Path) -> int:
    raw = (proc_root / str(pid) / "stat").read_text()
    _prefix, separator, suffix = raw.rpartition(")")
    fields = suffix.split()
    if separator != ")" or len(fields) <= 1:
        raise ValueError("malformed /proc PID stat record")
    return int(fields[1])


def _validated_pass_fds(pass_fds: Sequence[int]) -> tuple[int, ...]:
    validated: list[int] = []
    seen: set[int] = set()
    for value in pass_fds:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("pass_fds entries must be integer file descriptors")
        if value < 3:
            raise ValueError("pass_fds must not include standard input/output/error")
        if value in seen:
            raise ValueError(f"duplicate pass_fds entry: {value}")
        try:
            os.fstat(value)
        except OSError as error:
            raise ValueError(f"pass_fds entry is not open: {value}") from error
        seen.add(value)
        validated.append(value)
    return tuple(validated)


def _minimal_move_to_cgroup(cgroup_procs_fd: int) -> None:
    """Async-signal-minimal callback used between fork and exec."""
    try:
        written = os.write(cgroup_procs_fd, b"0")
        if written != 1:
            raise OSError("short write to cgroup.procs")
    finally:
        os.close(cgroup_procs_fd)


class _PrivateCgroup:
    def __init__(
        self,
        *,
        parent: Path,
        parent_membership: str,
        prefix: str,
        proc_root: Path,
    ) -> None:
        self.parent = parent
        self.parent_membership = parent_membership.rstrip("/") or "/"
        if (
            re.fullmatch(r"[A-Za-z0-9_.-]+", prefix) is None
            or prefix in {".", ".."}
            or len(prefix) > 64
        ):
            raise ValueError("invalid private cgroup prefix")
        self.name = f"{prefix}-{os.getpid()}-{secrets.token_hex(8)}"
        self.path = parent / self.name
        self.membership = (
            f"/{self.name}"
            if self.parent_membership == "/"
            else f"{self.parent_membership}/{self.name}"
        )
        self.proc_root = proc_root
        self.owner_pid = os.getpid()
        self.parent_fd = -1
        self.directory_fd = -1
        self.procs_fd = -1
        self.kill_fd = -1
        self.events_fd = -1
        self.identity: tuple[int, int] | None = None
        self.parent_identity: tuple[int, int] | None = None
        self.kill_writes = 0
        self.removed = False

        try:
            self.parent_fd = os.open(
                parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            parent_path_stat = os.stat(parent, follow_symlinks=False)
            parent_fd_stat = os.fstat(self.parent_fd)
            self.parent_identity = (parent_fd_stat.st_dev, parent_fd_stat.st_ino)
            if (
                not stat.S_ISDIR(parent_path_stat.st_mode)
                or (parent_path_stat.st_dev, parent_path_stat.st_ino)
                != self.parent_identity
            ):
                raise RuntimeError("current cgroup parent identity changed")
            os.mkdir(self.name, mode=0o700, dir_fd=self.parent_fd)
            self.directory_fd = os.open(
                self.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self.parent_fd,
            )
            directory_stat = os.fstat(self.directory_fd)
            self.identity = (directory_stat.st_dev, directory_stat.st_ino)
            self.procs_fd = os.open(
                "cgroup.procs", os.O_WRONLY | os.O_CLOEXEC, dir_fd=self.directory_fd
            )
            self.kill_fd = os.open(
                "cgroup.kill", os.O_WRONLY | os.O_CLOEXEC, dir_fd=self.directory_fd
            )
            self.events_fd = os.open(
                "cgroup.events", os.O_RDONLY | os.O_CLOEXEC, dir_fd=self.directory_fd
            )
        except BaseException:
            self._best_effort_close()
            try:
                if self.parent_fd >= 0:
                    os.rmdir(self.name, dir_fd=self.parent_fd)
                else:
                    self.path.rmdir()
            except OSError:
                pass
            self._close_fd("parent_fd")
            raise

    @classmethod
    def create(
        cls,
        *,
        cgroup_mount: Path,
        self_cgroup: Path,
        prefix: str,
        proc_root: Path,
    ) -> _PrivateCgroup:
        membership = _read_unified_cgroup(self_cgroup)
        relative = membership.lstrip("/")
        parent = cgroup_mount / relative if relative else cgroup_mount
        parent_stat = parent.stat()
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("current cgroup path is not a directory")
        return cls(
            parent=parent,
            parent_membership=membership,
            prefix=prefix,
            proc_root=proc_root,
        )

    def _close_fd(self, attribute: str) -> None:
        fd = getattr(self, attribute)
        if fd >= 0:
            # Linux close() may release an FD and still report an error.  Drop
            # numeric ownership first so a retry can never close a reused FD.
            setattr(self, attribute, -1)
            os.close(fd)

    @property
    def resources_closed(self) -> bool:
        return all(
            getattr(self, attribute) < 0
            for attribute in (
                "procs_fd",
                "kill_fd",
                "events_fd",
                "directory_fd",
                "parent_fd",
            )
        )

    def _best_effort_close(self) -> None:
        for attribute in (
            "procs_fd",
            "kill_fd",
            "events_fd",
            "directory_fd",
        ):
            try:
                self._close_fd(attribute)
            except OSError:
                pass

    def verify_identity(self) -> None:
        if self.removed or self.identity is None:
            raise RuntimeError("cgroup scope is no longer available")
        observed = os.stat(self.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(observed.st_mode):
            raise RuntimeError("cgroup scope path is not a directory")
        if (observed.st_dev, observed.st_ino) != self.identity:
            raise RuntimeError("cgroup scope path identity changed")

    def verify_owner_outside_scope(self) -> None:
        if os.getpid() != self.owner_pid:
            raise RuntimeError("cgroup controls used by a different process")
        if self.parent_identity is None:
            raise RuntimeError("cgroup parent identity was not captured")
        parent_stat = os.fstat(self.parent_fd)
        if (parent_stat.st_dev, parent_stat.st_ino) != self.parent_identity:
            raise RuntimeError("cgroup parent descriptor identity changed")
        owner_membership = _read_pid_cgroup(self.owner_pid, self.proc_root)
        if owner_membership != self.parent_membership:
            raise RuntimeError(
                "supervisor moved from the attested parent cgroup: "
                f"{owner_membership!r} != {self.parent_membership!r}"
            )

    def populated(self) -> bool:
        self.verify_identity()
        raw = os.pread(self.events_fd, 4096, 0).decode("ascii", "strict")
        populated_values: list[str] = []
        for line in raw.splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[0] == "populated":
                populated_values.append(fields[1])
        if len(populated_values) != 1 or populated_values[0] not in {"0", "1"}:
            raise ValueError("cgroup.events lacks an exact populated value")
        return populated_values[0] == "1"

    def hard_kill(self) -> None:
        self.verify_identity()
        self.verify_owner_outside_scope()
        written = os.write(self.kill_fd, b"1")
        if written != 1:
            raise OSError("short write to cgroup.kill")
        self.kill_writes += 1

    def _remove_descendants(self, directory: Path) -> None:
        children = sorted(
            (entry for entry in directory.iterdir() if entry.is_dir()),
            key=lambda entry: entry.name,
        )
        for child in children:
            self._remove_descendants(child)
            child.rmdir()

    def cleanup(self) -> tuple[SupervisionIssue, ...]:
        if self.removed:
            issues: list[SupervisionIssue] = []
            for attribute in (
                "procs_fd",
                "kill_fd",
                "events_fd",
                "directory_fd",
                "parent_fd",
            ):
                try:
                    self._close_fd(attribute)
                except BaseException as error:
                    issues.append(
                        _issue("cgroup_cleanup", f"close_{attribute}", error)
                    )
            return tuple(issues)
        issues: list[SupervisionIssue] = []
        try:
            self.verify_identity()
            self._remove_descendants(self.path)
        except BaseException as error:
            issues.append(_issue("cgroup_cleanup", "remove_descendants", error))
            return tuple(issues)

        try:
            self.verify_identity()
            os.rmdir(self.name, dir_fd=self.parent_fd)
            self.removed = True
        except BaseException as error:
            issues.append(_issue("cgroup_cleanup", "remove_scope", error))
            return tuple(issues)
        for attribute in (
            "procs_fd",
            "kill_fd",
            "events_fd",
            "directory_fd",
            "parent_fd",
        ):
            try:
                self._close_fd(attribute)
            except BaseException as error:
                issues.append(_issue("cgroup_cleanup", f"close_{attribute}", error))
        return tuple(issues)


class WrappedProcessSupervisor:
    """Own one wrapped command until cgroup containment and exactly one reap."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[Any],
        pidfd: int,
        scope: _PrivateCgroup,
        proc_root: Path,
    ) -> None:
        self._process = process
        self._pidfd = pidfd
        self._scope = scope
        self._proc_root = proc_root
        self.pid = process.pid
        self.pgid = process.pid
        self.sid = process.pid
        self.start_time_ticks = _read_pid_start_time_ticks(self.pid, proc_root)
        self._observed_returncode: int | None = None
        self._observation_complete = False
        self._anchor_safe = True
        self._wait_ownership_lost = False
        self._leader_reaped = False
        self._reap_attempted = False
        self._reaped_returncode: int | None = None
        self._process_group_term_sent = False
        self._terminal = False
        self._cgroup_empty_proven = False
        self._issues: list[SupervisionIssue] = []

    @property
    def returncode(self) -> int | None:
        return self._observed_returncode

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def cgroup_path(self) -> Path:
        return self._scope.path

    def _record(self, phase: str, operation: str, error: BaseException | str) -> None:
        self._issues.append(_issue(phase, operation, error))

    def verify_launch_identity(self) -> None:
        observed_parent = _read_pid_parent(self.pid, self._proc_root)
        if observed_parent != os.getpid():
            raise RuntimeError(
                f"wrapped leader PPID mismatch: {observed_parent} != {os.getpid()}"
            )
        observed_pgid = os.getpgid(self.pid)
        observed_sid = os.getsid(self.pid)
        if observed_pgid != self.pgid or observed_sid != self.sid:
            raise RuntimeError(
                f"start_new_session identity mismatch: pid={self.pid}, "
                f"pgid={observed_pgid}, sid={observed_sid}"
            )
        observed_start = _read_pid_start_time_ticks(self.pid, self._proc_root)
        if observed_start != self.start_time_ticks:
            raise RuntimeError("wrapped leader start-time identity changed")
        observed_cgroup = _read_pid_cgroup(self.pid, self._proc_root)
        if observed_cgroup != self._scope.membership:
            raise RuntimeError(
                f"wrapped leader cgroup mismatch: {observed_cgroup!r} != "
                f"{self._scope.membership!r}"
            )
        # P_PIDFD waitid returns ECHILD for a live non-child too.  A normal
        # WNOHANG return therefore proves direct wait ownership without
        # reaping; an already-exited command is cached under WNOWAIT.
        result = os.waitid(
            os.P_PIDFD,
            self._pidfd,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
        if result is not None:
            if int(result.si_pid) != self.pid:
                raise RuntimeError("launch waitid returned the wrong child identity")
            self._observed_returncode = _decode_waitid(result)
            self._observation_complete = True

    def observe_returncode(self) -> int | None:
        """Observe without reaping; repeated observations are stable."""
        if self._observation_complete:
            return self._observed_returncode
        if self._leader_reaped:
            return self._reaped_returncode
        try:
            result = os.waitid(
                os.P_PIDFD,
                self._pidfd,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except ChildProcessError as error:
            self._anchor_safe = False
            self._wait_ownership_lost = True
            self._observation_complete = True
            self._record("observe", "waitid_wnowait", error)
            return None
        except BaseException as error:
            self._anchor_safe = False
            self._record("observe", "waitid_wnowait", error)
            return None
        if result is None:
            return None
        try:
            returncode = _decode_waitid(result)
        except BaseException as error:
            self._anchor_safe = False
            self._record("observe", "decode_waitid", error)
            return None
        if int(result.si_pid) != self.pid:
            self._anchor_safe = False
            self._record(
                "observe",
                "waitid_identity",
                f"waitid returned pid {result.si_pid}, expected {self.pid}",
            )
            return None
        self._observed_returncode = returncode
        self._observation_complete = True
        return returncode

    def _report(self, *, cgroup_empty: bool) -> ContainmentReport:
        return ContainmentReport(
            pid=self.pid,
            observed_returncode=self._observed_returncode,
            reaped_returncode=self._reaped_returncode,
            cgroup_path=str(self._scope.path),
            cgroup_kill_writes=self._scope.kill_writes,
            process_group_term_sent=self._process_group_term_sent,
            cgroup_empty=cgroup_empty,
            leader_reaped=self._leader_reaped,
            terminal=self._terminal,
            issues=tuple(self._issues),
        )

    def _verify_numeric_anchor(self) -> bool:
        if not self._anchor_safe or self._leader_reaped:
            return False
        # Re-prove that this exact direct child is still waitable before every
        # numeric group operation.  WNOWAIT keeps the PID/PGID anchor pinned.
        self.observe_returncode()
        if not self._anchor_safe:
            return False
        try:
            if (
                _read_pid_start_time_ticks(self.pid, self._proc_root)
                != self.start_time_ticks
            ):
                raise RuntimeError("wrapped leader start-time identity changed")
            if os.getpgid(self.pid) != self.pgid or os.getsid(self.pid) != self.sid:
                raise RuntimeError("wrapped leader process-group identity changed")
            if self.pgid == os.getpgrp():
                raise RuntimeError("refusing to signal the supervisor process group")
        except BaseException as error:
            self._anchor_safe = False
            self._record("contain", "verify_numeric_anchor", error)
            return False
        return True

    def _send_group_signal(self, signum: int, operation: str) -> None:
        if not self._verify_numeric_anchor():
            return
        try:
            os.killpg(self.pgid, signum)
            if signum == signal.SIGTERM:
                self._process_group_term_sent = True
        except ProcessLookupError:
            if signum == signal.SIGTERM:
                self._process_group_term_sent = True
        except BaseException as error:
            self._record("contain", operation, error)

    def _send_pidfd_kill(self) -> None:
        if self._pidfd < 0 or self._leader_reaped:
            return
        try:
            signal.pidfd_send_signal(self._pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            self._record("contain", "pidfd_kill", error)

    def _close_pidfd(self) -> None:
        if self._pidfd < 0:
            return
        pidfd = self._pidfd
        self._pidfd = -1
        try:
            os.close(pidfd)
        except BaseException as error:
            self._record("contain", "close_pidfd", error)

    def _wait_empty(self, deadline_ns: int) -> bool:
        while True:
            try:
                if not self._scope.populated():
                    return True
            except BaseException as error:
                self._record("contain", "read_cgroup_events", error)
                return False
            if time.monotonic_ns() >= deadline_ns:
                self._record("contain", "wait_cgroup_empty", "deadline expired")
                return False
            time.sleep(0.001)

    def contain(
        self,
        timeout_seconds: float,
        *,
        graceful_term_seconds: float = 0.0,
    ) -> ContainmentReport:
        """Hard-contain the scope, then reap once.

        ``graceful_term_seconds`` is explicit opt-in.  The default writes
        ``cgroup.kill`` immediately; ``timeout_seconds`` only bounds proof that
        the scope became empty and the final reap/cleanup.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if graceful_term_seconds < 0:
            raise ValueError("graceful_term_seconds must be nonnegative")
        if self._terminal:
            return self._report(cgroup_empty=True)

        # Once the leader has been reaped its numeric PID/PGID may be reused.
        # A cleanup retry is therefore a scope-resource operation only.
        if self._leader_reaped:
            self._close_pidfd()
            self._issues.extend(self._scope.cleanup())
            self._terminal = (
                self._scope.removed
                and self._scope.resources_closed
                and self._pidfd < 0
            )
            return self._report(cgroup_empty=self._cgroup_empty_proven)

        deadline_ns = time.monotonic_ns() + int(timeout_seconds * 1_000_000_000)
        self.observe_returncode()

        if graceful_term_seconds > 0 and self._anchor_safe:
            self._send_group_signal(signal.SIGTERM, "killpg_term")
            graceful_deadline = min(
                deadline_ns,
                time.monotonic_ns()
                + int(graceful_term_seconds * 1_000_000_000),
            )
            if not self._wait_empty(graceful_deadline):
                # A grace timeout is expected to fall through to hard kill;
                # retain only non-timeout read failures as permanent issues.
                if (
                    self._issues
                    and self._issues[-1].operation == "wait_cgroup_empty"
                ):
                    self._issues.pop()

        cgroup_empty = False
        cgroup_kill_failed = False
        try:
            # cgroup.kill is safe and idempotent even for an empty scope.  It
            # must not be gated on a fallible cgroup.events read.
            self._scope.hard_kill()
        except BaseException as error:
            cgroup_kill_failed = True
            self._record("contain", "cgroup_kill", error)
            # Retained pidfd action is identity-safe even if waitid or /proc
            # identity checks failed.  Numeric PGID fallback is separately
            # gated on a fresh WNOWAIT/start-time/PGID/SID proof.
            self._send_pidfd_kill()
            self._send_group_signal(signal.SIGKILL, "killpg_kill_fallback")
        cgroup_empty = self._wait_empty(deadline_ns)
        if not cgroup_empty:
            if not cgroup_kill_failed:
                self._send_pidfd_kill()
            return self._report(cgroup_empty=False)
        self._cgroup_empty_proven = True

        # A zombie preserves /proc/PID/cgroup until reap.  Any drift means the
        # private scope cannot prove it contained an intentionally migrated
        # leader, even though killing the retained scope itself was safe.
        if self._anchor_safe:
            try:
                observed_cgroup = _read_pid_cgroup(self.pid, self._proc_root)
                if observed_cgroup != self._scope.membership:
                    self._record(
                        "contain",
                        "leader_cgroup_drift",
                        f"{observed_cgroup!r} != {self._scope.membership!r}",
                    )
            except FileNotFoundError as error:
                self._anchor_safe = False
                self._record("contain", "leader_identity_before_reap", error)
            except BaseException as error:
                self._record("contain", "leader_identity_before_reap", error)

        if not self._observation_complete:
            self.observe_returncode()
        if not self._observation_complete and not self._wait_ownership_lost:
            # The private cgroup is empty but the direct leader is still live:
            # it migrated out.  Kill through its retained pidfd, never by PID.
            self._send_pidfd_kill()
            while not self._observation_complete:
                self.observe_returncode()
                if self._observation_complete or self._wait_ownership_lost:
                    break
                if time.monotonic_ns() >= deadline_ns:
                    self._record("contain", "wait_leader_exit", "deadline expired")
                    return self._report(cgroup_empty=True)
                time.sleep(0.001)
        if not self._reap_attempted and not self._wait_ownership_lost:
            self._reap_attempted = True
            try:
                remaining = max(
                    (deadline_ns - time.monotonic_ns()) / 1_000_000_000,
                    0.001,
                )
                self._reaped_returncode = self._process.wait(timeout=remaining)
                self._leader_reaped = True
                if (
                    self._observed_returncode is not None
                    and self._reaped_returncode != self._observed_returncode
                ):
                    self._record(
                        "contain",
                        "reap_returncode_mismatch",
                        f"{self._reaped_returncode} != {self._observed_returncode}",
                    )
            except BaseException as error:
                self._anchor_safe = False
                self._record("contain", "reap_once", error)

        # ECHILD/identity loss permanently disables raw PID/PGID action.  The
        # retained cgroup can still be proven empty and cleaned, but the report
        # remains non-ok because the exact leader reap was not ours.
        if not self._leader_reaped:
            return self._report(cgroup_empty=True)
        self._close_pidfd()
        self._issues.extend(self._scope.cleanup())
        self._terminal = (
            self._scope.removed
            and self._scope.resources_closed
            and self._leader_reaped
            and self._pidfd < 0
        )
        return self._report(cgroup_empty=True)


class _FailedLaunchSupervisor:
    """Retryable ownership for a child/scope whose launch attestation failed."""

    def __init__(
        self,
        *,
        process: subprocess.Popen[Any] | None,
        pidfd: int,
        scope: _PrivateCgroup,
    ) -> None:
        self._process = process
        self._pidfd = pidfd
        self._scope = scope
        self.pid = -1 if process is None else process.pid
        # A Popen exception is delivered only after its failed fork/exec child
        # has been reaped internally.  The private scope can still require a
        # recursive kill/empty proof (for example after post-migration failure).
        self._leader_reaped = process is None
        self._reaped_returncode: int | None = None
        self._cgroup_empty_proven = False
        self._terminal = False
        self._issues: list[SupervisionIssue] = []

    @property
    def terminal(self) -> bool:
        return self._terminal

    def _record(self, operation: str, error: BaseException | str) -> None:
        self._issues.append(_issue("launch_emergency", operation, error))

    def _report(self) -> ContainmentReport:
        return ContainmentReport(
            pid=self.pid,
            observed_returncode=None,
            reaped_returncode=self._reaped_returncode,
            cgroup_path=str(self._scope.path),
            cgroup_kill_writes=self._scope.kill_writes,
            process_group_term_sent=False,
            cgroup_empty=self._cgroup_empty_proven,
            leader_reaped=self._leader_reaped,
            terminal=self._terminal,
            issues=tuple(self._issues),
        )

    def _kill_direct_child(self) -> None:
        if self._leader_reaped or self._process is None:
            return
        try:
            if self._pidfd >= 0:
                signal.pidfd_send_signal(self._pidfd, signal.SIGKILL)
            else:
                # The direct Popen child has not been polled or reaped, so this
                # numeric PID is still pinned and cannot identify another task.
                os.kill(self._process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as error:
            self._record("kill_direct_child", error)

    def _finish_resources(self) -> None:
        if self._leader_reaped and self._pidfd >= 0:
            pidfd = self._pidfd
            self._pidfd = -1
            try:
                os.close(pidfd)
            except BaseException as error:
                self._record("close_pidfd", error)
        self._issues.extend(self._scope.cleanup())
        self._terminal = (
            self._leader_reaped
            and self._scope.removed
            and self._scope.resources_closed
            and self._pidfd < 0
        )

    def contain(self, timeout_seconds: float) -> ContainmentReport:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self._terminal:
            return self._report()
        if self._leader_reaped and self._cgroup_empty_proven:
            self._finish_resources()
            return self._report()

        deadline_ns = time.monotonic_ns() + int(timeout_seconds * 1_000_000_000)
        if not self._cgroup_empty_proven:
            try:
                self._scope.hard_kill()
            except BaseException as error:
                self._record("cgroup_kill", error)
            self._kill_direct_child()
            while True:
                try:
                    if not self._scope.populated():
                        self._cgroup_empty_proven = True
                        break
                except BaseException as error:
                    self._record("read_cgroup_events", error)
                    return self._report()
                if time.monotonic_ns() >= deadline_ns:
                    self._record("wait_cgroup_empty", "deadline expired")
                    return self._report()
                time.sleep(0.001)

        if not self._leader_reaped and self._process is not None:
            try:
                remaining = max(
                    (deadline_ns - time.monotonic_ns()) / 1_000_000_000,
                    0.001,
                )
                self._reaped_returncode = self._process.wait(timeout=remaining)
                self._leader_reaped = True
            except BaseException as error:
                self._record("reap", error)
                return self._report()
        self._finish_resources()
        return self._report()


class ChildProcessRuntime:
    """Runtime guard shared by the watcher and lower-cadence profiler."""

    def __init__(
        self,
        *,
        cgroup_mount: Path = _CGROUP_MOUNT,
        self_cgroup: Path = _SELF_CGROUP,
        proc_root: Path = _PROC_ROOT,
        cgroup_prefix: str = "skyrl-supervised",
    ) -> None:
        self._cgroup_mount = cgroup_mount
        self._self_cgroup = self_cgroup
        self._proc_root = proc_root
        self._cgroup_prefix = cgroup_prefix
        self._started = False
        self._saved_sigchld: Any = None
        self._supervisors: list[
            WrappedProcessSupervisor | _FailedLaunchSupervisor
        ] = []
        self._restore_issues: list[SupervisionIssue] = []

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> ChildProcessRuntime:
        if self._started:
            return self
        _require_linux_primitives()
        _require_single_thread()
        current = signal.getsignal(signal.SIGCHLD)
        if current is not signal.SIG_DFL:
            raise ProcessSupervisionError(
                "SIGCHLD must have the exact default disposition"
            )
        self._probe_sigchld_waitability()
        self._saved_sigchld = current
        self._started = True
        return self

    def _probe_sigchld_waitability(self) -> None:
        read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
        child_pid = -1
        pidfd = -1
        probe_reaped = False
        try:
            child_pid = os.fork()
            if child_pid == 0:
                try:
                    os.close(write_fd)
                    os.read(read_fd, 1)
                finally:
                    os._exit(_PROBE_EXIT_STATUS)
            closing_read_fd = read_fd
            read_fd = -1
            os.close(closing_read_fd)
            pidfd = os.pidfd_open(child_pid, 0)
            if os.write(write_fd, b"1") != 1:
                raise OSError("short write to SIGCHLD probe gate")
            closing_write_fd = write_fd
            write_fd = -1
            os.close(closing_write_fd)
            readiness = select.poll()
            readiness.register(pidfd, select.POLLIN)
            if not readiness.poll(2_000):
                raise TimeoutError("SIGCHLD waitability probe did not exit")
            result = os.waitid(
                os.P_PIDFD,
                pidfd,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
            if (
                result is None
                or int(result.si_pid) != child_pid
                or result.si_code != os.CLD_EXITED
                or int(result.si_status) != _PROBE_EXIT_STATUS
            ):
                raise RuntimeError("SIGCHLD waitability probe returned wrong identity")
            reaped = os.waitid(
                os.P_PIDFD, pidfd, os.WEXITED | os.WNOHANG
            )
            if (
                reaped is None
                or int(reaped.si_pid) != child_pid
                or reaped.si_code != os.CLD_EXITED
                or int(reaped.si_status) != _PROBE_EXIT_STATUS
            ):
                raise RuntimeError("SIGCHLD waitability probe reap mismatch")
            probe_reaped = True
            child_pid = -1
        except BaseException as error:
            cleanup_issues: list[SupervisionIssue] = []
            # Without a pidfd, the raw-fork child is identity-safe only while
            # it is definitely blocked behind our still-open gate.  Kill it
            # before release; hidden SA_NOCLDWAIT could otherwise auto-reap it
            # and permit numeric PID reuse before a raw signal.
            if child_pid > 0 and not probe_reaped and pidfd < 0:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except BaseException as signal_error:
                    cleanup_issues.append(
                        _issue("runtime", "kill_blocked_probe_child", signal_error)
                    )
            # Release the gate before any wait.  When a pidfd exists, all
            # post-release signaling remains pidfd-only.
            if write_fd >= 0:
                closing_write_fd = write_fd
                write_fd = -1
                try:
                    os.close(closing_write_fd)
                except BaseException as close_error:
                    cleanup_issues.append(
                        _issue("runtime", "close_probe_gate", close_error)
                    )
            if child_pid > 0 and not probe_reaped:
                if pidfd >= 0:
                    try:
                        signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except BaseException as signal_error:
                        cleanup_issues.append(
                            _issue(
                                "runtime",
                                "kill_probe_child_pidfd",
                                signal_error,
                            )
                        )
                reap_deadline_ns = time.monotonic_ns() + 2_000_000_000
                while True:
                    try:
                        waited_pid, _status = os.waitpid(child_pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    except BaseException as wait_error:
                        cleanup_issues.append(
                            _issue("runtime", "reap_probe_child", wait_error)
                        )
                        break
                    if waited_pid == child_pid:
                        break
                    if time.monotonic_ns() >= reap_deadline_ns:
                        cleanup_issues.append(
                            _issue(
                                "runtime",
                                "reap_probe_child",
                                "bounded cleanup deadline expired",
                            )
                        )
                        break
                    time.sleep(0.001)
            raise ProcessSupervisionError(
                "effective SIGCHLD disposition is not reliably waitable",
                issues=(
                    _issue("runtime", "probe_sigchld_waitability", error),
                    *cleanup_issues,
                ),
            ) from error
        finally:
            for fd in (read_fd, write_fd, pidfd):
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def launch(
        self,
        command: Sequence[str],
        *,
        pass_fds: Sequence[int] = (),
        env: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        pre_spawn_check: Callable[[], None] | None = None,
    ) -> WrappedProcessSupervisor:
        if not self._started:
            raise ProcessSupervisionError("runtime must be started before launch")
        _require_single_thread()
        if signal.getsignal(signal.SIGCHLD) is not signal.SIG_DFL:
            raise ProcessSupervisionError(
                "SIGCHLD changed after runtime start; refusing launch"
            )
        self._probe_sigchld_waitability()
        if not command:
            raise ValueError("command must not be empty")
        launch_command = list(command)
        launch_env = None if env is None else dict(env)
        validated_pass_fds = _validated_pass_fds(pass_fds)
        scope = _PrivateCgroup.create(
            cgroup_mount=self._cgroup_mount,
            self_cgroup=self._self_cgroup,
            prefix=self._cgroup_prefix,
            proc_root=self._proc_root,
        )
        process: subprocess.Popen[Any] | None = None
        pidfd = -1
        try:
            inherited_fds = (*validated_pass_fds, scope.procs_fd)
            if pre_spawn_check is not None:
                pre_spawn_check()
            process = subprocess.Popen(
                launch_command,
                start_new_session=True,
                close_fds=True,
                pass_fds=inherited_fds,
                preexec_fn=lambda: _minimal_move_to_cgroup(scope.procs_fd),
                env=launch_env,
                cwd=cwd,
            )
            pidfd = os.pidfd_open(process.pid, 0)
            supervisor = WrappedProcessSupervisor(
                process=process,
                pidfd=pidfd,
                scope=scope,
                proc_root=self._proc_root,
            )
            supervisor.verify_launch_identity()
        except BaseException as error:
            issues = [_issue("launch", "create_wrapped_process", error)]
            failed = _FailedLaunchSupervisor(
                process=process,
                pidfd=pidfd,
                scope=scope,
            )
            containment = failed.contain(1.0)
            if not failed.terminal:
                self._supervisors.append(failed)
            raise ProcessSupervisionError(
                "wrapped process launch attestation failed",
                issues=issues,
                containment=containment,
                supervisor=failed,
            ) from error
        self._supervisors.append(supervisor)
        return supervisor

    def restore(self, *, timeout_seconds: float = 1.0) -> RuntimeReport:
        reports: list[ContainmentReport] = []
        for supervisor in self._supervisors:
            reports.append(supervisor.contain(timeout_seconds))
        restored = not self._started
        if self._started:
            try:
                if signal.getsignal(signal.SIGCHLD) is not self._saved_sigchld:
                    raise RuntimeError("SIGCHLD disposition changed during supervision")
                restored = True
                self._started = False
            except BaseException as error:
                issue = _issue("runtime", "verify_sigchld_unchanged", error)
                self._restore_issues.append(issue)
        return RuntimeReport(
            containment=tuple(reports),
            sigchld_restored=restored,
            issues=tuple(self._restore_issues),
        )

    def __enter__(self) -> ChildProcessRuntime:
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        report = self.restore()
        if not report.ok:
            issues = list(report.issues)
            for containment in report.containment:
                issues.extend(containment.issues)
                if not containment.ok and not containment.issues:
                    issues.append(
                        _issue(
                            "runtime",
                            "incomplete_containment",
                            f"PID {containment.pid} was not terminally contained",
                        )
                    )
            supervision_error = ProcessSupervisionError(
                "process runtime restoration failed",
                issues=issues,
            )
            if exc is not None:
                raise BaseExceptionGroup(
                    "wrapped operation and process supervision both failed",
                    [exc, supervision_error],
                )
            raise supervision_error
        return False
