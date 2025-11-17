import argparse
import time
import subprocess
import psutil
import sys
import os
import signal
from typing import List, Tuple, Sequence

DEFAULT_COMPAT_PYTHON_PATH = os.environ.get(
    "FLASH_ANSR_COMPAT_PYTHON", "/home/psaegert/miniconda3/envs/flash-ansr-compat/bin/python3.13"
)
LOG_FILE = os.environ.get("FLASH_ANSR_WATCHDOG_LOG", "./watchdog.log")


def _normalise_exec_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.realpath(path)
    except OSError:
        return path


_TARGET_EXEC_REALPATH = _normalise_exec_path(DEFAULT_COMPAT_PYTHON_PATH)


def configure_target_python(path: str | None) -> None:
    global _TARGET_EXEC_REALPATH
    _TARGET_EXEC_REALPATH = _normalise_exec_path(path)


def set_log_file(path: str) -> None:
    global LOG_FILE
    LOG_FILE = path


def log(message: str) -> None:
    """Append a timestamped message to the watchdog log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} {message}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(entry + "\n")
    except Exception as exc:  # pragma: no cover - defensive logging
        sys.stderr.write(f"Failed to write log entry '{entry}': {exc}\n")


def _is_target_python_process(proc: psutil.Process) -> Tuple[bool, str]:
    try:
        info = proc.as_dict(attrs=["pid", "exe", "cmdline"])
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False, ""
    except psutil.AccessDenied:  # pragma: no cover - permission edge case
        log(f"WATCHDOG: Access denied inspecting process {proc.pid}")
        return False, ""

    exe = info.get("exe")
    cmdline = info.get("cmdline") or []
    normalized_candidates: List[str] = []

    if exe:
        try:
            normalized_candidates.append(os.path.realpath(exe))
        except OSError:
            normalized_candidates.append(exe)

    if cmdline:
        first = cmdline[0]
        if os.path.isabs(first):
            try:
                normalized_candidates.append(os.path.realpath(first))
            except OSError:
                normalized_candidates.append(first)
        else:
            normalized_candidates.append(first)

    if _TARGET_EXEC_REALPATH and _TARGET_EXEC_REALPATH in normalized_candidates:
        return True, " ".join(cmdline)

    return False, ""


def find_lingering_compat_processes() -> List[Tuple[psutil.Process, str]]:
    if not _TARGET_EXEC_REALPATH:
        return []
    matches: List[Tuple[psutil.Process, str]] = []
    for proc in psutil.process_iter(attrs=["pid", "exe", "cmdline"]):
        is_match, command = _is_target_python_process(proc)
        if is_match and proc.pid != os.getpid():
            matches.append((proc, command))
    return matches


def kill_lingering_compat_processes() -> bool:
    """Terminate lingering flash-ansr compat python processes.

    Returns True when no matching processes remain after the termination attempt,
    otherwise False.
    """
    if not _TARGET_EXEC_REALPATH:
        return True
    lingering = find_lingering_compat_processes()
    if not lingering:
        log("WATCHDOG: No lingering flash-ansr compat python processes detected.")
        return True

    log("WATCHDOG: Terminating lingering flash-ansr compat python processes...")

    for proc, command in lingering:
        try:
            proc.terminate()
            log(f"WATCHDOG: Sent SIGTERM to compat process {proc.pid} {command}")
        except psutil.NoSuchProcess:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            log(f"WATCHDOG: Failed to terminate process {proc.pid}: {exc}")
            continue

    gone, alive = psutil.wait_procs([p for p, _ in lingering], timeout=5)
    for proc in alive:
        try:
            proc.kill()
            log(f"WATCHDOG: Sent SIGKILL to compat process {proc.pid}")
        except psutil.NoSuchProcess:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            log(f"WATCHDOG: Failed to kill process {proc.pid}: {exc}")
    return not bool(find_lingering_compat_processes())


def wait_until_no_compat_processes(
    timeout_seconds: int = 60,
    max_force_attempts: int = 3,
    retry_sleep_seconds: float = 5.0,
) -> None:
    if not _TARGET_EXEC_REALPATH:
        return
    attempts = 0
    while True:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            lingering = find_lingering_compat_processes()
            if not lingering:
                return
            time.sleep(1)

        attempts += 1
        log(
            "WATCHDOG: Timed out waiting for compat python processes to exit. Forcing termination again."
        )

        if kill_lingering_compat_processes():
            log("WATCHDOG: Lingering compat python processes cleared after forced termination.")
            return

        if attempts > max_force_attempts:
            raise RuntimeError("Exceeded maximum attempts to clear lingering compat python processes.")

        log(
            f"WATCHDOG: Waiting {retry_sleep_seconds:.1f}s before re-checking for lingering compat processes."
        )
        time.sleep(retry_sleep_seconds)


def _safe_cpu_percent(proc: psutil.Process, interval: float | None) -> float:
    try:
        return proc.cpu_percent(interval=interval)
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return 0.0
    except psutil.AccessDenied:  # pragma: no cover - permission edge case
        log(f"WATCHDOG: Access denied collecting cpu percent for process {proc.pid}")
        return 0.0
    except Exception as exc:  # pragma: no cover - defensive logging
        log(f"WATCHDOG: Unexpected error collecting cpu percent for process {proc.pid}: {exc}")
        return 0.0


def _safe_thread_count(proc: psutil.Process) -> int:
    try:
        return proc.num_threads()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return 0
    except psutil.AccessDenied:  # pragma: no cover - permission edge case
        return 0
    except Exception:  # pragma: no cover - defensive logging adds noise
        return 0


def _safe_process_name(proc: psutil.Process) -> str:
    try:
        return proc.name()
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return "<terminated>"
    except psutil.AccessDenied:  # pragma: no cover - permission edge case
        return "<access-denied>"


def _gather_process_group_members(pid: int) -> tuple[list[psutil.Process], int | None]:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return [], None

    members: list[psutil.Process] = []
    for proc in psutil.process_iter(attrs=["pid"]):
        if proc.pid == pid:
            continue
        try:
            if os.getpgid(proc.pid) == pgid:
                members.append(proc)
        except ProcessLookupError:
            continue
        except psutil.Error:  # pragma: no cover - defensive
            continue
    return members, pgid


def terminate_process_group(process: subprocess.Popen, reason: str, grace_period: float = 5.0) -> int:
    if process.poll() is not None:
        return process.returncode

    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return process.wait()

    log(f"WATCHDOG: {reason} Initiating termination for process group {pgid}.")

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait()

    deadline = time.time() + grace_period
    while time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.5)

    if process.poll() is None:
        log(f"WATCHDOG: Process group {pgid} still active after {grace_period:.1f}s; sending SIGKILL.")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    return process.wait()


def monitor_and_kill_if_idle(
    process: subprocess.Popen,
    *,
    cpu_threshold: float = 1000.0,
    idle_duration_minutes: int = 5,
    poll_interval_seconds: float = 30.0,
    termination_grace_period: float = 5.0,
) -> int:
    """
    Monitors a process and its children. If the combined CPU usage is below
    a threshold for a specified duration, it terminates the process group.
    Since the program is multi-threaded, the threshold is set high to account for
    multiple threads each using some CPU. Polling cadence can be adjusted via
    ``poll_interval_seconds`` to trade off responsiveness versus overhead.
    """
    start_time = None
    idle_duration_seconds = idle_duration_minutes * 60

    while True:
        try:
            if process.poll() is not None:
                log(f"WATCHDOG: Process {process.pid} has terminated with exit code {process.returncode}.")
                return process.returncode

            main_process = psutil.Process(process.pid)
            group_members, pgid = _gather_process_group_members(process.pid)
            total_cpu_percent = _safe_cpu_percent(main_process, interval=1.0)
            total_threads = _safe_thread_count(main_process)
            for member in group_members:
                # Use non-blocking sampling after the parent's blocking sample to reduce delays.
                total_cpu_percent += _safe_cpu_percent(member, interval=None)
                total_threads += _safe_thread_count(member)

            if group_members:
                member_summary = ", ".join(
                    f"{member.pid}:{_safe_process_name(member)}" for member in group_members
                )
            else:
                member_summary = "none"

            pgid_note = f" pgid={pgid}" if pgid is not None else ""
            log(
                f"WATCHDOG: PID {process.pid}{pgid_note}, total CPU {total_cpu_percent:.2f}% across group: [{member_summary}], threads={total_threads}"
            )

            if total_cpu_percent < cpu_threshold:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > idle_duration_seconds:
                    log(
                        f"WATCHDOG: Process group {process.pid} has been idle for more than {idle_duration_minutes} minutes. Terminating."
                    )
                    return terminate_process_group(
                        process,
                        "Idle watchdog threshold reached",
                        grace_period=termination_grace_period,
                    )
            else:
                start_time = None

        except psutil.NoSuchProcess:
            log(f"WATCHDOG: Process {process.pid} no longer exists.")
            return process.returncode if process.returncode is not None else -1
        except Exception as e:
            log(f"WATCHDOG: An error occurred: {e}")
            return terminate_process_group(
                process,
                "Monitor encountered an exception",
                grace_period=termination_grace_period,
            )

        time.sleep(max(1.0, poll_interval_seconds))  # Avoid busy waiting

    return terminate_process_group(
        process,
        "Monitor loop exited unexpectedly",
        grace_period=termination_grace_period,
    )


def _select_eval_python(cli_value: str | None) -> str:
    candidates = [
        cli_value,
        os.environ.get("FLASH_ANSR_EVAL_PYTHON"),
        DEFAULT_COMPAT_PYTHON_PATH,
        sys.executable,
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.realpath(candidate)
    # Fallback to whatever python is running this script.
    return sys.executable


def _build_flash_ansr_command(
    *,
    eval_python: str,
    config: str,
    experiment: str | None,
    limit: int | None,
    output_file: str | None,
    save_every: int | None,
    no_resume: bool,
    verbose: bool,
    extra_args: Sequence[str],
) -> list[str]:
    resolved_config = os.path.abspath(config)
    command: list[str] = [eval_python, "-m", "flash_ansr", "evaluate-run", "-c", resolved_config]
    if limit is not None:
        command.extend(["-n", str(limit)])
    if output_file is not None:
        command.extend(["-o", output_file])
    if save_every is not None:
        command.extend(["--save-every", str(save_every)])
    if no_resume:
        command.append("--no-resume")
    if experiment:
        command.extend(["--experiment", experiment])
    if verbose:
        command.append("-v")
    if extra_args:
        command.extend(extra_args)
    return command


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watchdog wrapper that restarts flash_ansr evaluate-run when PySR stalls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", "--config", required=True, help="Path to evaluation run YAML.")
    parser.add_argument("--experiment", help="Experiment name inside the config (defaults to all).")
    parser.add_argument("-n", "--limit", type=int, help="Override runner limit before watchdog restarts.")
    parser.add_argument("-o", "--output-file", help="Override runner output file path.")
    parser.add_argument("--save-every", type=int, help="Override save cadence.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume even if output exists.")
    parser.add_argument("--eval-python", help="Python executable for running flash_ansr evaluate-run.")
    parser.add_argument(
        "--cleanup-python",
        help="Python executable whose lingering processes should be terminated (defaults to eval python).",
    )
    parser.add_argument("--log-file", default=LOG_FILE, help="Where to write watchdog logs.")
    parser.add_argument("--cpu-threshold", type=float, default=1000.0, help="CPU% threshold before declaring idle.")
    parser.add_argument("--idle-minutes", type=int, default=5, help="Minutes below threshold before restart.")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Seconds between CPU checks.")
    parser.add_argument("--grace-period", type=float, default=5.0, help="Seconds to wait after SIGTERM before SIGKILL.")
    parser.add_argument("--restart-delay", type=float, default=10.0, help="Seconds to sleep before restarting after failure.")
    parser.add_argument("--wait-timeout", type=int, default=60, help="Timeout while waiting for lingering processes to exit.")
    parser.add_argument(
        "--max-force-attempts",
        type=int,
        default=3,
        help="Number of forced cleanup attempts before aborting.",
    )
    parser.add_argument("--retry-sleep", type=float, default=5.0, help="Delay between forced cleanup attempts.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Pass -v through to flash_ansr.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded verbatim to flash_ansr evaluate-run (prefix with --).",
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()

    eval_python = _select_eval_python(args.eval_python)
    cleanup_python = args.cleanup_python or eval_python
    configure_target_python(cleanup_python)
    set_log_file(os.path.abspath(args.log_file))

    command = _build_flash_ansr_command(
        eval_python=eval_python,
        config=args.config,
        experiment=args.experiment,
        limit=args.limit,
        output_file=args.output_file,
        save_every=args.save_every,
        no_resume=args.no_resume,
        verbose=args.verbose,
        extra_args=args.extra_args,
    )

    log("WATCHDOG: Starting up...")
    print(f'Log file: {os.path.abspath(LOG_FILE)}')
    log(f"WATCHDOG: Using python executable {eval_python}")
    log(f"WATCHDOG: Target config {os.path.abspath(args.config)} (experiment={args.experiment or 'ALL'})")

    process = None
    try:
        while True:
            kill_lingering_compat_processes()
            try:
                wait_until_no_compat_processes(
                    timeout_seconds=args.wait_timeout,
                    max_force_attempts=args.max_force_attempts,
                    retry_sleep_seconds=args.retry_sleep,
                )
            except RuntimeError as err:
                log(f"WATCHDOG: Aborting start because lingering compat processes remain: {err}")
                break

            log(f"WATCHDOG: Starting evaluation command: {' '.join(command)}")
            process = subprocess.Popen(command, preexec_fn=os.setsid)

            return_code = monitor_and_kill_if_idle(
                process,
                cpu_threshold=args.cpu_threshold,
                idle_duration_minutes=args.idle_minutes,
                poll_interval_seconds=args.poll_seconds,
                termination_grace_period=args.grace_period,
            )

            if process and process.poll() is None:
                log("WATCHDOG: Process still running after monitor returned; forcing termination.")
                return_code = terminate_process_group(
                    process,
                    "Monitor finished but process still alive",
                    grace_period=args.grace_period,
                )

            kill_lingering_compat_processes()
            try:
                wait_until_no_compat_processes(
                    timeout_seconds=args.wait_timeout,
                    max_force_attempts=args.max_force_attempts,
                    retry_sleep_seconds=args.retry_sleep,
                )
            except RuntimeError as err:
                log(f"WATCHDOG: Aborting restart because lingering compat processes remain: {err}")
                break

            if return_code == 0:
                log("WATCHDOG: Evaluation script finished successfully.")
                break
            else:
                log(f"WATCHDOG: Evaluation script terminated with code {return_code}. Restarting...")

            time.sleep(args.restart_delay)

    except KeyboardInterrupt:
        log("WATCHDOG: Keyboard interrupt received. Shutting down.")
        if process and process.poll() is None:
            log(f"WATCHDOG: Terminating process group {os.getpgid(process.pid)}.")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)  # Give it a moment to clean up
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                # This can happen if the process group has already terminated
                log("WATCHDOG: Process group already terminated.")

        log("WATCHDOG: Cleaning up lingering compat processes...")
        kill_lingering_compat_processes()
        try:
            wait_until_no_compat_processes(
                timeout_seconds=args.wait_timeout,
                max_force_attempts=args.max_force_attempts,
                retry_sleep_seconds=args.retry_sleep,
            )
        except RuntimeError as err:
            log(f"WATCHDOG: Lingering compat processes could not be cleared during shutdown: {err}")
        log("WATCHDOG: Exiting due to keyboard interrupt.")
    except FileNotFoundError:
        log("WATCHDOG: Error: The evaluation command executable was not found.")
    except Exception as e:
        log(f"WATCHDOG: An unexpected error occurred: {e}")
    finally:
        log("WATCHDOG: Exiting.")


if __name__ == "__main__":
    main()
