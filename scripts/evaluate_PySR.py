import time
import subprocess
import psutil
import sys
import os
import signal
from typing import List, Tuple

COMPAT_PYTHON_PATH = "/home/psaegert/miniconda3/envs/flash-ansr-compat/bin/python3.13"
LOG_FILE = "watchdog.log"
_TARGET_EXEC_REALPATH = os.path.realpath(COMPAT_PYTHON_PATH)


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

    if _TARGET_EXEC_REALPATH in normalized_candidates:
        return True, " ".join(cmdline)

    return False, ""


def find_lingering_compat_processes() -> List[Tuple[psutil.Process, str]]:
    matches: List[Tuple[psutil.Process, str]] = []
    for proc in psutil.process_iter(attrs=["pid", "exe", "cmdline"]):
        is_match, command = _is_target_python_process(proc)
        if is_match and proc.pid != os.getpid():
            matches.append((proc, command))
    return matches


def kill_lingering_compat_processes() -> None:
    """Terminate lingering flash-ansr compat python processes."""
    lingering = find_lingering_compat_processes()
    if not lingering:
        log("WATCHDOG: No lingering flash-ansr compat python processes detected.")
        return

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


def wait_until_no_compat_processes(timeout_seconds: int = 60) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        lingering = find_lingering_compat_processes()
        if not lingering:
            return
        time.sleep(1)
    log(
        "WATCHDOG: Timed out waiting for compat python processes to exit. Forcing termination again."
    )
    kill_lingering_compat_processes()
    if find_lingering_compat_processes():
        log("WATCHDOG: Warning: compat python processes still detected after forced termination.")


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


def monitor_and_kill_if_idle(process: subprocess.Popen, cpu_threshold: float = 10.0, idle_duration_minutes: int = 5) -> int:
    """
    Monitors a process and its children. If the combined CPU usage is below
    a threshold for a specified duration, it terminates the process group.
    """
    start_time = None
    idle_duration_seconds = idle_duration_minutes * 60

    while True:
        try:
            if process.poll() is not None:
                log(f"WATCHDOG: Process {process.pid} has terminated with exit code {process.returncode}.")
                return process.returncode

            main_process = psutil.Process(process.pid)
            children = main_process.children(recursive=True)
            total_cpu_percent = _safe_cpu_percent(main_process, interval=1.0)
            for child in children:
                # Use non-blocking sampling after the parent's blocking sample to reduce delays.
                total_cpu_percent += _safe_cpu_percent(child, interval=None)

            log(f"WATCHDOG: PID: {process.pid}, Total CPU Usage: {total_cpu_percent:.2f}%")

            if total_cpu_percent < cpu_threshold:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > idle_duration_seconds:
                    log(
                        f"WATCHDOG: Process group {process.pid} has been idle for more than {idle_duration_minutes} minutes. Terminating."
                    )
                    return terminate_process_group(process, "Idle watchdog threshold reached")
            else:
                start_time = None

        except psutil.NoSuchProcess:
            log(f"WATCHDOG: Process {process.pid} no longer exists.")
            return process.returncode if process.returncode is not None else -1
        except Exception as e:
            log(f"WATCHDOG: An error occurred: {e}")
            return terminate_process_group(process, "Monitor encountered an exception")

        time.sleep(30)  # Check every 30 seconds

    return terminate_process_group(process, "Monitor loop exited unexpectedly")


def main() -> None:
    if len(sys.argv) < 2:
        log("WATCHDOG: Usage: python watchdog.py <TEST_SET>")
        sys.exit(1)

    test_set = sys.argv[1]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_to_run = [os.path.join(script_dir, 'evaluate_PySR_niterations.sh'), test_set]

    process = None
    try:
        while True:
            kill_lingering_compat_processes()
            wait_until_no_compat_processes()
            log(f"WATCHDOG: Starting evaluation for test set: {test_set}")

            process = subprocess.Popen(
                script_to_run,
                preexec_fn=os.setsid
            )

            return_code = monitor_and_kill_if_idle(process)

            if process and process.poll() is None:
                log("WATCHDOG: Process still running after monitor returned; forcing termination.")
                return_code = terminate_process_group(process, "Monitor finished but process still alive")

            kill_lingering_compat_processes()
            wait_until_no_compat_processes()

            if return_code == 0:
                log("WATCHDOG: Evaluation script finished successfully.")
                break
            else:
                log(f"WATCHDOG: Evaluation script terminated with code {return_code}. Restarting...")

            time.sleep(30)

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
        log("WATCHDOG: Exiting due to keyboard interrupt.")
    except FileNotFoundError:
        log(f"WATCHDOG: Error: The script '{script_to_run[0]}' was not found.")
    except Exception as e:
        log(f"WATCHDOG: An unexpected error occurred: {e}")
    finally:
        log("WATCHDOG: Exiting.")


if __name__ == "__main__":
    main()
