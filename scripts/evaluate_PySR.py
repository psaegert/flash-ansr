import time
import subprocess
import psutil
import sys
import os
import signal

COMPAT_PYTHON_PATH = "/home/psaegert/miniconda3/envs/flash-ansr-compat/bin/python3.13"
LOG_FILE = "watchdog.log"


def log(message: str) -> None:
    """Append a timestamped message to the watchdog log file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} {message}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(entry + "\n")
    except Exception as exc:  # pragma: no cover - defensive logging
        sys.stderr.write(f"Failed to write log entry '{entry}': {exc}\n")


def kill_lingering_compat_processes() -> None:
    """Terminate lingering flash-ansr compat python processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", COMPAT_PYTHON_PATH],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        log("WATCHDOG: 'pgrep' command not found; cannot terminate lingering compat processes.")
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        log(f"WATCHDOG: Failed to query lingering compat processes: {exc}")
        return

    output = result.stdout.strip()
    if not output:
        log("WATCHDOG: No lingering flash-ansr compat python processes detected.")
        return

    log("WATCHDOG: Terminating lingering flash-ansr compat python processes...")

    for line in output.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue

        if pid == os.getpid():
            continue

        command = parts[1] if len(parts) > 1 else ""

        try:
            proc = psutil.Process(pid)
            proc.terminate()
            log(f"WATCHDOG: Sent SIGTERM to compat process {pid} {command}")
        except psutil.NoSuchProcess:
            continue
        except Exception as exc:  # pragma: no cover - defensive logging
            log(f"WATCHDOG: Failed to terminate process {pid}: {exc}")
            continue

        try:
            proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            try:
                proc.kill()
                log(f"WATCHDOG: Sent SIGKILL to compat process {pid}")
            except psutil.NoSuchProcess:
                pass


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
            all_processes = [main_process] + children

            total_cpu_percent = sum(p.cpu_percent(interval=1) for p in all_processes)

            log(f"WATCHDOG: PID: {process.pid}, Total CPU Usage: {total_cpu_percent:.2f}%")

            if total_cpu_percent < cpu_threshold:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > idle_duration_seconds:
                    log(f"WATCHDOG: Process group {process.pid} has been idle for more than {idle_duration_minutes} minutes. Terminating.")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(5)
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    return process.wait()
            else:
                start_time = None

        except psutil.NoSuchProcess:
            log(f"WATCHDOG: Process {process.pid} no longer exists.")
            return process.returncode if process.returncode is not None else -1
        except Exception as e:
            log(f"WATCHDOG: An error occurred: {e}")
            break

        time.sleep(30)  # Check every 30 seconds

    return -1


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
            log(f"WATCHDOG: Starting evaluation for test set: {test_set}")

            process = subprocess.Popen(
                script_to_run,
                preexec_fn=os.setsid
            )

            return_code = monitor_and_kill_if_idle(process)

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
