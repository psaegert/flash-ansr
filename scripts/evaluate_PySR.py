import time
import subprocess
import psutil
import sys
import os
import signal


def monitor_and_kill_if_idle(process: subprocess.Popen, cpu_threshold: float = 5.0, idle_duration_minutes: int = 5) -> None:
    """
    Monitors a process and its children. If the combined CPU usage is below
    a threshold for a specified duration, it terminates the process group.
    """
    start_time = None
    idle_duration_seconds = idle_duration_minutes * 60

    while True:
        try:
            # Check if the main process is still running
            if process.poll() is not None:
                print(f"Process {process.pid} has terminated with exit code {process.returncode}.")
                break

            # Get the main process and all its children
            main_process = psutil.Process(process.pid)
            children = main_process.children(recursive=True)
            all_processes = [main_process] + children

            # Sum up CPU usage for all related processes
            total_cpu_percent = sum(p.cpu_percent(interval=1) for p in all_processes)

            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}, PID: {process.pid}, Total CPU Usage: {total_cpu_percent:.2f}%")

            if total_cpu_percent < cpu_threshold:
                if start_time is None:
                    # Start the timer if CPU usage is low
                    start_time = time.time()
                elif time.time() - start_time > idle_duration_seconds:
                    # If idle for too long, terminate the process group
                    print(f"Process group {process.pid} has been idle for more than {idle_duration_minutes} minutes. Terminating.")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(5)  # Give it a moment to terminate
                    # If it's still alive, force kill
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    break
            else:
                # Reset the timer if CPU usage is high enough
                start_time = None

        except psutil.NoSuchProcess:
            print(f"Process {process.pid} no longer exists.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

        time.sleep(30)  # Check every 30 seconds


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python evaluate_PySR.py <TEST_SET>")
        sys.exit(1)

    test_set = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_to_run = [os.path.join(script_dir, 'evaluate_PySR_niterations.sh'), test_set]

    while True:
        print(f"Starting evaluation for test set: {test_set}")
        try:
            # Start the process in a new process group
            process = subprocess.Popen(
                script_to_run,
                text=True,
                preexec_fn=os.setsid
            )

            # Start the monitor
            monitor_and_kill_if_idle(process)

            # Check the exit code to see if we should stop
            if process.returncode == 0:
                print("Evaluation script finished successfully.")
                break
            else:
                print(f"Evaluation script terminated with code {process.returncode}. Restarting...")

        except FileNotFoundError:
            print(f"Error: The script '{script_to_run[0]}' was not found.")
            break
        except Exception as e:
            print(f"An error occurred while running the script: {e}")
            break

        # Optional: wait a bit before restarting
        time.sleep(30)


if __name__ == "__main__":
    main()
