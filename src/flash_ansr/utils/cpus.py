"""How many CPU cores this process may use."""
import os


def available_cores() -> int:
    """The CPU cores this process may run on, at least 1.

    A SLURM allocation or ``taskset`` restricts the process to some of the machine's cores; ``os.cpu_count()``
    counts all of them, so a pool sized by it oversubscribes the cores the process was given. This reads the
    restriction: ``os.process_cpu_count()`` (Python 3.13+), else the affinity mask (Linux), else ``os.cpu_count()``.
    """
    if hasattr(os, "process_cpu_count"):          # Python 3.13+
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):        # Linux
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return max(1, n or 1)
