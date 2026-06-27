"""Timeout-guarded SymPy simplification, isolated as a dependency-light leaf utility.

This module deliberately depends only on the standard library plus a *lazily*-imported
``sympy`` so that importing it does NOT pull in the data/sampling stack. It backs the
optional ``simplify == 'sympy'`` canonicalization mode (a secondary canonicaliser on top
of the primary simplipy engine). ``sympy`` is an optional extra::

    pip install flash-ansr[sympy]
"""
import os


def _sympy_simplify_call(expr_str: str) -> str:
    """Run sympy.simplify (called in a forked child to allow timeout)."""
    from sympy import simplify as _sp_simplify, parse_expr as _sp_parse  # noqa: delayed import
    expr = _sp_parse(expr_str)
    result = _sp_simplify(expr, ratio=1)
    return str(result)


def _sympy_simplify_with_timeout(expr_str: str, timeout_seconds: float = 1.0) -> tuple[str, float] | None:
    """Return (simplified_str, elapsed) or None on timeout / error.

    Uses os.fork() so that hung SymPy calls can be killed via SIGKILL,
    preventing zombie-thread accumulation that degrades performance.

    Requires the optional ``sympy`` dependency (``pip install flash-ansr[sympy]``);
    raises a clear ``ImportError`` with that hint if it is missing.
    """
    import time
    import signal
    import select
    try:
        import sympy  # noqa: F401 – ensure imported before fork
    except ImportError as exc:  # pragma: no cover - only hit without the [sympy] extra
        raise ImportError(
            "The simplify=='sympy' mode requires the optional 'sympy' dependency. "
            "Install it with: pip install flash-ansr[sympy]"
        ) from exc

    r_fd, w_fd = os.pipe()
    start = time.time()
    pid = os.fork()

    if pid == 0:
        # ── child process ──
        os.close(r_fd)
        try:
            result = _sympy_simplify_call(expr_str)
            os.write(w_fd, result.encode('utf-8'))
        except Exception:
            pass
        finally:
            os.close(w_fd)
            os._exit(0)

    # ── parent process ──
    os.close(w_fd)

    ready, _, _ = select.select([r_fd], [], [], timeout_seconds)
    elapsed = time.time() - start

    if ready:
        chunks = []
        while True:
            chunk = os.read(r_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(r_fd)
        data = b''.join(chunks)
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        if data:
            return (data.decode('utf-8'), elapsed)
        return None
    else:
        os.close(r_fd)
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        return None
