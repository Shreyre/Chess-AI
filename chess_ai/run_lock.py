"""Run ownership, with an OS lock released even when a trainer crashes."""

import os
from pathlib import Path
import sys


def training_pid(directory):
    """Return a live owner, None if stopped, or -1 if ownership is uncertain."""
    try:
        pid = int((Path(directory) / ".training.lock").read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return -1  # Never take over an unreadable or partially written lock.
    if not 0 < pid <= 0xFFFFFFFF:
        return -1
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL
        handle = api.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE; no process mutation.
        if not handle:
            return None if ctypes.get_last_error() == 87 else -1  # ERROR_INVALID_PARAMETER
        try:
            return None if api.WaitForSingleObject(handle, 0) == 0 else pid
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except OSError:
        return -1
    return pid


def acquire_training_lock(directory):
    # Keep this file: unlinking an OS lock can let two owners lock different files.
    guard = (Path(directory) / ".training.guard").open("a+b")
    try:
        guard.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(guard.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if training_pid(directory) is not None:
            raise BlockingIOError("Training already owns this run")
        lock = Path(directory) / ".training.lock"
        lock.unlink(missing_ok=True)
        with lock.open("x") as stream:
            stream.write(str(os.getpid()))
        return guard
    except BaseException:
        guard.close()
        raise
