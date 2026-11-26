"""Prevent idle system sleep during an explicitly requested overnight run.

Only a temporary Windows thread request is used; power settings are never changed.
Delete .cache/keep-awake to end the request. Explicit sleep/lid actions still apply.
"""
import ctypes
import os
import time
from pathlib import Path

if __name__ == "__main__" and os.name == "nt":
    flag = Path(__file__).resolve().parents[1] / ".cache/keep-awake"
    flag.parent.mkdir(exist_ok=True)
    flag.write_text("Overnight project execution; delete this file to allow idle sleep.\n", encoding="utf-8")
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        deadline = time.monotonic() + 9 * 3600
        while flag.exists() and time.monotonic() < deadline:
            time.sleep(30)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
