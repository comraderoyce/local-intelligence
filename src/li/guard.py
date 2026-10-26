"""Memory guard: end this process before macOS runs out of memory.

On a 16GB Mac, MLX's GPU buffers count as wired or active memory that macOS
cannot page out. When they and other apps squeeze available memory to near
zero, WindowServer stalls and the watchdog panics the kernel: the machine
reboots and everything unsaved is lost. Process RSS doesn't show MLX buffers,
so the guard watches system-wide available memory (free + inactive +
speculative pages) and exits this process when it drops below a floor.
Losing one run beats a kernel panic.

LI_MIN_FREE_GB sets the floor (default 1.2); 0 disables the guard.
"""
from __future__ import annotations

import os
import sys
import threading
import time

DEFAULT_MIN_FREE_GB = 1.2
POLL_S = 0.5


def start() -> None:
    floor = float(os.environ.get("LI_MIN_FREE_GB", DEFAULT_MIN_FREE_GB))
    if floor <= 0 or sys.platform != "darwin":
        return
    from .stream import available_gb

    def watch() -> None:
        while True:
            free = available_gb()
            if free < floor:
                sys.stderr.write(
                    f"\nli: memory guard: available memory {free:.2f}GB < {floor:.1f}GB "
                    "floor; exiting to avoid a macOS memory panic. Close other apps, "
                    "lower --budget-gb or --batch, or set LI_MIN_FREE_GB (0 disables).\n")
                sys.stderr.flush()
                os._exit(137)
            time.sleep(POLL_S)

    threading.Thread(target=watch, name="li-memory-guard", daemon=True).start()
