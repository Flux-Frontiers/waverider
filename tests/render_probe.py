"""Crash-safe probe for pyvista off-screen rendering support.

When VTK cannot reach a usable OpenGL implementation it aborts the process with
SIGSEGV instead of raising, so wrapping the probe in ``try/except Exception``
does not contain it: the segfault takes the whole interpreter down. Because the
probe runs at module import time, that happens during pytest's *collection*
phase, so a bare ``pytest`` on a machine with no GL stack dies before a single
test executes rather than skipping the rendering tests as intended.

Running the probe in a subprocess turns the crash into an exit code the parent
can inspect, which lets collection finish and the rendering tests skip cleanly.
Under a working display (for example ``xvfb-run -a pytest``) the probe succeeds
and the tests run as usual.
"""

import subprocess
import sys
from functools import lru_cache

# Executed in a child interpreter -- a crash here must not reach the parent.
_PROBE_SOURCE = """
import pyvista as pv

p = pv.Plotter(off_screen=True, window_size=(32, 32))
p.add_mesh(pv.Sphere())
p.screenshot(None, return_img=True)
p.close()
"""

# Generous enough for a cold VTK import on a slow machine, bounded so a hung
# driver cannot stall collection indefinitely.
_PROBE_TIMEOUT = 120


@lru_cache(maxsize=1)
def can_render() -> bool:
    """True if pyvista can produce an off-screen render in this environment.

    The result is cached, so the subprocess is spawned once per session no
    matter how many test modules ask.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0
