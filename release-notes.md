# Release Notes — v0.13.0

> Released: 2026-08-09

WaveRider sheds its holographic output layer. Everything that turned arrays into Looking Glass quilts, HLD video, or POV-Ray scenes now lives in [quiltwright](https://github.com/suchanek/quiltwright), a separate BSD-3 package that WaveRider depends on. This is a boundary correction, not a feature release — plus a Bridge hang that had to be fixed and a lock-file security sweep.

## What changed

**Holographic output is now quiltwright.** `waverider.lfd`, `waverider.hld` and `waverider.povray` moved out wholesale. The split was overdue: none of that code ever imported anything from WaveRider — `lfd.py` needed only stdlib, numpy and pillow — while three unrelated consumers had accumulated around it. Manifold visualisation, molecular rendering through proteusPy, and a POV-Ray scene archive were all reaching for the same module, and none of them should have to carry the other two as dependencies. Git history for the moved files is preserved in the quiltwright repository, so `git log --follow` still works there.

`waverider.voxel_viz` and its CLI are untouched.

**`stop_quilt()` no longer hangs Bridge.** The v0.12.0 implementation followed bridge.js's documented `stopStudioPlaylist` pattern — `delete_playlist`, then `show_window(false)`. On Looking Glass Bridge 2.6.3 (macOS), `delete_playlist` reliably hung the process: reproduced twice, once mid-video and once on a single still, each time requiring `kill -9` and a Bridge relaunch to recover. `stop_quilt()` now reaches the same end state — nothing visible, playback halted — using only `transport_control_pause` and `show_window(false)`, both independently verified safe. Confirmed against a live device: Bridge stayed responsive, the display went blank.

**Dependency security sweep.** Seventeen packages flagged by OSV.dev were bumped in `poetry.lock`, including `aiohttp`, `cryptography`, `pillow`, `torch` and `starlette`. No `pyproject.toml` bounds changed, so this is lock-only and does not constrain downstream resolution.

## Upgrading

`from waverider import render_quilt, QUILT_PRESETS, cast_quilt, ...` continues to work unchanged — `waverider/__init__` re-exports the full holographic surface from quiltwright.

Direct submodule imports must be repointed:

```python
from waverider.lfd import render_quilt   # no longer resolves
from quiltwright.lfd import render_quilt # use this
```

The same applies to `waverider.hld` and `waverider.povray`. `quiltwright>=0.1.0` installs automatically as a WaveRider dependency; nothing extra to add.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
