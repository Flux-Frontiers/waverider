# Release Notes -- v0.16.3

> Released: 2026-11-02

A security release. WaveRider renders its holograms through quiltwright, and
a fresh install could resolve an older quiltwright and, through it, Pillow
12.3.0 and its 41 advisories, some of them critical. WaveRider now floors
quiltwright at 0.17.0, the first release that floors Pillow at 13.0.0. The
lock is refreshed past every open Dependabot alert. The geometry, the
benchmarks and the visualizations do not change.

## What changed

**quiltwright 0.17.0 is the floor.** The core dependency moves from
`quiltwright>=0.16.0` to `>=0.17.0`, which carries `pillow>=13.0.0` in its
own metadata, so every install gets the patched Pillow.

**The lock is refreshed.** Three refreshes since 0.16.2 move Pillow to
13.0.0, Streamlit to 1.66.1, tornado to 6.6, and certifi, filelock, msgpack,
nbconvert, setuptools, aiohttp, anyio, fsspec, gitpython, keras, multidict
and virtualenv to patched releases.

## Upgrading

`pip install -U waverider`. No API changes.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
