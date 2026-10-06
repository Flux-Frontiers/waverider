# Release Notes -- v0.16.2

> Released: 2026-10-06

A bug-fix release. `waverider-voxel-viz --hld --still` now writes a single PNG
for manifold datasets, as it already did for the CT and TVB demos, instead of
silently producing the ten-second turntable video.

## What changed

**`--still` works in manifold mode.** The flag exports one 3840x2160 PNG with
the HLD white background and needs no ffmpeg. The CT and TVB paths honored it,
but the manifold path checked only `--hld` and always rendered the video, so
the flag did nothing and gave no warning. `render_hld_single()` now takes a
`still` argument and the command line passes it through. Two new tests cover
the function and the dispatch; the full suite passes 361 tests.

**`proteusPy` floor current.** The `bench` extra now requires `proteusPy`
0.100.6, the current release. The lock also moved `urllib3` from 2.7.0 to
2.8.0, a transitive update that changes none of WaveRider's own constraints.

## Upgrading

`pip install --upgrade waverider`. Add `[bench]` to update `proteusPy` too.
If you were passing `--hld --still` on a manifold dataset and getting a video,
you will now get a PNG named `<stem>_hld.png`. No other behavior changes.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
