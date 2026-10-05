# Release Notes -- v0.16.1

> Released: 2026-10-05

A dependency-only release. WaveRider now requires the current releases of
the fleet packages it builds on, so a fresh install can no longer resolve to
the older `turtlend` and `quiltwright` that 0.16.0 still allowed.

## What changed

**Fleet floors current.** `turtlend` is floored at 0.1.1 and `quiltwright` at
0.16.0, and the `bench` extra's `proteusPy` at 0.100.5, which itself now
requires `turtlend` 0.1.1. No code in WaveRider changes and the public API is
the same; the full suite passes 359 tests against the new versions.

## Upgrading

`pip install --upgrade waverider` pulls in the newer `turtlend` and
`quiltwright`. Add `[bench]` to update `proteusPy` too. No code changes are
needed.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
