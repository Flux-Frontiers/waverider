# Release Notes — v0.12.0

> Released: 2026-08-04

The Looking Glass Bridge integration now covers a full playback lifecycle, not just casting: pause, resume, and stop are real functions with real Bridge endpoints behind them, discovered the hard way after two plausible-sounding endpoint names turned out not to exist.

## What changed

**Quilt playback controls.** `waverider.lfd` gains `pause_quilt()`, `resume_quilt()`, and `stop_quilt()` alongside the existing `cast_quilt()`, all exported from the top-level `waverider` package. Bridge has no `stop_playlist` or `pause_playlist` endpoint — calling one silently returns `200 OK` with an empty body instead of erroring, which looks identical to a slow success until you notice the response has no `status` field. The real control group is Bridge's *transport control* API (`transport_control_play`/`_pause`, plus `delete_playlist` to remove a playlist outright), confirmed against the official [bridge.js](https://github.com/Looking-Glass/bridge.js) SDK source rather than guessed from naming conventions. All three functions were verified live against a physical Looking Glass. `docs/waverider/lfd.md` gained a "Control playback" section documenting the full endpoint reference and the trap itself, so the next person doesn't have to reverse-engineer it again.

## Upgrading

Nothing to do — this is additive. Existing `cast_quilt()` / `--cast` usage is unchanged.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
