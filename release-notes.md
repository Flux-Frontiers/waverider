# Release Notes — v0.11.0

> Released: 2026-08-04

CT/MRI demo mode can now render straight to a Looking Glass light-field display, closing a gap where that path only reached the Hololuminescent line. The README's Looking Glass section also gained links to the vendor and an explicit non-affiliation disclaimer.

## What changed

**CT/MRI demo mode reaches light-field displays.** `--ct-demo` previously only reached the Hololuminescent (HLD) line via `--hld`; the light-field (LFD) quilt path was manifold-mode-only. A new `render_ct_quilt()` mirrors `render_quilt_single()` for the CT isosurface scene, and `--ct-demo --quilt <device>` now renders a still or, with `--quilt-video`, a turntable MP4 — reusing the same `--quilt-grid`, `--quilt-zoom`, `--cast`, and video-timing flags manifold mode already had. `--hld` and `--quilt` remain mutually exclusive, and that guard now covers CT demo mode too.

**Looking Glass attribution.** The README's holographic-display sections now link to Looking Glass Factory's site and documentation (the quilt spec, Bridge, and HLD spec) and close with a disclaimer: WaveRider's author is a customer and user of the hardware, not affiliated with, sponsored by, or endorsed by Looking Glass Factory.

## Upgrading

Nothing to do — this is additive. Existing `--hld` usage and manifold-mode `--quilt` usage are unchanged.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
