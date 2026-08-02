# Release Notes — v0.10.0

> Released: 2026-08-02

WaveRider manifolds now render as true holograms. This release adds two output paths for Looking Glass hardware — `waverider.lfd` for the light-field line, which consumes multi-view quilts, and `waverider.hld` for the Hololuminescent line, which plays ordinary video into a fixed optical volume. Any PyVista scene can drive either. The whole path was validated end to end against a physical Gen3 16″ landscape panel, which is how three bugs that had never been exercised came to light.

## What changed

**Holographic output.** The quilt renderer sweeps a camera across the device's view cone using off-axis asymmetric-frustum projections rather than orbiting it. That distinction matters: rotating the camera ("toe-in") shears the focal plane differently in each view, and a lenticular display cannot fuse views that disagree about where the focal plane sits. Keeping it pixel-identical across all views is what makes a quilt work. Output covers still quilts, turntable quilt videos, and live casting to a connected display through Looking Glass Bridge.

**Hardware truth over documentation.** The published quilt table and the panel on the desk disagreed. The `16-landscape` preset described a 7×7 grid at 5999², where a Gen3 panel reports 8×6 at 7680×4320 with a 50° view cone rather than the 35° default — meaning roughly a third of the available parallax was being discarded. The preset now matches what Bridge reports, and the docs carry the probe command so you can confirm your own hardware instead of trusting a table that has no generation column.

**Casting used to fail silently.** Bridge's HTTP API requires `PUT`. The client sent `POST`, which Bridge answers with `200 OK` and an empty body — so the orchestration token came back empty, every subsequent call quietly did nothing, and `--cast` reported success while the display stayed dark. An empty token now raises.

**Framing as a depth budget.** Perceived depth scales with how much of each view the subject fills, so the default framing was wasting both resolution and parallax. Quilt rendering now dollies the camera in by default, which preserves the field of view the parallax geometry assumes. On the iris manifold this moved subject coverage from 35% to 85% of frame width and more than tripled the measured difference between extreme views. The colour scale bar is off by default for quilts — a 2-D overlay has no parallax, so the display pins it to the focal plane where it reads as a flat pane through the middle of the hologram.

**CT and MRI mode.** `--ct-demo` bypasses manifold fitting entirely and renders layered isosurfaces from PyVista's built-in biomedical volumes, either interactively or as a display-ready turntable.

**Project infrastructure.** GitHub Actions CI now runs lint, type-check, and tests on every push and pull request, and tagged releases build and publish automatically. Type checking moved from mypy to `ty` and pylint was dropped in favour of ruff alone, matching the pycode_kg and doc_kg repos. `pyproject.toml` migrated to PEP 621. Building the CI surfaced a latent defect: `graph_reasoner` imported `TurtleND` from `proteusPy`, an optional extra, so its test module could not be collected in a lean install.

## Upgrading

Rendering needs the viz extras — `poetry install --with viz` — plus ffmpeg for video, which `imageio-ffmpeg` now supplies. Casting to a display additionally needs Looking Glass Bridge 2.2 or newer running on the machine the panel is plugged into.

If you imported `waverider.looking_glass`, it is now `waverider.lfd`. The rename pairs it with `waverider.hld` so each module names the display technology it targets rather than the vendor, since HLDs are Looking Glass products too. There is no compatibility shim, because that module had not appeared in a tagged release.

Before your first quilt, ask the panel what it wants rather than trusting the preset table — the procedure is in `docs/waverider/lfd.md`. Two new knobs are worth knowing: `--quilt-zoom` controls framing (1.6 by default) and `--quilt-scalar-bar` restores the colour bar if you need the values more than the depth.

Nothing changes for existing manifold or classifier code.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
