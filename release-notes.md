# Release Notes — v0.14.0

> Released: 2026-08-10

Real brains on holographic hardware. `waverider-voxel-viz` gains a `--tvb-demo` mode that renders cortical surfaces and structural connectomes from [The Virtual Brain](https://www.thevirtualbrain.org/) to Looking Glass quilts and HLD video, sitting alongside the biomedical `--ct-demo` presets that were already there. Everything else in this release is the dependency floor that makes it work.

## What changed

**Eight brain scenes, one flag.** `--tvb-demo` takes a `--tvb-dataset` of `cortex`, `cortex_80k`, `cortex_hires`, `connectome`, `connectome_998`, `head_layers`, `macaque` or `macaque_connectome`. Surface density and connectome thresholding are tunable through `--tvb-decimate`, `--tvb-smooth` and `--tvb-percentile`, and `--tvb-clear-cache` drops the downloaded data when you are done with it. The shared output flags — `--hld`, `--quilt`, `--still`, `--cast` — behave exactly as they do for `--ct-demo`, so an existing quilt or HLD workflow transfers over without change.

**The data lives in quiltwright, not here.** Loading is handled by `quiltwright.tvb_data`, new in quiltwright 0.3.0. Brain geometry is a scene source in the same sense as a POV-Ray scene or a PyVista example dataset — it is something to put on a display, not manifold science — and quiltwright already owns that job. The ~337 MB `tvb-data` archive ([doi:10.5281/zenodo.10128131](https://doi.org/10.5281/zenodo.10128131), GPL-3.0) is fetched on demand and cached there. Nothing is vendored into either package, and the GPL-3.0 corpus stays outside both distributions.

WaveRider re-exports the renderers that matter for this mode: `render_tvb_viewer`, `render_tvb_quilt`, `render_tvb_hld`, `render_tvb_still` and `TVB_PRESETS`. Full reference in [docs/waverider/tvb_data.md](docs/waverider/tvb_data.md).

**Citation metadata now points at the concept DOI.** `CITATION.cff` and the README declared `10.5281/zenodo.20383651` — not a mistyped concept DOI but a different kind of identifier entirely, the frozen version-specific archive for v0.8.0. Every citation of WaveRider had been resolving to that snapshot, and would have kept doing so through every future release. All four declarations (both badges, APA, BibTeX) now carry the concept DOI, `10.5281/zenodo.20383650`, which always resolves to the newest archived version.

## Upgrading

Nothing to migrate. The minimum `quiltwright` is now 0.3.0 and resolves automatically as a WaveRider dependency — `poetry update quiltwright` or a fresh install is all it takes.

The brain datasets are not bundled. The first `--tvb-demo` run downloads and caches them, so budget the bandwidth and disk on that first invocation:

```bash
waverider-voxel-viz --tvb-demo                                     # interactive
waverider-voxel-viz --tvb-demo --tvb-dataset cortex --hld --out cortex
waverider-voxel-viz --tvb-demo --tvb-dataset connectome \
    --quilt portrait --out connectome --cast
```

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
