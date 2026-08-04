# Release Notes — v0.10.1

> Released: 2026-08-04

A documentation-integrity release. v0.10.0 shipped the holographic output paths but left a README that had outgrown itself and drifted from the repository it describes: 429 lines, half the modules missing from the Algorithms table, four result links pointing at files git had never tracked, and a flagship CIFAR-100 comparison measured against a network that failed to train. Everything a reader meets first is now half the length and checked against the artifact it describes. No public API, CLI, or on-disk format changed.

## What changed

**The README is half its former size, with nothing deleted.** It went from 429 to 229 lines by relocating detail rather than dropping it: the zero-parameter-classifier and parameter-efficiency tables and both provenance notes moved to the new `docs/RESULTS.md`, each left behind as a one-line claim with a link; Quick Start, Installation, and Usage merged into one Getting Started; the Algorithms table became "The Stack", one row per layer naming all 17 modules inline — it previously listed 8 of them, omitting `Turtle3D`, `Vector3D`, `discover_dimensionality`, `UniversalEmbedder`, `GeodesicEncoder`, the Keras `ManifoldAdam` optimizer (now explicitly distinguished from `ManifoldAdamWalker`), the three `backbone_*` modules, `KnowledgeGraph`, and the two new rendering modules.

**The CIFAR-100 headline now cites a baseline that converged.** The parameter-efficiency row quoted 5.21% at 3.7M parameters — the standard arm of a 30-epoch run in which all three trials diverged, with test losses of 23,371 / 30,574 / 46,555 against roughly 2.5 for the manifold arms. Keeping the comparison same-run had the side effect of measuring the manifold against a network that never trained. The row now cites the converged 21.31% from the 100-epoch results, which puts the accuracy advantage at 1.8× rather than 7×; the 5.8× parameter reduction is unchanged. The divergence is recorded as a finding in its own right — the 3.7M-parameter dense net is unstable at the 30-epoch setting — rather than as the point being made. Alongside it, the intrinsic-dimension and noise figures were reconciled with the results JSONs, and the dimension-probe claim was corrected to match Table 8 of the paper: the geometry and class-separation components *sum* to d\*, they do not sit beside it.

**Benchmark reports are tracked and current.** `benchmarks/canonical_tests/*.pdf` was gitignored, so the CIFAR result rows linked to PDFs that could never resolve on github.com. The 11 typeset `*_report.pdf` files are now committed through a narrow gitignore negation that still excludes ad-hoc build output, CIFAR-10 and CIFAR-100 gained the Markdown reports every other dataset already had, and both CIFAR reports were regenerated after drifting from an April run.

**`pytest` works on headless machines again.** A rendering-capability probe ran at module import time in two test files, and VTK aborts with SIGSEGV rather than raising when no OpenGL implementation is reachable — so a `try/except` could not contain it and a bare `pytest` died during collection, exit 139, before any test ran. The probe now executes in a subprocess, turning the crash into an inspectable exit code: 281 passed / 9 skipped without a display, all 290 under `xvfb-run -a pytest`.

**The private `agent-kg` dependency is gone.** The optional `kgdeps` group pointed at a private git repository, so any resolution touching it — including a plain `poetry lock` — failed for anyone without access. Nothing in `waverider` imports `agent_kg`.

## Upgrading

Nothing to do. If you previously installed with `--with kgdeps`, that group no longer exists — drop the flag; plain `poetry install` was never affected by it. Readers who miss the detail that left the README will find it in `docs/RESULTS.md` and `docs/INDEX.md`.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
