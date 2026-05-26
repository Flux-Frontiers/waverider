# Release Notes — v0.9.0

> Released: 2026-05-26

**The headline change is `UniversalEmbedder`** — a geometry-grounded, modality-agnostic dimensionality reducer that exposes the same `fit` / `transform` / `fit_transform` surface as `sklearn.decomposition.PCA`. Any benchmark pipeline that currently calls `PCA(n_components=…)` can swap in `UniversalEmbedder()` without touching the surrounding code. Under the hood it discovers d\* from local manifold geometry (`ManifoldModel`) and then auto-selects between a global-PCA projection (for near-linear data) and BFS Procrustes-transported TurtleND frames (for genuinely curved manifolds) based on the **Manifold Linearity Index** `MLI = global_d_at_τ / d*`. Explicit modes are available when you want to force one strategy. Ships with 362 lines of unit tests covering all four modes plus the sklearn drop-in contract.

The release also lands a documentation audit triggered by an external claim-verification request: the README's CIFAR-10 "+8.5 pp over ResNet" UB headline has been re-grounded against the raw JSON trial data, the column header now honestly names the dropout variant (`ManifoldResNet-UB+Drop`), and per-row stats across CIFAR-10 / Fashion-MNIST / MNIST are re-aligned to JSON-computed sample stds. A standalone [`CIFAR10_CLAIM_VERIFICATION.md`](CIFAR10_CLAIM_VERIFICATION.md) report sits at the repo root with the full audit, including a "What 'Matched' Means" architecture table so a reviewer who clones the repo can reproduce.

Smaller items:

- New **Manifold Voxel Visualizer** section in the README under Algorithms — hero figure, CLI examples for every built-in dataset, per-voxel scalar-field inventory, and links to the CLI+API reference, USAGE examples, and method paper.
- File-tree refresh and a handful of dead `*_report.md` README links redirected to the existing `.pdf` reports.
- Minor: `pyproject.toml` section-header comment retitled `CodeKG → PyCodeKG`.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
