[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: Elastic-2.0](https://img.shields.io/badge/License-Elastic%202.0-blue.svg)](https://www.elastic.co/licensing/elastic-license)
[![Version](https://img.shields.io/badge/version-0.12.0-blue.svg)](https://github.com/Flux-Frontiers/waverider/releases)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20383651.svg)](https://doi.org/10.5281/zenodo.20383651)


# WaveRider

**The geometry of your data tells you the exact size of network you need. Most of what your model is computing is noise.**

*Eric G. Suchanek, PhD — Flux-Frontiers*

[Technical Paper (PDF)](papers/waverider_article/waverider_jmlr.pdf)

---

## 📡 Breaking News — WaveRider renders to holographic displays

**As of v0.10.0, any WaveRider scene can be pushed to real
[Looking Glass](https://lookingglassfactory.com/) holographic hardware.**
Two device families are supported, and they take different media — render
for the display you own:

- **[Light-field quilts](https://lfdocs.lookingglassfactory.com/keyconcepts/quilts)**
  — `waverider.lfd`, 9 device presets, stills, MP4, and live casting via
  [Looking Glass Bridge](https://lookingglassfactory.com/software/looking-glass-bridge)
  → **[docs/waverider/lfd.md](docs/waverider/lfd.md)**
- **[Hololuminescent video](https://hlddocs.lookingglassfactory.com/)** —
  `waverider.hld`, 4K turntable masters to the official spec →
  **[docs/waverider/hld.md](docs/waverider/hld.md)**
- **CT / MRI demo mode** — real biomedical volumes, no model fitting
  → **[docs/waverider/voxel_viz.md](docs/waverider/voxel_viz.md)**

```bash
waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast   # light-field, live cast
waverider-voxel-viz --dataset iris --hld --out iris                     # HLD video
waverider-voxel-viz --ct-demo --ct-dataset brain --hld --out brain_hld  # MRI brain → HLD
```

Needs the viz extras (`poetry install --with viz`); `--quilt` and `--hld` are
mutually exclusive.

---

## The Core Finding

Machine learning spaces are **99% noise** by dimension. CIFAR-10 images live in a 34-dimensional manifold inside a 3,072-dimensional ambient space. Tiny ImageNet: 20 intrinsic dimensions inside 12,288. Standard algorithms treat every dimension equally — spending 99%+ of their compute on dimensions that carry no signal, while momentum, distance metrics, and gradient updates are polluted by that noise.

WaveRider measures the actual geometry, builds models constrained to the signal manifold, and derives a closed-form formula for optimal network width from first principles:

> **w\* = d\* + C − 1**

Measure the intrinsic dimensionality d\*. Count the classes C. That's your optimal bottleneck width. No grid search. No hyperparameter sweep.

---

## Headline Results

### Universal Bottleneck — formula-derived architectures beat ResNet

| Dataset | d\* | C | w\* = d\*+C−1 | ManifoldResNet-UB+Drop | Accuracy | vs ResNet-32 | Δ |
|---------|-----|---|--------------|------------------------|----------|-------------|---|
| [**CIFAR-10**](benchmarks/canonical_tests/cifar10_report.md) | 19 | 10 | 28 | 36,942 params | **71.83% ± 0.60%** | 47,978 params → 63.26% ± 3.09% | **+8.57 pp, 23% fewer params** |
| [**Fashion-MNIST**](benchmarks/canonical_tests/mnist_report.md) | 18 | 10 | 27 | 33,868 params | **88.38% ± 0.37%** | 47,338 params → 82.85% ± 2.60% | **+5.53 pp, 28% fewer params** |
| [**MNIST**](benchmarks/canonical_tests/mnist_report.md) | 16 | 10 | 25 | 29,110 params | **98.98% ± 0.21%** | 47,338 params → 99.27% ± 0.13% | within 0.3 pp, 38% fewer params |
| [**CIFAR-100**](benchmarks/canonical_tests/cifar100_report.md) | 19 | 100 | 118 | 644,262 params | **38.3% ± 3.8%** | 50,948 params → 37.6% ± 0.9% | +0.7 pp |

*UB+Drop = w\* filters with dropout=0.3 — dropout is the regularizer that lets the formula-derived width generalize.*

Two more results families, in **[docs/RESULTS.md](docs/RESULTS.md)** with full tables and provenance notes:

- **Zero-parameter classifiers** — `ManifoldModel` beats a trained MLP on Heart Disease (**83.82% vs 80.96%**) and stays within 1 pp on Breast Cancer and Dermatology, with **zero trained parameters**.
- **Parameter efficiency** — manifold-constrained models match or beat dense baselines with **105×–724×** fewer parameters (MNIST, CIFAR-10) and beat them outright on Tiny ImageNet and CIFAR-100.

All benchmark reports are indexed in **[docs/INDEX.md](docs/INDEX.md)**; every figure traces to a results JSON committed beside its script.

---

## The Dimension Probe

When a network is given a bottleneck of exactly w\* = d\* + C − 1 neurons, it
**spontaneously partitions** that space — with zero instruction — into a geometry
subspace plus exactly C−1 class-separation coordinates, and the two together
recover d\* precisely.

On CIFAR-10 (d\*=16, C=10, w\*=25), PCA on the w\*-dimensional bottleneck yields
k₉₀ = 7 geometry components (the on-manifold subspace, Whitney bound) and
n_extra = 9 class-separation coordinates. Both identities hold exactly:

> **k₉₀ + n_extra = 7 + 9 = 16 = d\***  and  **n_extra = 9 = C − 1**

The semantic content is interpretable: PC11 selects four-legged animals, PC9
flat/low-profile objects, PC12 wheeled vehicles. *(Paper, Table 8.)*

**Gradient descent independently discovers the theorem's decomposition.**

---

## The Stack

All 17 modules in `src/waverider`, by layer. Full per-component detail lives in
the **[stack summary](docs/waverider/waverider_stack_summary.md)**; worked code
examples in **[docs/USAGE.md](docs/USAGE.md)**.

| Layer | Modules | What it does |
|-------|---------|--------------|
| **Core geometry** | `TurtleND`, `Turtle3D`, `Vector3D`, `ManifoldWalker`, `ManifoldAdamWalker`, `ManifoldModel`, `ManifoldObserver` | Navigation primitives (N-dim position + orthonormal frame), Riemannian gradient descent with tangent-space Adam momentum, the zero-parameter classifier, and the (N+1)-dim extrinsic observer |
| **Dimensionality & embedding** | `discover_dimensionality`, `UniversalEmbedder`, `GeodesicEncoder`, `ManifoldAdam` | Local-PCA measurement of d\* (the primitive behind every benchmark), sklearn-PCA-compatible reduction to d\* coordinates, geodesic encoding, and a Keras optimizer that zeroes gradient noise dimensions (distinct from `ManifoldAdamWalker`) |
| **Domain applications** | `BackboneResidue`/`BackboneEmbedder`/`fit_backbone_manifold`, `KnowledgeGraph` | Protein backbone (φ, ψ, ω) latent-space discovery; semantic reasoning over knowledge graphs (module `graph_reasoner` — its entry point is `KnowledgeGraph`) |
| **Rendering** | `voxel_viz`, `lfd`, `hld` | Interactive 3-D voxel slicing, Looking Glass light-field quilts, and Hololuminescent 4K video — see [Visualization](#visualization--holographic-output) below |

---

## Getting Started

**Requirements:** Python 3.12

```bash
git clone https://github.com/Flux-Frontiers/waverider.git
cd waverider
poetry install                          # core
poetry install --with viz               # + PyVista visualization & holographic output
poetry install --with benchmarks        # + TensorFlow (Metal GPU on Apple Silicon)
poetry install --with viz,benchmarks    # everything
```

As a dependency: `poetry add git+https://github.com/Flux-Frontiers/waverider.git`
(or `pip install git+…`).

Complete code examples for every component: **[docs/USAGE.md](docs/USAGE.md)**.
The full documentation map is **[docs/INDEX.md](docs/INDEX.md)**; code lives in
`src/waverider/`, locked benchmarks in `benchmarks/canonical_tests/`, and papers
in `papers/`.

---

## Visualization & Holographic Output

### Looking Glass holographic displays — new in v0.10.0

**WaveRider renders any PyVista scene to real
[Looking Glass](https://lookingglassfactory.com/) holographic hardware**,
validated end-to-end on a physical Gen3 16″ panel. Both device families are
supported — they take different media, so render for the display you own:

- **`waverider.lfd` —
  [light-field quilts](https://lfdocs.lookingglassfactory.com/keyconcepts/quilts).**
  Off-axis asymmetric-frustum view sweep tiled into a quilt; 9 official device
  presets (Portrait, Go, 16″–65″), stills, MP4, and **live casting** to a
  connected display via
  [Looking Glass Bridge](https://lookingglassfactory.com/software/looking-glass-bridge).
  → [docs/waverider/lfd.md](docs/waverider/lfd.md)
- **`waverider.hld` —
  [Hololuminescent video](https://hlddocs.lookingglassfactory.com/resources/media-specs-and-encoding).**
  4K turntable masters to the official spec (3840×2160, HEVC, bt709); white
  renders invisible, so the subject floats.
  → [docs/waverider/hld.md](docs/waverider/hld.md)

```bash
waverider-voxel-viz --dataset iris --quilt portrait --out iris --cast   # light-field, live cast
waverider-voxel-viz --dataset iris --hld --out iris                     # HLD video
```

### Voxel Visualizer

`waverider-voxel-viz` makes high-dimensional manifolds visible: the
`ManifoldObserver`'s scalar fields (curvature, height, local intrinsic
dimensionality, …) are projected into a 3-D PCA subspace, voxelised, and served
as interactive orthogonal slice planes in PyVista. A CT/MRI demo mode
(`--ct-demo`) renders real biomedical volumes with no model fitting — and both
modes output straight to the holographic paths above.

![Manifold Voxel Visualizer — pipeline, scalar fields, datasets, controls](docs/waverider/manifold_voxel_viz.png)

```bash
waverider-voxel-viz --dataset iris --multi-scalar         # manifold mode, all fields
waverider-voxel-viz --ct-demo                             # T1 MRI brain, interactive
waverider-voxel-viz --ct-demo --ct-dataset brain --hld --out brain_hld  # MRI → HLD video
waverider-voxel-viz --ct-demo --ct-dataset brain --quilt portrait --out brain --cast  # MRI → light-field
```

- **Full CLI + API reference** (all datasets, scalar fields, flags): [docs/waverider/voxel_viz.md](docs/waverider/voxel_viz.md)
- **Worked examples:** [docs/USAGE.md](docs/USAGE.md#manifold-voxel-visualizer--interactive-3-d-manifold-anatomy) · **Method paper:** [papers/voxel_viz/voxel_viz.pdf](papers/voxel_viz/voxel_viz.pdf)

---

## Method

Gradient-diversity PCA finds the tangent space of the loss manifold: decompose
the covariance of mini-batch gradients, and the top-d eigenvectors span the
gradient's active subspace while the remaining P−d point into noise. Every
update is then projected onto that subspace before Adam sees it — momentum
accumulates signal, never noise, and its state lives in global R^P so nothing
is lost when the PCA basis rotates. The eigenvalue weighting is a form of
natural gradient using the data covariance as an empirical Fisher matrix
(Amari, 1998).

Full derivations, the projected-step algorithm, and the ambient-space failure
modes (noise-inflated KNN distances, noise-adapted Adam denominators) are in
the **[technical paper](papers/waverider_article/waverider_jmlr.pdf)** and the
**[ManifoldWalker spec](docs/manifold_walker_spec/manifold_walker_spec.md)**.

---

## Benchmarks

Every benchmark in `benchmarks/canonical_tests/` is a standalone script, run
directly with Python:

```bash
python benchmarks/canonical_tests/cifar10_manifold_architecture.py             # per-dataset (cifar100, mnist, tiny_imagenet, digits, iris likewise)
python benchmarks/canonical_tests/clinical/disease_manifold_architecture.py    # all clinical datasets
python benchmarks/canonical_tests/mnist_ub_phase_boundary.py                   # Universal Bottleneck phase boundary
```

Seed-locked results (seeds 42–51, 3–10 trials) are committed as JSON alongside
each script — the locked numbers cited in the papers. Each benchmark ships a
rendered report (`*_report.md` / `.tex` / `.pdf`) generated from its JSON by
`report_generator.py`. Full report index: **[docs/INDEX.md](docs/INDEX.md)**;
all results tables and provenance notes: **[docs/RESULTS.md](docs/RESULTS.md)**.

---

## References

- Bengio, Y. et al. (2013). *Representation Learning: A Review and New Perspectives.* TPAMI.
- Gur-Ari, G. et al. (2018). *Gradient Descent Happens in a Tiny Subspace.* arXiv:1812.04754.
- Ghorbani, B. et al. (2019). *An Investigation into Neural Net Optimization via Hessian Eigenvalue Density.* ICML.
- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning.* Neural Computation.
- Kingma, D. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.

---

## Citation

If you use WaveRider in your research or project, please cite it. Citation metadata is also provided machine-readably in [CITATION.cff](CITATION.cff).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20383651.svg)](https://doi.org/10.5281/zenodo.20383651)

> Suchanek, E. G. (2026). *WaveRider: Manifold-Aware Geometric Machine Learning* (Version 0.12.0) [Software]. Flux-Frontiers. https://doi.org/10.5281/zenodo.20383651

```bibtex
@software{suchanek_waverider,
  author    = {Suchanek, Eric G.},
  title     = {{WaveRider}: Manifold-Aware Geometric Machine Learning},
  version   = {0.12.0},
  year      = {2026},
  publisher = {Flux-Frontiers},
  url       = {https://github.com/Flux-Frontiers/waverider},
  doi       = {10.5281/zenodo.20383651}
}
```

---

## License

[Elastic License 2.0 (ELv2)](https://www.elastic.co/licensing/elastic-license) — see [LICENSE](LICENSE).

Free to use, modify, and distribute. May not be offered as a hosted or managed service to third parties.

---

*[Looking Glass](https://lookingglassfactory.com/) is a trademark of Looking Glass Factory, Inc. WaveRider is an independent project; its author is a customer and user of Looking Glass hardware, not affiliated with, sponsored by, or endorsed by Looking Glass Factory.*
