# WaveRider — Usage Guide

See the [Documentation Index](INDEX.md) for API specs, papers, and benchmark reports.

---

## ManifoldModel — zero-parameter geometry classifier

```python
from waverider import ManifoldModel

model = ManifoldModel(k_pca=20, tau=0.85)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
# Zero learned parameters — the manifold geometry is the classifier
```

---

## ManifoldWalker — Riemannian gradient descent

```python
from waverider import ManifoldWalker

walker = ManifoldWalker(k_samples=30, tau=0.90, reorient_every=10)
walker.fit(X, y, epochs=100, lr=0.01)
```

---

## ManifoldAdamWalker — manifold-projected Adam

```python
from waverider import ManifoldAdamWalker

walker = ManifoldAdamWalker(
    k_samples=30, tau=0.90,
    beta1=0.9, beta2=0.999, lr=0.001,
    reorient_every=10
)
walker.fit(X, y, epochs=200)
```

---

## TurtleND — N-dimensional frame navigation

```python
from waverider import TurtleND

turtle = TurtleND(dim=10)
turtle.forward(0.1)          # step along heading
turtle.turn(axis=1, angle=0.3)   # rotate frame in the (0,1) plane
print(turtle.position, turtle.frame)
```

Frames are orthonormal by construction via Givens rotations. The `frame` attribute is an (N×N) rotation matrix; `position` is an N-vector.

---

## ManifoldObserver — extrinsic (N+1)-dimensional sensor

```python
from waverider import ManifoldModel, ManifoldObserver

subject = ManifoldModel(k_graph=10, k_pca=20, k_vote=7, variance_threshold=0.90)
subject.fit(X, y)

observer = ManifoldObserver(subject)
observer.lift_data()
field = observer.observe()
# Returns list of ObservationResult with fields:
#   curvature, height, intrinsic_dim, local_variance, class_label
```

The observer sits one dimension above the manifold surface and measures local curvature and intrinsic dimensionality at each data point.

---

## Manifold Voxel Visualizer — interactive 3-D manifold anatomy

Requires `poetry install --with viz`.

### Programmatic API

```python
from waverider import fit_and_observe, voxelize, build_grid, render_single

subject, observer, pf, pca_info = fit_and_observe(
    X, y, k_graph=10, k_pca=20, k_vote=7, tau=0.90
)
vox  = voxelize(pf, resolution=32)
grid = build_grid(vox)
render_single(grid, pf, scalar="density", pca_info=pca_info)
```

Available scalars: `"density"`, `"curvature"`, `"intrinsic_dim"`, `"local_variance"`, `"height"`.

### CLI

```bash
# Interactive viewer — synthetic helix (default)
waverider-voxel-viz

# Iris dataset, 2×2 panel of all scalar fields
waverider-voxel-viz --dataset iris --multi-scalar

# Headless PNG export
waverider-voxel-viz --dataset breast_cancer --off-screen --out bc_voxels.png

# CIFAR-10 — subsample 1,000 pts, pre-reduce to 40-D
waverider-voxel-viz --dataset cifar10 --n-points 1000 --pre-pca 40
```

See [waverider/manifold_voxel_viz.md](waverider/manifold_voxel_viz.md) for the full argument reference.

---

## Intrinsic Dimensionality Probe

```python
from waverider import ManifoldModel

model = ManifoldModel(k_pca=50, tau=0.90)
model.fit(X, y)

# Per-class intrinsic dimensionality
for cls, d in model.intrinsic_dims_.items():
    print(f"Class {cls}: d* = {d:.1f}")

# Global estimate
print(f"Global d*: {model.global_intrinsic_dim_:.1f}")
print(f"Noise fraction: {1 - model.global_intrinsic_dim_ / X.shape[1]:.1%}")
```

---

## ManifoldWalker with intrinsic dim reporting

```python
from waverider import ManifoldWalker

walker = ManifoldWalker(k_samples=30, tau=0.90, reorient_every=10)
walker.fit(X, y, epochs=100, lr=0.01)

print(f"Discovered intrinsic dim: {walker.intrinsic_dim_}")
print(f"Noise suppressed: {1 - walker.intrinsic_dim_ / X.shape[1]:.1%}")
```
