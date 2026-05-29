# Protein Backbone Secondary Structure — Study Artifacts

Intermediate results and per-experiment reports for the paper *Protein Backbone
Secondary Structure Classification: A Feature Engineering Study with
Manifold-Aware Architectures* (E. G. Suchanek).

## Reproducing

The three benchmark scripts live under
[`benchmarks/canonical_tests/`](../../benchmarks/canonical_tests/):

```bash
# Intrinsic-dimensionality probe + ManifoldModel k-NN (d* = 2 on the torus)
python benchmarks/canonical_tests/protein_backbone_manifold.py --remap-u-rama

# MLP architecture sweep (Standard / Manifold / Wide / UnivBottleneck)
python benchmarks/canonical_tests/backbone_mlp_benchmark.py --remap-u-rama

# Context-only experiment + Richardson density maps
python benchmarks/canonical_tests/backbone_rama_benchmark.py --remap-u-rama
```

All experiments use random seed 42, an 80/20 stratified split, 40 epochs, batch
size 512, Adam at 1e-3, and a variance threshold tau = 0.90 for the
intrinsic-dimensionality probe. The scripts read a PISCES-1000 backbone parquet
cache produced by ProteusPy's `BackboneLoader`; pass the cache location via the
script's cache argument.

## Contents

| File | Description |
|------|-------------|
| `feature_engineering_study.md` | Full working study writeup |
| `backbone_analysis.md` | Extended analysis notes |
| `backbone_mlp_{report,gpo,phys,omega,noaa}.md` | Per-augmentation metric tables |
| `backbone_rama_report.md` | Context-only experiment metrics |
| `pisces_{1000,50k_rama,5k_test}_report.md` | Dataset / corpus reports |
| `backbone_dssp_report.md` | DSSP labelling report |
| `backbone_manifold_{grid.vti,pointfield.npz,voxels.png}` | Manifold field artifacts |
| `{gpo,noaa,omega,phys}/`, `rama/` | Per-experiment result figures |
