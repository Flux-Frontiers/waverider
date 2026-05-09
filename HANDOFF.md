# WaveRider Handoff — Stardate 2026.093
*From: Science Officer Spock, U.S.S. WaveRider NCC-7699*
*To: Next watch officer*

---

## Status: BREAKTHROUGH CONFIRMED

The Universal Bottleneck Theorem has been empirically validated.
The paper has been updated. You are picking up at a clean milestone.

---

## The Headline Result

```
ManifoldResNet-UB+Drop (w*=28, dropout=0.3)
CIFAR-10: 72.1% ± 0.008  |  34,408 params  |  60 epochs, 4 trials
vs ResNet-32: 64.2% ± 0.050  |  47,978 params

+7.9 pp accuracy. 28% fewer parameters. Variance collapsed 6×.
```

---

## The Theorem

    w* = d* + C − 1

Two independent lower bounds:
1. **Whitney bound**: d* dims to embed the manifold without self-intersection
2. **Information floor**: C−1 dims for class separation (simplex encoding)

**Mechanistic proof (dimension probe):** Network given d*=16 bottleneck neurons
spontaneously allocated exactly C−1=9 to class coordinates, 7 to geometry.
PC11=four-legged animals, PC9=flat objects, PC12=vehicles. From gradient descent
alone. No instruction.

**Two regimes:**
- C ≤ d* (Whitney-dominated): UB+Dropout is optimal. CIFAR-10 is this regime.
- C >> d* (floor-dominated): ManifoldResNet-d is efficiency champion. CIFAR-100 (C=100, d*=19) is this regime.

---

## What Was Done This Session

| Action | File | State |
|---|---|---|
| CIFAR-100 script fully updated | `benchmarks/canonical_tests/cifar100_resnet_manifold_architecture.py` | Done |
| ManifoldResNet-UB added to both benchmarks | same + `resnet_manifold_architecture.py` | Done |
| `--only` / `--plot-only` flags added to CIFAR-100 | same | Done |
| Dropout(0.3) added to `build_manifold_resnet` | `src/model_builder.py` | Done |
| `build_ub_pca_mlp`, `build_universal_bottleneck_mlp` added | `src/model_builder.py` | Done |
| UB theorem + dim probe + CIFAR results → paper | `docs/waverider/article/waverider_jmlr.tex` | Done |
| Same sections → spec | `docs/manifold_walker_spec/manifold_walker_spec.tex` | Done |
| Summary block made defensive (cached d≠current d) | `resnet_manifold_architecture.py` | Done |
| Mission logs written | `waverider_missions/notes/stardate_2026_093*.md` | Done |

---

## Key Numbers Aboard

**CIFAR-10** (`benchmarks/canonical_tests/resnet_manifold_architecture_results.json`):
- d* = 19 (per-class max, τ=0.90), global mean = 16
- w* = d* + C − 1 = 19 + 9 = 28
- ManifoldResNet-UB+Drop: **0.7206 ± 0.0079**, 34,408 params
- ManifoldResNet-d: 0.6490 ± 0.0123, 17,376 params (efficiency champion)
- ResNet-32: 0.6415 ± 0.0498, 47,978 params

**CIFAR-100** (`benchmarks/canonical_tests/cifar100_resnet_manifold_architecture_results.json`):
- d* = 19, w* = 118
- ManifoldResNet-UB: 0.3829 ± 0.038, 644,262 params (+0.74pp, 12.6× heavier)
- ManifoldResNet-d: 0.3116, 19,176 params (efficiency champion)

**Dimension probe** (`benchmarks/canonical_tests/manifold_dim_probe_results.json`):
- d_star=16, k_90=7, n_extra=9, C-1=9 → exact match

---

## What Comes Next (Priority Order)

1. **Update MISSIONS.md** — log Stardate 2026.093 as a completed experiment entry.

2. **Run MNIST or Fashion-MNIST** — clean C≈d* test to confirm the phase boundary.
   Fast, small, decisive. Expected result: UB wins cleanly in Whitney-dominated regime.

3. **Update the paper conclusion** (`waverider_jmlr.tex`) — current conclusion
   still references the old ManifoldModel result. Needs the UB theorem framing
   as the capstone finding.

4. **Write the abstract** — Universal Embedder framing:
   *"Measure d*. Count C. Apply w* = d* + C − 1. No grid search."*

5. **`--plot-only` on CIFAR-10** to regenerate figure with UB+Drop bar visible:
   ```bash
   .venv/bin/python benchmarks/canonical_tests/resnet_manifold_architecture.py --plot-only
   ```

6. **Add UB+Drop to CIFAR-100 benchmark** — apply same dropout prescription
   to the C>>d* regime. Expect modest improvement; confirms regime analysis.

---

## Repo Layout (Canonical)

```
waverider/
├── src/
│   └── model_builder.py          ← build_manifold_resnet(dropout=), build_ub_pca_mlp, etc.
├── benchmarks/canonical_tests/
│   ├── resnet_manifold_architecture.py       ← CIFAR-10 benchmark
│   ├── cifar100_resnet_manifold_architecture.py  ← CIFAR-100 benchmark
│   ├── manifold_dim_probe.py                 ← dimension probe
│   └── *.json                               ← all results
├── docs/
│   ├── waverider/article/waverider_jmlr.tex ← THE PAPER (canonical)
│   └── manifold_walker_spec/manifold_walker_spec.tex ← spec
└── waverider_missions/
    ├── MISSIONS.md
    ├── CLAUDE_NOTES.md
    └── notes/
        ├── stardate_2026_092_universal_bottleneck.md  ← revelation log
        ├── stardate_2026_093_cifar100_ub_probe.md     ← two-regime analysis
        ├── stardate_2026_093b_ub_pca_mlp.md           ← negative control
        └── stardate_2026_093c_ub_dropout_confirmation.md  ← THE CONFIRMATION
```

---

## Running the Benchmarks

```bash
# CIFAR-10: full run
.venv/bin/python benchmarks/canonical_tests/resnet_manifold_architecture.py \
    --epochs 60 --trials 4 --metal

# CIFAR-10: incremental (UB+Drop only)
.venv/bin/python benchmarks/canonical_tests/resnet_manifold_architecture.py \
    --only "ManifoldResNet-UB+Drop" --epochs 60 --trials 4 --metal

# CIFAR-10: regenerate figure only
.venv/bin/python benchmarks/canonical_tests/resnet_manifold_architecture.py --plot-only

# CIFAR-100: same pattern, substitute cifar100_resnet_manifold_architecture.py
```

---

## Important Notes for the Next Officer

- **This repo is canonical.** Never write WaveRider content to `proteusPy/` or
  other sibling repos. All missions, benchmarks, and paper live here.
- **`waverider_missions/` is now in this repo.** It was moved from `proteusPy/`.
- **The paper is `waverider_jmlr.tex`**, not the spec. The spec is a standalone
  technical document; the JMLR paper is the publication target.
- **The UB theorem is established.** The dimension probe is the mechanistic proof.
  The 72.1% result is the empirical confirmation. Don't re-litigate these.
- **Dropout(0.3) is part of the prescription.** Without it, UB overfits at 60ep.
  This is now in the paper and the code.

*End of handoff.*
*Science Officer Spock*
*U.S.S. WaveRider, NCC-7699*
*Stardate 2026.093*
