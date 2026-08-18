# Estimator calibration — results

**Run:** 2026-08-18, `tesla` (macOS 26.6, arm64), Python 3.12.13, NumPy 1.26.4
**Code:** `benchmarks/canonical_tests/estimator_calibration.py` @ `7b39a7a`
**Artifacts:** `estimator_calibration_{synthetic,profile_cifar10,noise_cifar10}_results.json`

Experiments 1–3 ran. Experiments 4 and 5 have not been run yet.

**Correction to an earlier version of this file.** It said 4 and 5 were blocked
because TensorFlow reports no GPU here. That was wrong, and it was wrong because
it took the handoff's premise instead of checking the repo. **These benchmarks
run on CPU by design.** `benchmarks/tf_setup.py` defaults to CPU and says why —
the Accelerate/AMX path beats Metal for models this size, and per-op GPU sync
dominates on small MLPs. More to the point,
`resnet_manifold_architecture_results.json` — the run that produced
`d* = 19 → w* = 28`, and the baseline this experiment's width sweep must be
commensurable with — records `device_used: "CPU (forced)"`. So do both
dimension-probe runs, the MNIST run and the Iris run. Running experiment 4 on
CPU is not a compromise; it is the same device the numbers it is compared
against came from.

**Measured cost, rather than guessed:** one epoch of a `ManifoldResNet` at
w=28 on CIFAR-10 takes **11.3 s** on this machine. The full experiment is
30 widths × 3 trials × 60 epochs = **5,400 epochs, roughly 15–20 hours**. An
overnight run, not a different project.

**A deadlock in the harness had to be fixed first**, and it is worth knowing
about because it fails silently. `estimator_calibration.py` bootstrapped
TensorFlow lazily, inside `run_prescription`, after the estimator sweep had
already driven Accelerate's BLAS threadpool. Every other canonical benchmark
calls `setup_tensorflow()` at module scope before importing keras. With the lazy
version the first `model.fit()` hung: main thread parked in
`ProcessFunctionLibraryRuntime::RunSync → Notification::WaitForNotification`
while every Eigen worker idled in `WaitForWork`. Seven minutes at 0% CPU, no
error, no timeout — it would have looked like a slow run. Moving the bootstrap
to module scope fixes it; the smoke test then completes in about four minutes.

---

## The one-line verdict

**CIFAR-10 has no plateau.** The local-PCA estimate climbs monotonically from
4 to 74 as `k` goes 5 → 200, and never stops climbing. There is no probed scale
at which it stabilises, so **every `d*` in this repo is a function of the `k` it
was measured at** rather than a property of the dataset. That is the more
important of the two outcomes experiment 2 could have had, and it is not a failed
run.

The estimator is not broken. On synthetic manifolds of true dimension 5 and 10 it
*does* saturate, and the plateau value plus the τ-correction agrees with an
independent MLE to within 0.2 dimensions. It stops saturating at true `d` ≥ 20 —
which is the regime the image estimates sit in.

---

## 1 · `synthetic` — bias against known dimension

n=5000, ambient=256, 150 probe points, seeds fixed. Median `d̂` at τ=0.90:

| manifold | true `d` | k=5 | k=10 | k=25 | k=50 | k=100 | k=200 |
|---|---|---|---|---|---|---|---|
| cube | 5 | 3 | 4 | 4 | 4 | 4 | 4 |
| cube | 10 | 3 | 5 | 7 | 8 | 8 | 9 |
| cube | **20** | 3 | 7 | 12 | 14 | 16 | **17** |
| cube | **40** | 4 | 7 | 16 | 23 | 28 | **32** |
| sphere | 5 | 3 | 4 | 4 | 5 | 5 | 5 |
| sphere | 10 | 3 | 5 | 8 | 8 | 9 | 9 |
| sphere | **20** | 3 | 7 | 12 | 15 | 16 | **17** |
| sphere | **40** | 4 | 7 | 16 | 23 | 29 | **32** |

Three things this settles.

**It under-reports at every setting, and the shortfall widens with `d`.** At
true `d`=5 it converges on 4–5; at `d`=40 it is still at 32 after probing a fifth
of the dataset. The manuscript's retracted "conservative (upper bound)"
characterisation was wrong in direction, and this is the measurement that says
so. Note the ceiling is not the arithmetic one — a `k`-neighbour local covariance
has rank at most `k−1`, but at true `d`=5 the estimate sits at 4 while `k−1` is
199, so the saturation is the manifold's, not the budget's.

**Saturation is what distinguishes a usable measurement from an artefact**, and
it disappears between `d`=10 and `d`=20:

| manifold | true `d` | plateau | τ-corrected | MLE(k=5) | gap |
|---|---|---|---|---|---|
| cube | 5 | 3.83 | 4.26 | 4.49 | 0.23 |
| cube | 10 | 7.67 | **8.52** | **8.57** | **0.05** |
| sphere | 5 | 4.60 | 5.11 | 5.01 | 0.10 |
| sphere | 10 | 8.50 | **9.44** | **9.59** | **0.15** |
| swissroll | 5 | 3.83 | 4.26 | 4.20 | 0.06 |
| swissroll | 10 | 7.75 | 8.61 | 7.87 | 0.74 |
| cube/sphere/swissroll | 20, 40 | **none** | — | 13.2–33.7 | — |

Where a plateau exists, plateau + τ-correction and an independent MLE agree to
within a quarter of a dimension on the cube and sphere — two procedures with no
shared machinery landing on the same answer. The swissroll at `d`=10 is the
exception at 0.74. Where no plateau exists, no number should be quoted as `d*`
without its `k`.

**One caution on plateau detection, from the torus rows.** The torus is the only
manifold where the *clean* run found no plateau and the *noisy* one did — and the
plateau it found was **7.67 on a true `d`=5 manifold**, τ-corrected to 8.52. A
detected plateau is therefore not self-certifying: it can be produced by noise
structure rather than by a tangent space. Any plateau quoted as a `d*` should be
cross-checked against MLE, which on that row said 6.33 and was closer to the
truth. (The torus also needs `2d` coordinates to embed, so its `true_d` column is
the parameter passed, not the dimension of the surface — read those four rows
with that in mind.)

The `d` ≥ 20 breakdown is the Narayanan–Mitter sample-complexity wall arriving on
schedule: samples needed grow exponentially in the intrinsic dimension, and 5,000
points is not enough to certify 20. Recording where it appears as a function of
`n` is itself a result and has not been done yet.

**Ambient noise destroys the plateau outright**, and this was not anticipated in
the handoff. At σ=0.02 — small next to a unit-scale manifold — every plateau
vanishes and both reference estimators blow up:

| manifold | true `d` | MLE(k=5), clean | MLE(k=5), σ=0.02 |
|---|---|---|---|
| cube | 5 | 4.5 | **30.3** |
| swissroll | 5 | 4.2 | **65.2** |
| swissroll | 10 | 7.9 | **62.2** |
| sphere | 10 | 9.6 | **15.0** |

A 6–15× inflation from noise a plotting routine would not show. Real data is not
noiseless, so this bears directly on how the CIFAR-10 result below should be
read: "no plateau" is what both a too-high-dimensional manifold *and* a
moderately noisy low-dimensional one look like, and this experiment does not
separate them.

**TwoNN over-reports everywhere** — 8.8 at true `d`=5, 15.0 at true `d`=10 —
opposite in sign to the other two. Consistent with Pope et al.'s finding of poor
sample efficiency at higher dimension. Recorded, not tuned away.

## 2 · `profile` — is there a stable scale on CIFAR-10?

10,000 points, 300 probe points, τ=0.90, seed 20260818.

| k | median `d̂` | median k-NN radius |
|---|---|---|
| 5 | 4 | 42.41 |
| 10 | 7 | 43.50 |
| 25 | **16** | 45.16 |
| 50 | **29** | 46.73 |
| 100 | 48 | 48.39 |
| 200 | 74 | 50.30 |

`plateau: null`. `reach proxy: null`.

**No probed scale gives a stable tangent estimate.** The estimate rises by 18×
while the neighbourhood radius rises by 19% — so `k` is not selecting a scale
here in any meaningful sense. In 3,072 dimensions the nearest 5 and the nearest
200 neighbours sit at almost the same distance, and what `k` actually varies is
the sample size the local covariance is estimated from. The estimate tracks that.

Compare against the synthetic table: CIFAR-10 climbs *faster* than a true-`d`=40
manifold does (4 → 74 versus 4 → 32 over the same `k` range). Whatever this is,
it does not behave like a manifold of dimension ≤ 40 sampled at this density.

**It also reproduces the entire 34/29/19/16 spread from a single run**, which
until now had to be inferred across three committed artifacts:

| statistic | k=25 | k=50 | committed artifact |
|---|---|---|---|
| per-class max | **19** | **33** | 19 (`resnet_manifold_architecture`), 34 (`cifar10_architecture`) |
| global mean | **16** | **28** | 16 (probe), 29 (`cifar10_architecture`) |

Four numbers, one dataset, one script, one run — separated only by `k` and the
choice of statistic. The 33-versus-34 and 28-versus-29 differences are
probe-sampling noise; this run is seeded and those were not.

Reference estimators on the same data: **MLE 28.4 / 28.0 / 25.9 / 24.3** at
k=3/5/10/20, **TwoNN 50.3**. Pope et al. report CIFAR-10 at 13–26 by MLE. Ours
is at the top of that band or just above it, and our canonical `d*`=19 at k=25
sits *below* every MLE reading — the opposite of the direction one would guess
from the "large `k` over-estimates" warning.

## 3 · `noise` — Pope et al. Table 3 replication

Inject a `d`-dimensional uniform hypercube into CIFAR-10 and check the estimate
tracks the known increment.

| injected `d` | Pope MLE(k=3) | ours MLE(k=3) | ours MLE(k=5) |
|---|---|---|---|
| 0 (baseline) | — | 28.3 | 27.9 |
| 256 | 19.7 | 41.0 | 41.7 |
| 512 | 30.9 | 53.3 | 52.4 |
| 1024 | 57.1 | 71.8 | 71.5 |
| 1536 | 77.8 | 91.4 | 85.1 |
| 2048 | 110.0 | 113.7 | 108.8 |
| 2560 | 136.1 | 127.3 | 126.8 |

**The estimator responds to dimension.** A flat response would have been
serious; this is a 4.5× rise tracking the injected dimension monotonically, and
it converges on Pope's column by `d`=2048. The gap at low injected `d` is the
baseline offset — ours starts at 28.3 where theirs starts near 19.7, consistent
with the profile result above and with our using `StandardScaler` where they use
raw pixels. Exact agreement was not expected.

Worth noting what both columns show: injecting 2,560 known dimensions moves the
MLE by about 100. The estimator is directionally right and quantitatively short,
in both labs.

`plateau: none` at every level, including the un-injected baseline.

---

## What this changes

1. **`d*` cannot be quoted without `k`.** Now enforced: `DEFAULT_K_PCA = 25` is
   shared across the benchmark scripts and every results JSON carries an
   `"estimator"` block. The canonical convention is k=25, τ=0.90, per-class max.
2. **Prefer plateau + τ-correction to a single-`k` reading, where a plateau
   exists — and cross-check it against MLE.** The two agree to within a quarter
   of a dimension on clean cubes and spheres, but the torus rows show a plateau
   can also be manufactured by noise. On CIFAR-10 no plateau exists at all, and
   that has to be said rather than worked around.
3. **The manuscript's estimator-sensitivity limitation is now backed by
   measurement**, not by citation of someone else's spread.
4. **Experiment 4 is still the decision point and is now the only thing
   outstanding.** Nothing here tells us whether the optimal bottleneck width is
   invariant to which estimator you start from. Given that CIFAR-10 has no stable
   `d̂` at all, that question is sharper than it was this morning, not softer: if
   `w*` tracks whichever `k` was chosen, the prescription is circular; if the
   accuracy optimum sits in one place regardless, then `k=25` is a calibration
   constant and the rule survives with an honest caveat.

   The smoke run makes one thing concrete: on CIFAR-10 the twelve estimator
   settings prescribe `w` anywhere from the low twenties to **111**. Whatever the
   accuracy curve does, it will not endorse all of them.

## To run experiments 4 and 5

CPU, no flags — the default, and the device the comparison baselines were
produced on. `--metal` opts in to the GPU if you want to compare; `--gpu` is
accepted as an alias.

```bash
# smoke first: ~4 minutes, 5 widths x 1 trial x 3 epochs
python benchmarks/canonical_tests/estimator_calibration.py prescription \
    --dataset cifar10 --quick

# the real thing: 30 widths x 3 trials x 60 epochs, ~15-20 h
nohup python benchmarks/canonical_tests/estimator_calibration.py \
    prescription --dataset cifar10 > prescription.log 2>&1 &

python benchmarks/canonical_tests/estimator_calibration.py probe-convention --dataset cifar10
```

Use `nohup`; a run this long will not survive a terminal teardown, and the
failure leaves no marker.

If `prescription` returns `optimum_is_broad: true`, report it. Do not narrow the
width grid or re-seed to sharpen the peak.
