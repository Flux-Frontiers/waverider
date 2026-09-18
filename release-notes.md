# Release Notes — v0.15.0

> Released: 2026-09-18

Two completed calibration experiments, a negative result worth having, and a silent seven-minute hang fixed.

## What changed

**CIFAR-10 has no plateau, and the shipped design rule undershoots.** Experiment 4 (`prescription`) swept 30 widths over 3 trials each and found the optimum sharp rather than broad — only w=48, 57, 59 are statistically indistinguishable from the best — and the design formula the project ships lands at roughly half the empirical optimum. Experiment 5 (`probe-convention`) reran the dimension-probe identity under both aggregation conventions: it reproduces the published number exactly under global-mean aggregation, the one the probe was originally validated with, and fails under per-class-max, the one every `w*` in the ResNet experiments actually uses. The identity holds by construction; the one substantive claim doesn't survive the switch. Both runs are seeded, and estimator provenance — every parameter that went into a given `d*` — now travels with the artifact so a result can be checked without rerunning it.

**A calibration script was hanging for seven minutes at 0% CPU, silently.** `estimator_calibration.py` bootstrapped TensorFlow lazily, after the estimator sweep had already claimed Accelerate's BLAS threadpool — every other canonical benchmark initializes TensorFlow before that happens. The deadlock looked like a slow run, not a bug; the same fit now takes 11 seconds.

**Housekeeping.** `--k-pca` defaults are now unified at 25 across the benchmark scripts that had drifted to 25, 30 and 50 independently — the inconsistency was silently producing different CIFAR-10 readings from scripts that were supposed to agree. `--gpu` is renamed `--metal` to match the rest of the fleet's convention, with `--gpu` kept as an alias. `quiltwright` moves to `>=0.14.1`, a currency bump only.

## Upgrading

`poetry update quiltwright` or a fresh install. Reproducing a pre-existing artifact recorded at k-pca 30 or 50 needs the flag passed explicitly now that 25 is the default.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
