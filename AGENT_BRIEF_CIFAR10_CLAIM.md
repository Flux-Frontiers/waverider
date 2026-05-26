# Agent Brief — Verify the CIFAR-10 vs. ResNet +8.5 pp Claim

**Date:** 2026-05-24
**Requested by:** Eric (via the kgrag_priv outreach letter audit)
**Priority:** Blocker for outbound communication

---

## Context

We're sending an outreach letter to xAI / SpaceX (see
`/Users/egs/repos/kgrag_priv/docs/spacex_xai_outreach.md`).
The letter currently includes this claim under WaveRider:

> - **CIFAR-10:** formula-derived architecture beats ResNet by **+8.5 pp** with fewer parameters

We audited the letter against the most recent canonical runs and could
not reconcile this number. The new canonical CIFAR-10 results
(`benchmarks/canonical_tests/cifar10_analysis.md`, generated 2026-05-24)
show:

- Standard MLP (1024→512), 3.67M params → 51.67% ± 0.75%
- Best manifold (PCA→34D + MLP), 5,076 params → 49.12% ± 0.46%
- **Standard wins by +2.6 pp; no ResNet baseline appears in the run at all.**

This contradicts the letter's claim of +8.5 pp over ResNet. Before the
letter goes out, we need to know whether the claim is recoverable, or
whether it needs to be revised or dropped.

## What We Need You To Do

1. **Hunt for the source of the +8.5 pp claim.** Likely places:
   - `papers/manifold_classification/` — DATA.md and result PNGs
   - Older benchmark runs in `benchmarks/` (anything pre-2026-05-24)
   - `CHANGELOG.md` and recent commit history
   - The patent application (`patents/embedder_patent/` if present
     in waverider, or check kgrag_priv/patents/embedder_patent/)
   - Any cifar10_*.json result files

   Grep targets: `+8.5`, `8.5 pp`, `ResNet` + `CIFAR-10` in the same
   paragraph, `Universal Bottleneck` + CIFAR-10.

2. **If you find the run:** confirm the numbers, the experimental setup,
   and whether it's reproducible from the current code. Report:
   - Source file / commit where the result lives
   - The actual ResNet baseline accuracy and the UBT-derived architecture
     accuracy
   - Parameter counts for both
   - Whether the run used early stopping (the new canonical does)
   - Any caveats (e.g., different epoch budget, different data aug)

3. **If you cannot find it, or it doesn't reproduce:** propose a
   replacement bullet for the letter. Acceptable replacements:
   - The actual best CIFAR-10 efficiency number from the new canonical
     run (e.g., "724× fewer parameters at −2.6 pp")
   - A defensible UBT-derived result on Fashion-MNIST or another dataset
     if CIFAR-10 doesn't support the claim
   - Or honest deletion of the bullet

4. **Run a fresh CIFAR-10 UBT-vs-ResNet experiment** if practical.
   The formula is `w* = d* + C − 1`. For CIFAR-10: `d* = 34, C = 10`,
   so `w* = 43`. Build the UBT-spec'd architecture, compare against a
   matched-parameter or standard ResNet baseline with early stopping
   (patience=10 on val_accuracy) over the same 4-trial / 60-epoch
   protocol used in `cifar10_analysis.md`.

## Acceptance

A short markdown report (`CIFAR10_CLAIM_VERIFICATION.md` in the
waverider root) containing:

- **Verdict:** RECOVERED / REVISED / DROPPED
- The specific bullet text to use in the outreach letter
- Pointers to the evidence (file paths, commit SHAs, run logs)
- If REVISED or DROPPED: a one-sentence rationale Eric can paste into
  the outreach letter's audit trail

## Tone Note

The outreach letter goes to a sophisticated audience (Elon-orbit at
xAI/SpaceX). Any benchmark claim has to be defensible against someone
who clones the repo and runs the code themselves. Overstating is worse
than understating here. The CIFAR-100 result already in the letter
(25.70% at 279× fewer params, geometry beating brute force) is strong
on its own — we do not need to oversell CIFAR-10 to make the section
work.
