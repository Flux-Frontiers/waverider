#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# run_all_tests.sh — Drive all canonical benchmark tests
#
# Author: Eric G. Suchanek, PhD
# Affiliation: Flux-Frontiers
# Date: 2026-03-28
#
# Usage:
#   ./run_all_tests.sh            # run everything with defaults
#   EPOCHS=10 TRIALS=2 ./run_all_tests.sh   # quick smoke-test
#
# Results (JSON + PNG) land beside each script in this directory.
# Per-test stdout/stderr is tee'd to logs/<script>.log
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Tunables with environment-variable overrides
EPOCHS="${EPOCHS:-}"        # empty = use each script's default
TRIALS="${TRIALS:-}"

run() {
    local name="$1"
    shift
    local log="${LOG_DIR}/${name}.log"
    echo ""
    echo "================================================================"
    echo "  Running: ${name}"
    echo "  Log:     ${log}"
    echo "================================================================"
    local t0=$SECONDS
    python "$@" 2>&1 | tee "${log}"
    local elapsed=$(( SECONDS - t0 ))
    echo ""
    echo "  >> ${name} done in ${elapsed}s"
}

cd "${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# Iris — fast, CPU-only, no TF
# ---------------------------------------------------------------------------

run iris_manifold_adam_walker \
    iris_manifold_adam_walker.py \
    ${EPOCHS:+--epochs "$EPOCHS"} \
    ${TRIALS:+--trials "$TRIALS"}

run iris_adam_vs_manifold \
    iris_adam_vs_manifold.py \
    ${EPOCHS:+--epochs "$EPOCHS"} \
    ${TRIALS:+--trials "$TRIALS"}

# ---------------------------------------------------------------------------
# Digits — sklearn toy dataset, no CLI args
# ---------------------------------------------------------------------------

run digits_manifold_model \
    digits_manifold_model.py

run digits_manifold_knn \
    digits_manifold_knn.py

# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

#run mnist_manifold_model \
#    mnist_manifold_model.py

#run mnist_manifold_architecture \
#    mnist_manifold_architecture.py \
#    ${EPOCHS:+--epochs "$EPOCHS"} \
#    ${TRIALS:+--trials "$TRIALS"}

# ---------------------------------------------------------------------------
# CIFAR-10
# ---------------------------------------------------------------------------

run cifar10_manifold_model \
    cifar10_manifold_model.py

run cifar10_manifold_architecture \
    cifar10_manifold_architecture.py \
    ${EPOCHS:+--epochs "$EPOCHS"} \
    ${TRIALS:+--trials "$TRIALS"}

# ---------------------------------------------------------------------------
# CIFAR-100
# ---------------------------------------------------------------------------

run cifar100_manifold_architecture \
    cifar100_manifold_architecture.py \
    ${EPOCHS:+--epochs "$EPOCHS"} \
    ${TRIALS:+--trials "$TRIALS"}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "================================================================"
echo "  ALL TESTS COMPLETE"
echo "  Logs in: ${LOG_DIR}"
echo "================================================================"
