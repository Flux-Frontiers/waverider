#!/usr/bin/env bash
# E4 scale-hypothesis test: does w = TwoNN + C - 1 hit the empirical optimum
# on datasets beyond CIFAR-10?
#
# Background: the CIFAR-10 prescription run (estimator_calibration_report.md
# section 4) found the empirical optimum at w=59 -- an exact hit for the TwoNN
# prescription and a 2x undershoot for the shipped k=25 per-class-max
# convention.  One exact hit on one dataset is suggestive, not evidence -- but
# it is cheaply testable, and it is the difference between "our prescription
# failed" and "our prescription was measuring the wrong scale".  This script
# runs the identical E4 protocol (30 widths x 3 trials x 60 epochs, CPU, same
# defaults) on MNIST and Fashion-MNIST, where a full sweep costs a fraction of
# the CIFAR-10 run's ~29 h.
#
# Usage (from any machine with the repo + poetry env set up):
#
#     cd benchmarks/canonical_tests
#     nohup ./run_e4_scale_test.sh > /dev/null 2>&1 &
#
# or in the foreground:
#
#     ./run_e4_scale_test.sh                    # mnist then fashion_mnist
#     ./run_e4_scale_test.sh fashion_mnist      # just one dataset
#     E4_QUICK=1 ./run_e4_scale_test.sh mnist   # smoke test first (~minutes)
#
# Each dataset writes estimator_calibration_prescription_<dataset>_results.json
# beside estimator_calibration.py, and a timestamped log under logs/.
#
# Author: Eric G. Suchanek, PhD -- Flux-Frontiers

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=(mnist fashion_mnist)
fi

QUICK_FLAG=""
if [ "${E4_QUICK:-0}" = "1" ]; then
    QUICK_FLAG="--quick"
fi

cd "$REPO"

# Python block-buffers stdout when piped (as through tee below), which holds
# printed lines back for minutes during training; force line-by-line output.
export PYTHONUNBUFFERED=1

for ds in "${DATASETS[@]}"; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="$LOGDIR/e4_${ds}_${stamp}.log"
    echo "=== prescription --dataset $ds $QUICK_FLAG -> $log"
    # Defaults match the CIFAR-10 E4 run exactly: 3 trials, widths 8..64
    # step 4 plus every prescribed width, 60 epochs, batch 512, dropout 0.3,
    # CPU (the device every comparison baseline was produced on).
    poetry run python benchmarks/canonical_tests/estimator_calibration.py \
        prescription --dataset "$ds" $QUICK_FLAG 2>&1 | tee "$log"
done

echo "=== done: ${DATASETS[*]}"
