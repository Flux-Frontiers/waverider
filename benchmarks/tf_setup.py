"""Shared TensorFlow bootstrap for benchmark scripts.

Keeps startup behavior consistent across benchmarks:
- set TF log level early
- default to CPU (avoids per-op GPU sync overhead on small MLPs); pass
  gpu_flag or use --gpu / --metal to opt in
- initialize memory growth for detected GPUs
- print a standard device banner
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable


def setup_tensorflow(
    *,
    gpu_flag: str | None = None,
    argv: Iterable[str] | None = None,
    tf_log_level: str = "2",
):
    """Configure and import TensorFlow for benchmark scripts.

    CPU is the default device. Pass ``gpu_flag`` (e.g. ``"--gpu"`` or
    ``"--metal"``) and include that flag in argv to opt in to GPU execution.

    :param gpu_flag: CLI flag that enables GPU/Metal when present in argv.
    :param argv: Optional argv iterable. Defaults to sys.argv.
    :param tf_log_level: Value for TF_CPP_MIN_LOG_LEVEL.
    :returns: Tuple (tf_module, device_info_dict).
    """
    args = list(sys.argv if argv is None else argv)

    # --gpu / --metal anywhere in argv always opts in regardless of gpu_flag param
    explicit_gpu = "--gpu" in args or "--metal" in args
    use_gpu = explicit_gpu or bool(gpu_flag and gpu_flag in args)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(tf_log_level))
    if not use_gpu:
        # CPU default: avoids per-op Metal/CUDA sync overhead on small MLPs;
        # Accelerate/AMX path is faster than Metal for these model sizes.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import tensorflow as tf  # noqa: PLC0415

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if use_gpu and gpus:
        gpu_name = gpus[0].name
        device_used = f"Metal GPU ({gpu_name})" if "--metal" in args else f"GPU ({gpu_name})"
    elif use_gpu and not gpus:
        device_used = "CPU (no GPU detected)"
    else:
        device_used = "CPU"

    device_info = {
        "tensorflow_version": tf.__version__,
        "device_used": device_used,
    }
    print(f"TensorFlow {tf.__version__} | Device: {device_used}")

    return tf, device_info
