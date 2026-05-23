"""Shared TensorFlow bootstrap for benchmark scripts.

Keeps startup behavior consistent across benchmarks:
- set TF log level early
- optionally force CPU unless an explicit GPU flag is present
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

    :param gpu_flag: Optional CLI flag that enables GPU/Metal when present
        (example: "--metal" or "--gpu"). If absent or not provided, CPU is forced.
    :param argv: Optional argv iterable. Defaults to sys.argv.
    :param tf_log_level: Value for TF_CPP_MIN_LOG_LEVEL.
    :returns: Tuple (tf_module, device_info_dict).
    """
    args = list(sys.argv if argv is None else argv)
    use_gpu = bool(gpu_flag and gpu_flag in args)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(tf_log_level))
    if not use_gpu:
        # Keep benchmark behavior deterministic and avoid per-op GPU sync overhead
        # on smaller models unless explicitly requested.
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
        if gpu_flag == "--metal":
            device_used = f"Metal GPU ({gpu_name})"
        else:
            device_used = f"GPU ({gpu_name})"
    elif use_gpu and not gpus:
        device_used = "CPU (requested GPU, none detected)"
    else:
        device_used = "CPU (forced)"

    device_info = {
        "tensorflow_version": tf.__version__,
        "device_used": device_used,
    }
    print(f"TensorFlow {tf.__version__} | Device: {device_used}")

    return tf, device_info
