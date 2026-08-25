"""Phase 2: the local-vs-global reconstruction gap on real transformer KV-cache data.

Per ``MANIFOLD_REACH_QUANTIZATION_PLAN.md``, this is the phase that actually
answers the question Phase 0/1 only rehearsed on synthetic manifolds: does
real embedding geometry curve at a scale a quantizer's bit budget could
exploit? It only runs after Phase 0 (instrument validated) and reads Phase 1
(the instrument's own failure regime) as context, not as a stop condition --
Phase 1 used a fixed ``k = d*`` on generic sphere/torus manifolds, not a real
quantization rank budget on real data, so its d*=100 collapse does not by
itself say anything about the ``k`` a real bit budget would actually use here
(``head_dim / 8`` to ``head_dim / 2``, i.e. 8-32 for GPT-2's head_dim=64).

Pipeline:

1. Load a small causal LM (default: GPT-2 small, 12 layers x 12 heads x
   head_dim=64) and a real corpus (WikiText-2 raw test split -- the same
   calibration set ResQ uses).
2. Run each sequence through the model with ``use_cache=True`` and pull the
   per-layer, per-head key and value tensors straight out of the attention
   cache -- these are exactly the vectors a KV-cache quantizer would compress.
3. Unit-normalize each vector. This is not optional bookkeeping: TurboQuant's
   published MSE distortion floor (the decision criterion below) is derived
   for unit-norm vectors, and unit-normalized embeddings are also the
   standard representation for the vector-database application this plan
   targets (nearest-neighbour search is cosine similarity there). Mean
   centering per fit is still handled internally by
   ``waverider.manifold_reach``, same as Phase 0/1.
4. Run the identical ``reconstruction_gap`` / ``gaussian_null`` /
   ``corrected_gap`` pipeline as Phase 0/1, at rank budgets representative of
   an actual bit allocation rather than a manifold's true dimension.
5. Compare the corrected gap against TurboQuant's published MSE floor at
   b = 1, 2, 3, 4 bits/coordinate (0.36, 0.117, 0.03, 0.009 for unit-norm
   vectors). Per the plan's decision criterion, curvature below that floor
   cannot be resolved by a quantizer no matter how real it is.

Every run writes a JSON beside this script with the full per-(layer, head,
key/value) sweep, plus provenance -- same convention as
``estimator_calibration.py`` and ``manifold_reach_calibration.py``.

Cost note: probing every (layer, head) pair at every rank and radius is
expensive. Defaults probe a representative subset (first/middle/last layer,
first two heads) rather than the full grid; widen with ``--layers``/
``--heads`` at the cost of runtime. What is skipped is always logged, never
silently dropped.

Examples::

    python manifold_reach_embeddings.py
    python manifold_reach_embeddings.py --layers 0 5 11 --heads 0 1 2 --k-values 8 16 32
    python manifold_reach_embeddings.py --n-sequences 400 --max-length 96

Author: Eric G. Suchanek, PhD -- Flux-Frontiers
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from waverider.dimensionality_profile import knn_radii  # noqa: E402
from waverider.manifold_reach import (  # noqa: E402
    corrected_gap,
    gaussian_null,
    reconstruction_gap,
)

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20260825
RADIUS_MULTIPLIERS = (0.5, 0.75, 1.0, 1.5, 2.0)

# TurboQuant (arXiv 2504.19874) MSE distortion for unit-norm vectors, by bits
# per coordinate. The decision criterion: curvature below this floor cannot be
# resolved by the quantizer regardless of whether it is real.
TURBOQUANT_MSE_FLOOR = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


# ---------------------------------------------------------------------------
# Provenance (same convention as estimator_calibration.py)
# ---------------------------------------------------------------------------


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def _write(payload, name):
    payload = dict(payload)
    payload["provenance"] = {
        "script": Path(__file__).name,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = HERE / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Corpus: WikiText-2 raw test split, ResQ's own calibration set
# ---------------------------------------------------------------------------


def load_wikitext_sequences(n_sequences, min_words=8, seed=DEFAULT_SEED):
    """Sample real sentences/paragraphs from WikiText-2's test split.

    Fetched as the raw parquet file directly from the HF Hub rather than via
    the ``datasets`` package -- one file read with ``pyarrow`` avoids a much
    heavier dependency for the same data.

    :param n_sequences: Number of text rows to sample.
    :param min_words: Rows shorter than this (mostly WikiText's section-header
        rows, e.g. " = Title = ") are excluded before sampling.
    :param seed: Sampling seed.
    :returns: List of strings.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    texts = pq.read_table(path).column("text").to_pylist()
    candidates = [t.strip() for t in texts if len(t.split()) >= min_words]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=min(n_sequences, len(candidates)), replace=False)
    return [candidates[i] for i in idx]


# ---------------------------------------------------------------------------
# KV-cache extraction
# ---------------------------------------------------------------------------


def extract_kv_cache(model_name, sequences, max_length):
    """Run each sequence through ``model_name``, collecting per-(layer, head)
    key and value vectors straight out of the attention cache.

    :param model_name: HF Hub model id (e.g. ``"gpt2"``).
    :param sequences: List of raw text sequences.
    :param max_length: Truncate tokenized sequences to this many tokens.
    :returns: Tuple ``(keys, values, n_layers, n_heads, head_dim)``. ``keys``
        and ``values`` are dicts mapping ``(layer, head)`` to an array of
        shape ``(n_vectors, head_dim)``, pooling every token position from
        every sequence.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager")
    model.eval()
    n_layers = (
        model.config.num_hidden_layers
        if hasattr(model.config, "num_hidden_layers")
        else model.config.n_layer
    )
    n_heads = (
        model.config.num_attention_heads
        if hasattr(model.config, "num_attention_heads")
        else model.config.n_head
    )

    keys = {
        (layer_idx, head_idx): [] for layer_idx in range(n_layers) for head_idx in range(n_heads)
    }
    values = {
        (layer_idx, head_idx): [] for layer_idx in range(n_layers) for head_idx in range(n_heads)
    }
    head_dim = None
    with torch.no_grad():
        for seq in sequences:
            enc = tok(seq, return_tensors="pt", truncation=True, max_length=max_length)
            out = model(**enc, use_cache=True)
            for layer_idx, layer_cache in enumerate(out.past_key_values.layers):
                k = layer_cache.keys[0].float().numpy()  # (n_heads, seq_len, head_dim)
                v = layer_cache.values[0].float().numpy()
                head_dim = k.shape[-1]
                for head_idx in range(n_heads):
                    keys[(layer_idx, head_idx)].append(k[head_idx])
                    values[(layer_idx, head_idx)].append(v[head_idx])

    keys = {lh: np.concatenate(arrs, axis=0) for lh, arrs in keys.items()}
    values = {lh: np.concatenate(arrs, axis=0) for lh, arrs in values.items()}
    return keys, values, n_layers, n_heads, head_dim


def _unit_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def measure_case(X, k, n_holdout, seed):
    """Corrected-gap radius sweep on one (layer, head, key/value) vector set.

    Same pipeline as Phase 0/1's ``measure_case``: unit-normalized input,
    radii anchored to the data's own k-NN density, held-out reconstruction,
    Gaussian-null subtraction.
    """
    X = _unit_normalize(np.asarray(X, dtype=np.float64))
    n_probe = min(200, len(X) // 2)
    base_radius = knn_radii(X, k=max(k, 5), n_probe=n_probe, seed=seed)["median"]
    radii = [base_radius * m for m in RADIUS_MULTIPLIERS]

    real = reconstruction_gap(X, k=k, radii=radii, n_holdout=n_holdout, seed=seed)
    null_X = gaussian_null(X, seed=seed + 1)
    null = reconstruction_gap(null_X, k=k, radii=radii, n_holdout=n_holdout, seed=seed)
    corrected = corrected_gap(real, null)

    real_by_r = {e["radius"]: e for e in real["by_radius"]}
    null_by_r = {e["radius"]: e for e in null["by_radius"]}
    by_radius = [
        {
            **entry,
            "n_used_real": real_by_r[entry["radius"]]["n_used"],
            "n_used_null": null_by_r[entry["radius"]]["n_used"],
        }
        for entry in corrected
    ]
    return {
        "n_vectors": int(len(X)),
        "n_holdout": int(n_holdout),
        "base_radius_knn_median": base_radius,
        "mse_global_real": real["mse_global"],
        "mse_global_null": null["mse_global"],
        "by_radius": by_radius,
    }


def _usable_gap(case, min_used):
    usable = [
        e["gap_corrected"]
        for e in case["by_radius"]
        if e["n_used_real"] >= min_used and e["n_used_null"] >= min_used
    ]
    return max(usable) if usable else float("nan")


def run(args):
    print(f"Loading {args.n_sequences} WikiText-2 sequences...")
    sequences = load_wikitext_sequences(args.n_sequences, seed=DEFAULT_SEED)
    print(f"Running {args.model} over {len(sequences)} sequences (max_length={args.max_length})...")
    t0 = time.time()
    keys, values, n_layers, n_heads, head_dim = extract_kv_cache(
        args.model, sequences, args.max_length
    )
    print(
        f"  extracted KV cache for {n_layers} layers x {n_heads} heads x head_dim={head_dim} ({time.time() - t0:.1f}s)"
    )

    layers = args.layers or sorted({0, n_layers // 2, n_layers - 1})
    heads = args.heads or list(range(min(2, n_heads)))
    layers = [layer_idx for layer_idx in layers if 0 <= layer_idx < n_layers]
    heads = [h for h in heads if 0 <= h < n_heads]
    skipped_layers = sorted(set(range(n_layers)) - set(layers))
    skipped_heads = sorted(set(range(n_heads)) - set(heads))
    print(
        f"Probing layers {layers} x heads {heads} (skipping layers {skipped_layers}, heads {skipped_heads})"
    )

    k_values = [k for k in args.k_values if k < head_dim]
    n_holdout = min(
        args.n_holdout,
        min(len(keys[(layer_idx, head_idx)]) for layer_idx in layers for head_idx in heads) - 10,
    )

    cases = []
    for layer_idx in layers:
        for head_idx in heads:
            for kind, pool in (("key", keys), ("value", values)):
                X = pool[(layer_idx, head_idx)]
                for k in k_values:
                    t0 = time.time()
                    case = measure_case(
                        X, k, n_holdout, seed=DEFAULT_SEED + layer_idx * 1000 + head_idx * 10 + k
                    )
                    gap = _usable_gap(case, n_holdout // 2)
                    print(
                        f"  layer={layer_idx:>2} head={head_idx} {kind:<5} k={k:>2}  "
                        f"n={case['n_vectors']:>6}  best corrected gap = {gap:+.4g}  ({time.time() - t0:.1f}s)"
                    )
                    cases.append(
                        {
                            "layer": layer_idx,
                            "head": head_idx,
                            "kind": kind,
                            "k": k,
                            "gap_best": gap,
                            **case,
                        }
                    )

    print(
        f"\n  {'layer':>5} {'head':>4} {'kind':<5} {'k':>3}  {'best gap':>10}  vs TurboQuant floor (b=1..4)"
    )
    for c in cases:
        floors = [f"{TURBOQUANT_MSE_FLOOR[b]:.3f}" for b in (1, 2, 3, 4)]
        verdict = (
            "ABOVE b=1 floor" if c["gap_best"] > TURBOQUANT_MSE_FLOOR[1] else "below every floor"
        )
        print(
            f"  {c['layer']:>5} {c['head']:>4} {c['kind']:<5} {c['k']:>3}  {c['gap_best']:>10.4g}  "
            f"floors=[{', '.join(floors)}]  {verdict}"
        )

    _write(
        {
            "model": args.model,
            "n_sequences": len(sequences),
            "max_length": args.max_length,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "layers_probed": layers,
            "heads_probed": heads,
            "layers_skipped": skipped_layers,
            "heads_skipped": skipped_heads,
            "k_values": k_values,
            "n_holdout": n_holdout,
            "turboquant_mse_floor": TURBOQUANT_MSE_FLOOR,
            "cases": cases,
        },
        f"manifold_reach_embeddings_{args.model.replace('/', '_')}_results.json",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="gpt2", help="HF Hub causal LM id")
    parser.add_argument("--n-sequences", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-holdout", type=int, default=150)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="Default: first, middle, last layer"
    )
    parser.add_argument(
        "--heads", type=int, nargs="+", default=None, help="Default: first two heads"
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[8, 16, 32], help="Rank budgets to sweep"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
