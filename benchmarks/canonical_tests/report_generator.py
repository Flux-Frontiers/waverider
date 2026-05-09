#!/usr/bin/env python3
"""
Benchmark Run Report Generator
===============================

Generates comprehensive PDF and Markdown summary reports from benchmark
JSON results files.  Embeds the result figure, all provenance, manifold
discovery findings, architecture comparison tables, and key findings.

Supports all canonical benchmark datasets: MNIST, CIFAR-10, CIFAR-100,
Digits, Iris, ResNet manifold, and Tiny ImageNet.

Usage
-----
    python report_generator.py results.json [--figure results.png] [--out dir]
    python report_generator.py --all          # regenerate every known result

Part of WaveRider, https://github.com/Flux-Frontiers/waverider
Author: Eric G. Suchanek, PhD
Affiliation: Flux-Frontiers
Date: 2026-03-29
"""

import argparse
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Machine identity
# ---------------------------------------------------------------------------

MACHINE_DESCRIPTION = "Apple M5 Max MacBook Pro, 64 GB RAM, 2TB SSD"

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git_info(repo_root):
    """Return (short_hash, branch, commit_date, commit_msg) or placeholders."""
    try:

        def _run(cmd):
            return (
                subprocess.check_output(cmd, cwd=repo_root, stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )

        short_hash = _run(["git", "rev-parse", "--short", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-re", "HEAD"])
        commit_date = _run(["git", "log", "-1", "--format=%ai"])
        commit_msg = _run(["git", "log", "-1", "--format=%s"])
        return short_hash, branch, commit_date, commit_msg
    except Exception:
        return "unknown", "unknown", "unknown", "unknown"


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


def _aggregate(results_dict):
    """Return {name: {mean_acc, std_acc, mean_loss, std_loss, n_params, mean_time}} ."""
    agg = {}
    for name, trials in results_dict.items():
        accs = [t["test_acc"] for t in trials]
        losses = [t["test_loss"] for t in trials if t.get("test_loss") is not None]
        times = [t["wall_time"] for t in trials]
        n_params = trials[0]["n_params"]
        mean_acc = float(np.mean(accs))
        agg[name] = {
            "mean_acc": mean_acc,
            "std_acc": float(np.std(accs)),
            "mean_loss": float(np.mean(losses)) if losses else float("nan"),
            "std_loss": float(np.std(losses)) if losses else float("nan"),
            "mean_time": float(np.mean(times)),
            "n_params": n_params,
            "eff": mean_acc / n_params * 1000 if n_params > 0 else float("nan"),
            "n_trials": len(trials),
        }
    return agg


def _winner(agg):
    return max(agg, key=lambda n: agg[n]["mean_acc"])


def _standard_name(agg):
    for n in agg:
        if n.lower().startswith("standard"):
            return n
    return None


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _write_markdown(data, agg, git_info, report_dir, figure_path):
    short_hash, branch, commit_date, commit_msg = git_info
    dataset = data.get("dataset", "unknown").upper()
    input_dim = data.get("input_dim", "?")
    n_classes = data.get("n_classes", "?")
    idim = data.get("intrinsic_dim", data.get("global_dim", "?"))
    tau = data.get("tau", "?")
    epochs = data.get("epochs", "?")
    trials = data.get("trials", "?")
    device = data.get("device", {})
    tf_ver = device.get("tensorflow_version", "?")
    dev_used = device.get("device_used", "?")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(f"# Manifold-Informed Architecture Benchmark — {dataset}")
    lines.append(f"\n**Generated:** {generated}  ")
    lines.append(f"**Machine:** {MACHINE_DESCRIPTION}  ")
    lines.append(f"**Repository:** waverider @ `{short_hash}` ({branch})  ")
    lines.append(f"**Commit:** {commit_date} — {commit_msg}  ")
    lines.append(
        f"**Python:** {platform.python_version()}  |  "
        f"**TensorFlow:** {tf_ver}  |  **Device:** {dev_used}  "
    )
    lines.append(f"**Host:** {socket.gethostname()}  |  **OS:** {platform.platform()}")

    lines.append("\n---\n")
    lines.append("## Experimental Setup\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Dataset | {dataset} |")
    lines.append(f"| Input dimensionality | {input_dim:,} |")
    lines.append(f"| Classes | {n_classes} |")
    lines.append(f"| Intrinsic dim (d) | {idim} |")
    lines.append(f"| Variance threshold (τ) | {tau} |")
    lines.append(f"| Epochs | {epochs} |")
    lines.append(f"| Trials | {trials} |")
    if data.get("batch_size"):
        lines.append(f"| Batch size | {data['batch_size']} |")
    if data.get("lr"):
        lines.append(f"| Learning rate | {data['lr']} |")

    # Manifold discovery
    dim_report = data.get("dimensionality_report", {})
    if dim_report:
        lines.append("\n## Manifold Discovery\n")
        k_str = str(data["k_pca"]) if data.get("k_pca") else "not recorded"
        lines.append(f"Local PCA over the training set, k={k_str} neighbors.\n")
        lines.append("| τ | Mean d | Std | Min | Max | Noise % |")
        lines.append("|---|---|---|---|---|---|")
        for tau_key in sorted(dim_report.keys(), reverse=True):
            r = dim_report[tau_key]
            noise = (
                100.0 * (1.0 - r["mean"] / input_dim)
                if isinstance(input_dim, int)
                else float("nan")
            )
            lines.append(
                f"| {float(tau_key):.2f} | {r['mean']:.1f} | {r['std']:.1f} "
                f"| {r['min']} | {r['max']} | {noise:.1f}% |"
            )

    # Per-class dims (show top/bottom 5 by mean dim, or all if ≤ 20 classes)
    pcd = data.get("per_class_dims", {})
    class_names = data.get("class_names", {})
    if pcd:
        lines.append("\n### Per-Class Intrinsic Dimensionality\n")
        entries = []
        for k, v in pcd.items():
            if class_names and int(k) < len(class_names):
                label = class_names[int(k)]
            elif data.get("dataset", "").lower() in ("mnist", "digits"):
                label = f"Digit {k}"
            else:
                label = str(k)
            entries.append((label, v["mean"], v["std"], v["min"], v["max"]))
        entries.sort(key=lambda x: x[1], reverse=True)

        if len(entries) > 20:
            show = entries[:10] + [("...", None, None, None, None)] + entries[-10:]
            lines.append("*Showing 10 hardest + 10 easiest classes (sorted by mean d)*\n")
        else:
            show = entries

        lines.append("| Class | Mean d | Std | Min | Max |")
        lines.append("|---|---|---|---|---|")
        for e in show:
            if e[1] is None:
                lines.append("| … | … | … | … | … |")
            else:
                lines.append(f"| {e[0]} | {e[1]:.1f} | {e[2]:.1f} | {e[3]} | {e[4]} |")

    # Architecture results
    lines.append("\n## Architecture Comparison\n")
    lines.append("| Architecture | Params | Test Acc (mean ± std) | Test Loss | Acc/Kparam |")
    lines.append("|---|---|---|---|---|")

    std_name = _standard_name(agg)
    win_name = _winner(agg)
    for name, m in agg.items():
        tag = " ✦" if name == win_name and name != std_name else ""
        loss = "N/A" if np.isnan(m["mean_loss"]) else f"{m['mean_loss']:.4f}"
        eff = "N/A" if np.isnan(m["eff"]) else f"{m['eff']:.4f}"
        lines.append(
            f"| {name}{tag} | {m['n_params']:,} | "
            f"{m['mean_acc']:.4f} ± {m['std_acc']:.4f} | {loss} | {eff} |"
        )

    # Key findings
    lines.append("\n## Key Findings\n")
    win = agg[win_name]
    lines.append(f"- **Best architecture:** {win_name}")
    lines.append(f"  — test accuracy {win['mean_acc']:.4f} ± {win['std_acc']:.4f}")
    if std_name and win_name != std_name:
        std = agg[std_name]
        delta_acc = win["mean_acc"] - std["mean_acc"]
        param_ratio = std["n_params"] / win["n_params"] if win["n_params"] > 0 else float("inf")
        lines.append(
            f"- **vs Standard:** +{delta_acc:.4f} ({delta_acc * 100:.2f} pp) accuracy gain"
        )
        lines.append(
            f"- **Parameter reduction:** {param_ratio:.1f}× fewer parameters "
            f"({win['n_params']:,} vs {std['n_params']:,})"
        )
        lines.append(
            f"- **Parameter efficiency:** {win['eff']:.4f} acc/Kparam "
            f"vs {std['eff']:.4f} for Standard "
            f"({win['eff'] / std['eff']:.1f}× improvement)"
        )
    noise = (
        100.0 * (1.0 - idim / input_dim)
        if isinstance(idim, int) and isinstance(input_dim, int)
        else None
    )
    if noise is not None:
        lines.append(
            f"- **Manifold compression:** {input_dim:,}D → {idim}D "
            f"({noise:.1f}% of ambient dimensions are noise)"
        )

    # Figure
    if figure_path and Path(figure_path).exists():
        rel = Path(figure_path).name
        lines.append("\n## Result Figure\n")
        lines.append(f"![{dataset} Results]({rel})")

    md_path = Path(report_dir) / f"{data.get('dataset', 'benchmark')}_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md_path


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------


def _write_pdf(data, agg, git_info, report_dir, figure_path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib not available — skipping PDF")
        return None

    short_hash, branch, commit_date, commit_msg = git_info
    dataset = data.get("dataset", "unknown").upper()
    input_dim = data.get("input_dim", "?")
    n_classes = data.get("n_classes", "?")
    idim = data.get("intrinsic_dim", data.get("global_dim", "?"))
    tau = data.get("tau", "?")
    epochs = data.get("epochs", "?")
    trials = data.get("trials", "?")
    device = data.get("device", {})
    tf_ver = device.get("tensorflow_version", "?")
    dev_used = device.get("device_used", "?")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf_path = Path(report_dir) / f"{data.get('dataset', 'benchmark')}_report.pdf"

    noise_str = ""
    if isinstance(idim, int) and isinstance(input_dim, int):
        noise_pct = 100.0 * (1.0 - idim / input_dim)
        noise_str = f"  ({noise_pct:.1f}% of ambient dims are noise)"

    std_name = _standard_name(agg)
    win_name = _winner(agg)

    MONO = "DejaVu Sans Mono"
    SERIF = "DejaVu Sans"

    def new_fig():
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("#FAFAFA")
        return fig

    def header_bar(fig, title, subtitle=""):
        fig.add_axes([0, 0.93, 1, 0.07]).set_axis_off()
        ax = fig.axes[-1]
        ax.set_facecolor("#1a3a5c")
        ax.text(
            0.5,
            0.65,
            title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="white",
            transform=ax.transAxes,
            fontfamily=SERIF,
        )
        if subtitle:
            ax.text(
                0.5,
                0.15,
                subtitle,
                ha="center",
                va="center",
                fontsize=9,
                color="#add8e6",
                transform=ax.transAxes,
                fontfamily=SERIF,
            )

    def text_block(fig, lines, y_start=0.88, x=0.07, fontsize=8.5, dy=0.022):
        for line in lines:
            bold = line.startswith("##")
            clean = line.lstrip("# ").lstrip("**").rstrip("**")
            fs = fontsize + (2 if bold else 0)
            fw = "bold" if bold else "normal"
            col = "#1a3a5c" if bold else "#222222"
            fig.text(
                x,
                y_start,
                clean,
                fontsize=fs,
                fontweight=fw,
                color=col,
                fontfamily=MONO if not bold else SERIF,
                va="top",
            )
            y_start -= dy
            if y_start < 0.04:
                break
        return y_start

    with PdfPages(pdf_path) as pdf:
        # ------------------------------------------------------------------
        # Page 1 — Title + Executive Summary
        # ------------------------------------------------------------------
        fig = new_fig()
        header_bar(
            fig,
            "Manifold-Informed Architecture Benchmark",
            f"Dataset: {dataset}  |  Generated: {generated}",
        )

        win = agg[win_name]
        std = agg.get(std_name, win)

        summary_lines = [
            "## Executive Summary",
            "",
            f"Dataset           :  {dataset}",
            f"Input dimension   :  {input_dim:,}D  (ambient pixel space)",
            f"Classes           :  {n_classes}",
            f"Intrinsic dim (d) :  {idim}{noise_str}",
            f"Variance threshold:  τ = {tau}",
            "",
            "## Training Protocol",
            "",
            f"Epochs            :  {epochs}",
            f"Trials            :  {trials}",
            *([f"Batch size        :  {data['batch_size']}"] if data.get("batch_size") else []),
            *([f"Learning rate     :  {data['lr']}"] if data.get("lr") else []),
            "Optimizer         :  Adam",
            "",
            "## Best Architecture",
            "",
            f"  {win_name}",
            f"  Accuracy  :  {win['mean_acc']:.4f} ± {win['std_acc']:.4f}",
            f"  Params    :  {win['n_params']:,}",
            f"  Eff       :  {win['eff']:.4f} acc / Kparam",
        ]
        if std_name and win_name != std_name:
            delta = win["mean_acc"] - std["mean_acc"]
            ratio = std["n_params"] / win["n_params"] if win["n_params"] > 0 else float("inf")
            summary_lines += [
                "",
                "## vs Standard Architecture",
                "",
                f"  Accuracy delta    :  +{delta:.4f}  (+{delta * 100:.2f} pp)",
                f"  Parameter ratio   :  {ratio:.1f}x fewer parameters",
                f"  Efficiency gain   :  {win['eff'] / std['eff']:.1f}x acc/Kparam",
            ]

        text_block(fig, summary_lines, y_start=0.89)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 2 — Provenance
        # ------------------------------------------------------------------
        fig = new_fig()
        header_bar(fig, "Provenance & Environment")

        prov_lines = [
            "## Repository",
            "",
            "  Project  :  waverider",
            "  URL      :  https://github.com/Flux-Frontiers/waverider",
            f"  Commit   :  {short_hash}  ({branch})",
            f"  Date     :  {commit_date}",
            f"  Message  :  {commit_msg}",
            "",
            "## Software",
            "",
            f"  Python       :  {platform.python_version()}",
            f"  TensorFlow   :  {tf_ver}",
            f"  Platform     :  {platform.platform()}",
            f"  Host         :  {socket.gethostname()}",
            "",
            "## Hardware",
            "",
            f"  Machine      :  {MACHINE_DESCRIPTION}",
            f"  Compute      :  {dev_used}",
        ]
        phys = device.get("physical_devices", [])
        if phys:
            for p in phys:
                prov_lines.append(f"  Device       :  {p}")

        prov_lines += [
            "",
            "## Manifold Discovery Parameters",
            "",
            f"  k neighbors (local PCA)  :  {data.get('k_pca', '?')}",
            f"  Discovery samples        :  {data.get('discovery_samples', '?')}",
            f"  Samples per class        :  {data.get('samples_per_class', '?')}",
            f"  Variance threshold τ     :  {tau}",
            "",
            "## Results Files",
            "",
            f"  JSON   :  {data.get('dataset', '?')}_architecture_results.json",
            f"  Figure :  {data.get('dataset', '?')}_architecture_results.png",
            f"  Report :  {data.get('dataset', '?')}_report.pdf",
            f"  Report :  {data.get('dataset', '?')}_report.md",
            "",
            f"  Report generated  :  {generated}",
        ]

        text_block(fig, prov_lines, y_start=0.89)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 3 — Manifold Discovery Results
        # ------------------------------------------------------------------
        dim_report = data.get("dimensionality_report", {})
        if dim_report:
            fig = new_fig()
            header_bar(
                fig,
                "Manifold Discovery",
                f"Local PCA  |  k={data.get('k_pca', '?')}  |  τ={tau}",
            )

            disc_lines = ["## Intrinsic Dimensionality vs Variance Threshold", ""]
            disc_lines.append(
                f"  {'τ':>5}  {'Mean d':>8}  {'Std':>6}  {'Min':>5}  {'Max':>5}  {'Noise %':>8}"
            )
            disc_lines.append(f"  {'-' * 47}")
            for tau_key in sorted(dim_report.keys(), reverse=True):
                r = dim_report[tau_key]
                noise = (
                    100.0 * (1.0 - r["mean"] / input_dim)
                    if isinstance(input_dim, int)
                    else float("nan")
                )
                disc_lines.append(
                    f"  {float(tau_key):>5.2f}  {r['mean']:>8.1f}  {r['std']:>6.1f}  "
                    f"{r['min']:>5}  {r['max']:>5}  {noise:>7.1f}%"
                )

            disc_lines += [
                "",
                f"  Selected d = {idim}  (max per-class max at τ={tau}, "
                f"clamped to n_classes={n_classes})",
                "",
            ]

            pcd = data.get("per_class_dims", {})
            class_names = data.get("class_names", [])
            if pcd:
                disc_lines += ["## Per-Class Intrinsic Dimensionality", ""]
                entries = []
                for k, v in pcd.items():
                    if class_names and int(k) < len(class_names):
                        label = class_names[int(k)]
                    elif data.get("dataset", "").lower() in ("mnist", "digits"):
                        label = f"Digit {k}"
                    else:
                        label = str(k)
                    entries.append((label, v["mean"], v["std"], v["min"], v["max"]))
                entries.sort(key=lambda x: x[1], reverse=True)

                show = entries if len(entries) <= 20 else entries[:10] + entries[-10:]
                if len(entries) > 20:
                    disc_lines.append(f"  (Top 10 hardest + 10 easiest of {len(entries)} classes)")
                    disc_lines.append("")

                disc_lines.append(
                    f"  {'Class':>22}  {'Mean d':>7}  {'Std':>5}  {'Min':>4}  {'Max':>4}"
                )
                disc_lines.append(f"  {'-' * 48}")
                for i, e in enumerate(show):
                    if len(entries) > 20 and i == 10:
                        disc_lines.append(f"  {'...':>22}  {'...':>7}")
                    disc_lines.append(
                        f"  {e[0]:>22}  {e[1]:>7.1f}  {e[2]:>5.1f}  {e[3]:>4}  {e[4]:>4}"
                    )

            text_block(fig, disc_lines, y_start=0.89, fontsize=8)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Page 4 — Result Figure
        # ------------------------------------------------------------------
        if figure_path and Path(figure_path).exists():
            img = plt.imread(figure_path)
            fig = plt.figure(figsize=(11, 8.5))  # landscape for figure
            fig.patch.set_facecolor("#FAFAFA")
            ax = fig.add_axes([0.02, 0.02, 0.96, 0.93])
            ax.imshow(img)
            ax.axis("off")
            fig.text(
                0.5,
                0.985,
                f"{dataset} — Architecture Comparison Results",
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="#1a3a5c",
                fontfamily=SERIF,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Page 5 — Architecture Comparison Table
        # ------------------------------------------------------------------
        fig = new_fig()
        header_bar(
            fig,
            "Architecture Comparison",
            f"{dataset}  |  {epochs} epochs  |  {trials} trials  |  Adam optimizer",
        )

        col_w = [32, 11, 22, 10, 11]
        h_fmt = (
            f"  {'Architecture':<{col_w[0]}}  {'Params':>{col_w[1]}}  "
            f"{'Test Acc (mean±std)':>{col_w[2]}}  "
            f"{'Loss':>{col_w[3]}}  {'Acc/Kparam':>{col_w[4]}}"
        )
        sep = "  " + "-" * (sum(col_w) + 2 * len(col_w))

        tbl_lines = ["## Results Table", "", h_fmt, sep]

        for name, m in agg.items():
            marker = " ✦" if name == win_name else "  "
            short = name if len(name) <= col_w[0] else name[: col_w[0] - 1] + "…"
            tbl_lines.append(
                f"{marker}{short:<{col_w[0]}}  {m['n_params']:>{col_w[1]},}  "
                f"{m['mean_acc']:.4f} ± {m['std_acc']:.4f}  "
                f"{m['mean_loss']:>{col_w[3]}.4f}  {m['eff']:>{col_w[4]}.4f}"
            )

        tbl_lines += ["", "  ✦ = best performing architecture", ""]

        # Parameter efficiency ranking
        tbl_lines += ["## Parameter Efficiency Ranking  (acc / Kparam)", ""]
        ranked = sorted(agg.items(), key=lambda x: x[1]["eff"], reverse=True)
        for rank, (name, m) in enumerate(ranked, 1):
            tbl_lines.append(
                f"  {rank}. {name:<{col_w[0]}}  {m['eff']:.4f}  "
                f"({m['mean_acc']:.4f} / {m['n_params']:,})"
            )

        if std_name and win_name != std_name:
            win = agg[win_name]
            std = agg[std_name]
            delta = win["mean_acc"] - std["mean_acc"]
            ratio = std["n_params"] / win["n_params"] if win["n_params"] > 0 else float("inf")
            tbl_lines += [
                "",
                "## Summary vs Standard",
                "",
                f"  Best architecture  :  {win_name}",
                f"  Accuracy delta     :  +{delta:.4f}  (+{delta * 100:.2f} pp)",
                f"  Parameter ratio    :  {ratio:.1f}x fewer parameters",
                f"  Efficiency gain    :  {win['eff'] / std['eff']:.1f}x acc / Kparam",
            ]

        text_block(fig, tbl_lines, y_start=0.89, fontsize=8.2)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ------------------------------------------------------------------
        # Page 6 — Key Findings
        # ------------------------------------------------------------------
        fig = new_fig()
        header_bar(fig, "Key Findings & Interpretation")

        win = agg[win_name]
        findings = ["## Manifold Hypothesis Test Results", ""]

        if isinstance(idim, int) and isinstance(input_dim, int):
            noise_pct = 100.0 * (1.0 - idim / input_dim)
            findings += [
                f"  The {dataset} training set occupies a {idim}-dimensional manifold",
                f"  embedded in {input_dim:,}-dimensional pixel space.",
                f"  {noise_pct:.1f}% of ambient dimensions carry no structure.",
                "",
            ]

        if std_name and win_name != std_name:
            std = agg[std_name]
            delta = win["mean_acc"] - std["mean_acc"]
            ratio = std["n_params"] / win["n_params"] if win["n_params"] > 0 else float("inf")
            eff_gain = win["eff"] / std["eff"]
            findings += [
                "## Architecture Performance",
                "",
                f"  A manifold-informed architecture ({win_name})",
                f"  achieves {win['mean_acc']:.4f} test accuracy — {delta * 100:.2f} pp better",
                f"  than the Standard baseline ({std['mean_acc']:.4f}),",
                f"  using {ratio:.1f}x fewer parameters ({win['n_params']:,} vs {std['n_params']:,}).",
                "",
                f"  Parameter efficiency improves {eff_gain:.1f}x over Standard.",
                "",
            ]
        else:
            findings += [
                "## Architecture Performance",
                "",
                f"  Best architecture: {win_name}",
                f"  Test accuracy: {win['mean_acc']:.4f} ± {win['std_acc']:.4f}",
                "",
            ]

        # Compression insight for high-noise datasets
        if isinstance(idim, int) and isinstance(input_dim, int) and idim < input_dim // 10:
            findings += [
                "## PCA Pre-Projection Insight",
                "",
                f"  When ambient noise exceeds 90%, PCA pre-projection to d={idim}",
                "  removes uninformative variance before the network trains.",
                f"  Networks operating on raw {input_dim:,}D input must learn this",
                "  projection implicitly, wasting capacity on noise dimensions.",
                "",
            ]

        findings += [
            "## Interpretation",
            "",
            "  These results support the manifold hypothesis for classification:",
            "  sizing network bottlenecks to the intrinsic dimensionality of the",
            "  data manifold — rather than the ambient input dimension — yields",
            "  equal or better accuracy with substantially fewer parameters.",
            "",
            "  The parameter efficiency advantage scales with the ambient-to-",
            "  intrinsic dimensionality ratio (compression factor).",
        ]

        text_block(fig, findings, y_start=0.89)

        # Stamp
        fig.text(
            0.5,
            0.015,
            f"waverider @ {short_hash}  |  {generated}  |  {MACHINE_DESCRIPTION}",
            ha="center",
            fontsize=7,
            color="#888888",
            fontfamily=MONO,
        )

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = f"Manifold Architecture Benchmark — {dataset}"
        d["Author"] = "Eric G. Suchanek, PhD  |  Flux-Frontiers"
        d["Subject"] = "Manifold-informed neural architecture comparison"
        d["Keywords"] = (
            "manifold learning, intrinsic dimensionality, neural architecture, machine learning"
        )
        d["Creator"] = f"waverider report_generator @ {short_hash}"

    return pdf_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------


def _tex_escape(s):
    """Escape special LaTeX characters and map Unicode math to LaTeX commands.

    :param s: Input string (will be str-coerced).
    :returns: LaTeX-safe string.
    """
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    for ch, rep in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("^", r"\^{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("_", r"\_"),
    ]:
        s = s.replace(ch, rep)
    for ch, rep in [
        ("→", r"$\rightarrow$"),
        ("←", r"$\leftarrow$"),
        ("±", r"$\pm$"),
        ("×", r"$\times$"),
        ("≥", r"$\geq$"),
        ("≤", r"$\leq$"),
        ("τ", r"$\tau$"),
        ("✦", r"$\star$"),
    ]:
        s = s.replace(ch, rep)
    return s


def _write_tex(data, agg, git_info, report_dir, figure_path):
    """Generate a LaTeX source file for the benchmark report.

    Writes a self-contained .tex compilable with ``latexmk -pdf`` or
    ``pdflatex``.  Required LaTeX packages (all in texlive-latex-recommended
    or texlive-latex-extra): geometry, booktabs, graphicx, hyperref,
    microtype, xcolor, longtable, fancyhdr, titlesec, lmodern, enumitem.

    :param data: Parsed JSON benchmark data dict.
    :param agg: Aggregated metrics dict from :func:`_aggregate`.
    :param git_info: (short_hash, branch, commit_date, commit_msg) tuple.
    :param report_dir: Directory to write the .tex file into.
    :param figure_path: Optional path to the result PNG figure.
    :returns: Path to the written .tex file.
    """
    short_hash, branch, commit_date, commit_msg = git_info
    dataset = data.get("dataset", "unknown").upper()
    input_dim = data.get("input_dim", "?")
    n_classes = data.get("n_classes", "?")
    idim = data.get("intrinsic_dim", data.get("global_dim", "?"))
    tau = data.get("tau", "?")
    epochs = data.get("epochs", "?")
    trials = data.get("trials", "?")
    device = data.get("device", {})
    tf_ver = device.get("tensorflow_version", "?")
    dev_used = device.get("device_used", "?")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    std_name = _standard_name(agg)
    win_name = _winner(agg)
    win = agg[win_name]

    e = _tex_escape  # shorthand

    noise_pct_str = ""
    if isinstance(idim, int) and isinstance(input_dim, int):
        noise_pct_val = 100.0 * (1.0 - idim / input_dim)
        noise_pct_str = f"{noise_pct_val:.1f}\\%"

    # figure directory for \graphicspath (absolute so compilation works from any cwd)
    fig_dir = str(Path(figure_path).parent) + "/" if figure_path else ""

    L = []  # accumulate tex lines

    # ---------------------------------------------------------------- preamble
    L += [
        r"\documentclass[11pt,letterpaper]{article}",
        r"\usepackage[margin=2.5cm,top=3.5cm,bottom=3cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{microtype}",
        r"\usepackage[table,dvipsnames]{xcolor}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{caption}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{amsmath}",
        r"\usepackage{titlesec}",
        r"\usepackage{enumitem}",
        r"",
        r"\definecolor{wavblue}{RGB}{26,58,92}",
        r"\definecolor{wavgray}{RGB}{120,120,120}",
        r"\definecolor{wavlight}{RGB}{235,242,250}",
        r"\definecolor{wincolor}{RGB}{0,110,0}",
        r"",
    ]
    if fig_dir:
        L.append(rf"\graphicspath{{{{{fig_dir}}}}}")
    L += [
        r"",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\fancyhead[L]{{\small\color{{wavgray}}Manifold-Informed Architecture Benchmark"
        rf" \textemdash\ {e(dataset)}}}",
        rf"\fancyhead[R]{{\small\color{{wavgray}}\texttt{{waverider @ {e(short_hash)}}}}}",
        r"\fancyfoot[C]{\small\color{wavgray}\thepage}",
        r"\renewcommand{\headrulewidth}{0.4pt}",
        r"",
        r"\titleformat{\section}{\large\bfseries\color{wavblue}}{}{0em}{}[\color{wavblue}\titlerule]",
        r"\titleformat{\subsection}{\normalsize\bfseries\color{wavblue}}{}{0em}{}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt plus 1pt}",
        r"",
        r"\begin{document}",
        r"",
    ]

    # ---------------------------------------------------------------- title block
    L += [
        r"\begin{center}",
        r"  {\LARGE\bfseries\color{wavblue} Manifold-Informed Architecture Benchmark}\\[6pt]",
        rf"  {{\large {e(dataset)}: {n_classes} classes,"
        rf" {input_dim:,}D input}}\\[8pt]"
        if isinstance(input_dim, int)
        else rf"  {{\large {e(dataset)}: {n_classes} classes}}\\[8pt]",
        rf"  {{\small Generated: {e(generated)}"
        rf" \quad$|$\quad {e(MACHINE_DESCRIPTION)}}}\\[2pt]",
        rf"  {{\small\texttt{{waverider @ {e(short_hash)}}} ({e(branch)})"
        rf" \quad$|$\quad {e(commit_date)}}}",
        r"\end{center}",
        r"{\color{wavblue}\noindent\rule{\linewidth}{1.5pt}}",
        r"\medskip",
        r"",
    ]

    # ---------------------------------------------------------------- summary box
    summary_items = [
        rf"\textbf{{Best:}} {e(win_name)}"
        rf"\quad test acc $= {win['mean_acc']:.4f} \pm {win['std_acc']:.4f}$",
    ]
    if std_name and win_name != std_name:
        std = agg[std_name]
        delta = win["mean_acc"] - std["mean_acc"]
        ratio = std["n_params"] / win["n_params"] if win["n_params"] > 0 else float("inf")
        eff_gain = win["eff"] / std["eff"]
        summary_items.append(
            rf"\textbf{{vs Standard:}} $+{delta:.4f}$ ({delta * 100:.2f}\,pp)"
            rf"\quad {ratio:.1f}$\times$ fewer params"
            rf"\quad {eff_gain:.1f}$\times$ acc/Kparam"
        )
    if noise_pct_str and isinstance(idim, int) and isinstance(input_dim, int):
        summary_items.append(
            rf"\textbf{{Manifold:}} {input_dim:,}D $\rightarrow$ {idim}D"
            rf" ({noise_pct_str}\ noise dimensions)"
        )

    L.append(r"\noindent\colorbox{wavlight}{\parbox{\dimexpr\linewidth-2\fboxsep\relax}{")
    L.append(r"  \smallskip")
    for item in summary_items:
        L.append(f"  {item}\\\\[2pt]")
    L.append(r"  \smallskip")
    L.append(r"}}")
    L.append(r"\bigskip")
    L.append(r"")

    # ---------------------------------------------------------------- experimental setup
    L += [
        r"\section{Experimental Setup}",
        r"",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Value} \\",
        r"\midrule",
    ]
    setup_rows = [("Dataset", dataset)]
    if isinstance(input_dim, int):
        setup_rows.append(("Input dimensionality", f"{input_dim:,}D"))
    setup_rows += [
        ("Classes", str(n_classes)),
        (r"Intrinsic dim ($d$)", str(idim)),
        (r"Variance threshold ($\tau$)", str(tau)),
        ("Epochs", str(epochs)),
        ("Trials", str(trials)),
    ]
    for key, label in [
        ("batch_size", "Batch size"),
        ("lr", "Learning rate"),
        ("k_pca", r"$k$ neighbors (local PCA)"),
        ("discovery_samples", "Discovery samples"),
        ("samples_per_class", "Samples per class"),
    ]:
        if data.get(key):
            setup_rows.append((label, str(data[key])))
    for label, val in setup_rows:
        L.append(f"  {label} & {e(val)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r""]

    # ---------------------------------------------------------------- manifold discovery
    dim_report = data.get("dimensionality_report", {})
    if dim_report:
        k_str = str(data.get("k_pca", "?"))
        L += [
            r"\section{Manifold Discovery}",
            r"",
            rf"Local PCA over the training set, $k = {k_str}$ neighbors."
            rf" Intrinsic dimensionality $d$ is taken as the maximum per-class maximum at"
            rf" $\tau = {tau}$, clamped to $n_{{\mathrm{{classes}}}} = {n_classes}$.",
            r"",
            r"\begin{tabular}{@{}rrrrrr@{}}",
            r"\toprule",
            r"$\tau$ & Mean $d$ & Std & Min & Max & Noise (\%) \\",
            r"\midrule",
        ]
        for tau_key in sorted(dim_report.keys(), reverse=True):
            r = dim_report[tau_key]
            noise = (
                100.0 * (1.0 - r["mean"] / input_dim)
                if isinstance(input_dim, int)
                else float("nan")
            )
            L.append(
                f"  {float(tau_key):.2f} & {r['mean']:.1f} & {r['std']:.1f}"
                f" & {r['min']} & {r['max']} & {noise:.1f} \\\\"
            )
        L += [r"\bottomrule", r"\end{tabular}", r""]

        pcd = data.get("per_class_dims", {})
        class_names_list = data.get("class_names", [])
        if pcd:
            entries = []
            for k, v in pcd.items():
                if class_names_list and int(k) < len(class_names_list):
                    label = class_names_list[int(k)]
                elif data.get("dataset", "").lower() in ("mnist", "digits"):
                    label = f"Digit {k}"
                else:
                    label = str(k)
                entries.append((label, v["mean"], v["std"], v["min"], v["max"]))
            entries.sort(key=lambda x: x[1], reverse=True)

            show = entries if len(entries) <= 20 else entries[:10] + entries[-10:]
            tbl_env = "longtable" if len(show) > 30 else "tabular"

            L.append(r"\subsection{Per-Class Intrinsic Dimensionality}")
            L.append(r"")
            if len(entries) > 20:
                L.append(
                    rf"Showing the 10 highest- and 10 lowest-dimensional"
                    rf" classes of {len(entries)} total, sorted by mean $d$."
                )
                L.append(r"")
            L.append(rf"\begin{{{tbl_env}}}{{@{{}}lrrrr@{{}}}}")
            L.append(r"\toprule")
            L.append(r"Class & Mean $d$ & Std & Min & Max \\")
            L.append(r"\midrule")
            if tbl_env == "longtable":
                L.append(r"\endhead")
            for i, entry in enumerate(show):
                if len(entries) > 20 and i == 10:
                    L.append(r"  \multicolumn{5}{c}{\ldots} \\")
                L.append(
                    f"  {e(entry[0])} & {entry[1]:.1f} & {entry[2]:.1f}"
                    f" & {entry[3]} & {entry[4]} \\\\"
                )
            L += [r"\bottomrule", rf"\end{{{tbl_env}}}", r""]

    # ---------------------------------------------------------------- architecture comparison
    L += [
        r"\section{Architecture Comparison}",
        r"",
        r"{\small",
        r"\begin{tabular}{@{}p{7cm}rccr@{}}",
        r"\toprule",
        r"\textbf{Architecture} & \textbf{Params}"
        r" & \textbf{Test Acc ($\mu \pm \sigma$)} & \textbf{Loss} & \textbf{Acc/Kparam} \\",
        r"\midrule",
    ]
    for name, m in agg.items():
        star = r"\,$\star$" if name == win_name else ""
        color_cmd = r"\color{wincolor}" if name == win_name else ""
        loss_str = "---" if np.isnan(m["mean_loss"]) else f"{m['mean_loss']:.4f}"
        eff_str = "---" if np.isnan(m["eff"]) else f"{m['eff']:.4f}"
        L.append(
            f"  {{{color_cmd}{e(name)}{star}}} & {m['n_params']:,}"
            f" & ${m['mean_acc']:.4f} \\pm {m['std_acc']:.4f}$"
            f" & {loss_str} & {eff_str} \\\\"
        )
    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",  # end \small
        r"",
        r"\smallskip",
        r"{\small $\star$ = best performing architecture}",
        r"",
    ]

    # ---------------------------------------------------------------- key findings
    L += [r"\section{Key Findings}", r"", r"\begin{itemize}[leftmargin=*,itemsep=3pt]"]
    L.append(
        rf"  \item \textbf{{Best architecture:}} {e(win_name)}"
        rf" \textemdash\ test accuracy ${win['mean_acc']:.4f} \pm {win['std_acc']:.4f}$"
    )
    if std_name and win_name != std_name:
        std = agg[std_name]
        delta = win["mean_acc"] - std["mean_acc"]
        ratio = std["n_params"] / win["n_params"] if win["n_params"] > 0 else float("inf")
        eff_gain = win["eff"] / std["eff"]
        L.append(
            rf"  \item \textbf{{vs Standard:}} $+{delta:.4f}$ ({delta * 100:.2f}\,pp)"
            rf" accuracy gain"
        )
        L.append(
            rf"  \item \textbf{{Parameter reduction:}} {ratio:.1f}$\times$ fewer parameters"
            rf" ({win['n_params']:,} vs {std['n_params']:,})"
        )
        L.append(
            rf"  \item \textbf{{Parameter efficiency:}} {eff_gain:.1f}$\times$ improvement"
            rf" ({win['eff']:.4f} vs {std['eff']:.4f} acc/Kparam)"
        )
    if noise_pct_str and isinstance(idim, int) and isinstance(input_dim, int):
        L.append(
            rf"  \item \textbf{{Manifold compression:}} {input_dim:,}D $\rightarrow$ {idim}D"
            rf" ({noise_pct_str}\ of ambient dimensions carry no structure)"
        )
    L += [r"\end{itemize}", r""]

    # ---------------------------------------------------------------- result figure
    if figure_path and Path(figure_path).exists():
        fig_stem = Path(figure_path).stem
        L += [
            r"\clearpage",
            r"\section{Result Figure}",
            r"",
            r"\begin{figure}[htbp]",
            r"\centering",
            rf"\includegraphics[width=\textwidth]{{{fig_stem}}}",
            rf"\caption{{{e(dataset)} manifold-informed vs.\ standard architecture comparison."
            rf" $d = {idim}$, {epochs} epochs, {trials} trials.}}",
            r"\end{figure}",
            r"",
        ]

    # ---------------------------------------------------------------- provenance
    L += [
        r"\section{Provenance}",
        r"",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Value} \\",
        r"\midrule",
    ]
    prov_rows = [
        (
            "Repository",
            r"\href{https://github.com/Flux-Frontiers/waverider}{Flux-Frontiers/waverider}",
        ),
        ("Commit", rf"\texttt{{{e(short_hash)}}} ({e(branch)})"),
        ("Commit date", e(commit_date)),
        ("Commit message", e(commit_msg)),
        ("Python", e(platform.python_version())),
        ("TensorFlow", e(tf_ver)),
        ("Device", e(dev_used)),
        ("Host", e(socket.gethostname())),
        ("OS", e(platform.platform())),
        ("Machine", e(MACHINE_DESCRIPTION)),
        ("Generated", e(generated)),
    ]
    for label, val in prov_rows:
        L.append(f"  {label} & {val} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"", r"\end{document}", r""]

    tex_path = Path(report_dir) / f"{data.get('dataset', 'benchmark')}_report.tex"
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return tex_path


def _compile_tex(tex_path):
    """Compile a .tex file to PDF using latexmk or pdflatex.

    :param tex_path: Path to the .tex source file.
    :returns: Path to the compiled PDF, or None if unavailable or failed.
    """
    import shutil

    tex_path = Path(tex_path)
    out_dir = tex_path.parent

    if shutil.which("latexmk"):
        cmd = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            f"-outdir={out_dir}",
            str(tex_path),
        ]
        compiler = "latexmk"
    elif shutil.which("pdflatex"):
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={out_dir}",
            str(tex_path),
        ]
        compiler = "pdflatex"
    else:
        print(
            "  No LaTeX compiler found (latexmk/pdflatex) — "
            ".tex written; compile with: latexmk -pdf <file>"
        )
        return None

    print(f"  Compiling with {compiler}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir, timeout=120)
        # pdflatex needs a second pass for cross-references
        if compiler == "pdflatex" and result.returncode == 0:
            subprocess.run(cmd, capture_output=True, cwd=out_dir, timeout=120)
        pdf = tex_path.with_suffix(".pdf")
        if result.returncode == 0 and pdf.exists():
            return pdf
        tail = (result.stdout + result.stderr)[-400:]
        print(f"  Compilation failed (rc={result.returncode}):\n{tail}")
    except subprocess.TimeoutExpired:
        print("  LaTeX compilation timed out after 120 s")
    return None


# ---------------------------------------------------------------------------


def generate_report(json_path, figure_path=None, out_dir=None):
    """Generate PDF + Markdown reports for a single benchmark run.

    :param json_path: Path to the benchmark results JSON file.
    :param figure_path: Optional path to the result PNG figure.
    :param out_dir: Directory for output files (defaults to JSON directory).
    :returns: (pdf_path, md_path) as Path objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    with open(json_path) as f:
        data = json.load(f)

    # Auto-locate figure if not given
    if figure_path is None:
        dataset = data.get("dataset", "")
        candidates = [
            json_path.parent / f"{dataset}_architecture_results.png",
            json_path.parent / f"{dataset}_manifold_architecture_results.png",
            json_path.with_suffix(".png"),
        ]
        for c in candidates:
            if c.exists():
                figure_path = c
                break

    report_dir = Path(out_dir) if out_dir else json_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    repo_root = json_path.parent
    git_info = _git_info(repo_root)
    agg = _aggregate(data["results"])

    md_path = _write_markdown(data, agg, git_info, report_dir, figure_path)
    tex_path = _write_tex(data, agg, git_info, report_dir, figure_path)
    pdf_path = _compile_tex(tex_path)
    if pdf_path is None:
        # Fall back to matplotlib PDF when no LaTeX compiler is available
        pdf_path = _write_pdf(data, agg, git_info, report_dir, figure_path)

    print(f"  Markdown : {md_path}")
    print(f"  LaTeX    : {tex_path}")
    if pdf_path:
        print(f"  PDF      : {pdf_path}")

    return pdf_path, md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

KNOWN_RESULTS = [
    "mnist_architecture_results.json",
    "cifar10_architecture_results.json",
    "cifar100_architecture_results.json",
    "digits_manifold_architecture_results.json",
    "iris_architecture_results.json",
    "cifar100_resnet_manifold_architecture_results.json",
    "resnet_manifold_architecture_results.json",
    "tiny_imagenet_architecture_results.json",
    "tiny_imagenet_gpu_architecture_results.json",
]

DISEASE_RESULTS = [
    "alzheimers_disease_architecture_results.json",
    "breast_cancer_disease_architecture_results.json",
    "heart_disease_architecture_results.json",
    "diabetes_disease_architecture_results.json",
    "parkinsons_disease_architecture_results.json",
    "dermatology_disease_architecture_results.json",
]


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark summary reports")
    parser.add_argument("json", nargs="?", help="Path to results JSON file")
    parser.add_argument("--figure", help="Path to result PNG figure (auto-detected if omitted)")
    parser.add_argument("--out", help="Output directory (defaults to JSON directory)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Regenerate reports for all canonical result files (excludes disease datasets).",
    )
    parser.add_argument(
        "--diseases",
        action="store_true",
        help="Regenerate reports for all disease dataset result files.",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent

    if args.all and args.diseases:
        run_list = KNOWN_RESULTS + DISEASE_RESULTS
    elif args.all:
        run_list = KNOWN_RESULTS
    elif args.diseases:
        run_list = DISEASE_RESULTS
    else:
        run_list = None

    if run_list is not None:
        for name in run_list:
            p = base / name
            if p.exists():
                print(f"\nGenerating report for {name}...")
                generate_report(p, out_dir=args.out)
            else:
                print(f"  Skipping {name} (not found)")
    elif args.json:
        generate_report(args.json, figure_path=args.figure, out_dir=args.out)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
