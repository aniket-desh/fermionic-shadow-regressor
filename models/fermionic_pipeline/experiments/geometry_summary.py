"""Summarize geometry amortization curves and explicit acquisition accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fermionic_pipeline.eval.nature_style import apply_nature_style
from fermionic_pipeline.experiments.common import write_json


def scalar(aggregate, key):
    value = aggregate.get(key, {})
    return value.get("mean") if isinstance(value, dict) else value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry_root", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_pdf", required=True)
    ap.add_argument("--n_times", type=int, required=True)
    ap.add_argument("--n_q", type=int, default=500)
    ap.add_argument("--born_samples", type=int, default=1,
                    help="Use 1 for exact-marginal targets; set S for finite-shot accounting.")
    args = ap.parse_args()
    rows = []
    for metrics_path in Path(args.geometry_root).glob("*/spectral_metrics.json"):
        with metrics_path.open() as f: metrics = json.load(f)
        baseline_path = metrics_path.parent / "structured_baseline.json"
        baseline = json.load(baseline_path.open()) if baseline_path.exists() else None
        # The baseline embeds the exact split manifest.
        split = baseline["split"] if baseline else {}
        n_train = len(split.get("train", metrics.get("train_r_indices", [])))
        n_all = n_train + len(split.get("test", []))
        rows.append({
            "tag": metrics_path.parent.name, "kind": split.get("kind"),
            "n_train": n_train, "n_all": n_all,
            "train_fraction": n_train / n_all if n_all else None,
            "fsr_pearson": scalar(metrics["aggregate"], "temporal_pearson"),
            "fsr_spectral_wasserstein": scalar(metrics["aggregate"], "spectral_wasserstein"),
            "baseline_pearson": scalar(baseline["aggregate"], "temporal_pearson") if baseline else None,
            "baseline_spectral_wasserstein": scalar(baseline["aggregate"], "spectral_wasserstein") if baseline else None,
            "training_outcomes": int(n_train * args.n_times * args.n_q * args.born_samples),
            "direct_outcomes": int(n_all * args.n_times * args.n_q * args.born_samples),
            "acquisition_fraction": n_train / n_all if n_all else None,
        })
    rows.sort(key=lambda x: (x["kind"] or "", x["n_train"]))
    write_json(args.output_json, {"accounting": {
        "formula": "N_R * N_T * N_Q * S", "n_times": args.n_times,
        "n_q": args.n_q, "born_samples": args.born_samples,
    }, "results": rows})
    curve = [r for r in rows if r["kind"] == "learning_curve"]
    apply_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    if curve:
        x = np.array([r["n_train"] for r in curve])
        axes[0].plot(x, [r["fsr_pearson"] for r in curve], "o-", label="FSR")
        axes[0].plot(x, [r["baseline_pearson"] for r in curve], "s--", label="line interpolation")
        axes[1].plot(x, [r["fsr_spectral_wasserstein"] for r in curve], "o-", label="FSR")
        axes[1].plot(x, [r["baseline_spectral_wasserstein"] for r in curve], "s--", label="line interpolation")
    axes[0].set(xlabel="training geometries", ylabel="held-out temporal Pearson r")
    axes[1].set(xlabel="training geometries", ylabel="spectral Wasserstein distance")
    axes[0].legend(frameon=False); axes[1].legend(frameon=False)
    fig.tight_layout(); Path(args.output_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_pdf); plt.close(fig)
    print(f"[done] geometry summary -> {args.output_json}, {args.output_pdf}")


if __name__ == "__main__":
    main()
