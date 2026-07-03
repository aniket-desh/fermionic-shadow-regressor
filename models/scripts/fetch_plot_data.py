#!/usr/bin/env python3
"""One-shot fetch of the derived per-figure plot cache from HuggingFace into the
local ``results/`` tree, so the 8 fast (model-free) figure scripts and the
GPU-free coherence replot run out of the box.

Downloads ``plot_cache/`` from the private dataset repo
``aniketdesh/molecular-shadows-datasets`` and copies it into ``models/results/``
(the layout under ``plot_cache/`` mirrors ``results/`` exactly).

Requires read access to the repo. Set your token first (either):
    export HF_TOKEN=hf_xxx            # your token with read access
    # or put HF_TOKEN=hf_xxx in models/.env

Run from models/:
    python -m scripts.fetch_plot_data          # fetch the plot cache
    python -m scripts.fetch_plot_data --with-datasets   # also fetch raw .h5 + .pt
"""
import argparse
import os
import shutil
import sys

REPO = "aniketdesh/molecular-shadows-datasets"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def _token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    env = os.path.join(HERE, "..", ".env")
    if not tok and os.path.exists(env):
        for line in open(env):
            if line.strip().startswith(("HF_TOKEN", "HUGGINGFACE")):
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not tok:
        try:
            from huggingface_hub import HfFolder
            tok = HfFolder.get_token()
        except Exception:
            pass
    return tok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-datasets", action="store_true",
                    help="also download the raw {mol}/*.h5 + regressor.pt (needed only "
                         "to re-run the model-gated figures)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    tok = _token()   # optional: the repo is public, so no token is needed; a token is
                     # only required if it is ever set back to private.

    patterns = ["plot_cache/**"]
    if args.with_datasets:
        patterns += ["*/regression_targets.h5", "*/extrap_regression_targets.h5",
                     "*/regressor.pt", "*/regressor_eval.json"]

    try:
        local = snapshot_download(repo_id=REPO, repo_type="dataset", token=tok,
                                  allow_patterns=patterns)
    except Exception as e:
        sys.exit(f"Download failed ({type(e).__name__}). If the repo is private, set HF_TOKEN "
                 f"(read access) in your env or models/.env.\n  {str(e)[:200]}")
    print(f"[hf] downloaded to {local}")

    # copy plot_cache/<...> -> results/<...>  (mirrors the results/ layout)
    src_root = os.path.join(local, "plot_cache")
    n = 0
    for dirpath, _, files in os.walk(src_root):
        for f in files:
            src = os.path.join(dirpath, f)
            rel = os.path.relpath(src, src_root)          # path after plot_cache/
            dst = os.path.join(RESULTS, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    print(f"[done] placed {n} plot-cache files under results/")
    if args.with_datasets:
        print(f"[info] raw datasets/checkpoints are in the snapshot at:\n       {local}\n"
              f"       point the eval commands' --data_path/--checkpoint there "
              f"(e.g. {local}/h4/regression_targets.h5).")


if __name__ == "__main__":
    main()
