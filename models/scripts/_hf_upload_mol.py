#!/usr/bin/env python3
"""Upload one molecule's heavy artifacts to a private HF dataset repo.
Layout in repo:  <mol>/regression_targets.h5, <mol>/extrap_regression_targets.h5,
                 <mol>/regressor.pt, <mol>/regressor_eval.json
"""
import os, sys
from huggingface_hub import HfApi

REPO = "aniketdesh/molecular-shadows-datasets"
R = "results/fermionic_pipeline/regression"
tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

def main(mol):
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True, token=tok)
    files = [
        (f"{R}/{mol}_regress_v1/regression_targets.h5",        f"{mol}/regression_targets.h5"),
        (f"{R}/{mol}_regress_v1_extrap/regression_targets.h5", f"{mol}/extrap_regression_targets.h5"),
        (f"{R}/{mol}_regress_v1_orb_s42_model/regressor.pt",   f"{mol}/regressor.pt"),
        (f"{R}/{mol}_regress_v1_orb_s42_model/eval/regressor_eval.json", f"{mol}/regressor_eval.json"),
    ]
    for local, remote in files:
        if not os.path.exists(local):
            print(f"  [skip] missing {local}")
            continue
        sz = os.path.getsize(local) / 1e6
        print(f"  [up] {local} ({sz:.0f} MB) -> {REPO}:{remote}")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=REPO, repo_type="dataset", token=tok,
                        commit_message=f"add {remote}")
    print(f"  [done] {mol}")

if __name__ == "__main__":
    main(sys.argv[1])
