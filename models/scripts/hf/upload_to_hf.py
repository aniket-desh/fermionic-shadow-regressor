"""
Upload (or re-upload) a molecular-shadows regressor checkpoint to Hugging Face.

Re-runnable: every push creates a new git commit on `main`; --version_tag also
creates an immutable tag pinning that exact commit, so collaborators can pin to
a specific architecture version via `revision="v10"` regardless of later pushes.

Reads HF_TOKEN from .env or the environment.

Examples:
    # First-time upload of H2 v10
    python3 -m scripts.hf.upload_to_hf \\
        --model_dir results/fermionic_pipeline/regression/h2_regress_v10_model \\
        --data_h5   results/fermionic_pipeline/regression/h2_regress_v10/regression_targets.h5 \\
        --repo_id   aniketdesh/molecular-shadows-h2-v10 \\
        --model_card scripts/hf/model_card_h2_v10.md \\
        --version_tag v10

    # Re-upload after architecture change (e.g., post-v11 fix)
    python3 -m scripts.hf.upload_to_hf \\
        --model_dir results/fermionic_pipeline/regression/h4_regress_v11_s42_model \\
        --data_h5   results/fermionic_pipeline/regression/h4_regress_v10/regression_targets.h5 \\
        --repo_id   aniketdesh/molecular-shadows-h4-v10 \\
        --model_card scripts/hf/model_card_h4_v11.md \\
        --version_tag v11

Bundled into the repo (everything the collaborator needs to run inference):
    regressor.pt              torch payload (state_dict + config + R/t grid)
    observable_regressor.py   model architecture (single file, copied verbatim)
    inference.py              high-level loader (`MolecularShadowsRegressor.from_local`)
    orbital_energies.npz      R-grid + per-orbital HF energies (and omega_op if present)
    eval_results.json         held-out eval metrics (if available)
    history.json              training curves (if available)
    README.md                 model card (used as Hub landing page)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np


def load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise SystemExit(
        "HF_TOKEN not found. Set it in .env (HF_TOKEN=hf_...) or export it in the shell.\n"
        "Generate one at https://huggingface.co/settings/tokens with WRITE scope."
    )


def stage_artifacts(model_dir: Path, data_h5: Path, model_card: Path,
                    out_dir: Path, repo_root: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_src = model_dir / "regressor.pt"
    if not ckpt_src.exists():
        raise SystemExit(f"missing checkpoint: {ckpt_src}")
    shutil.copy2(ckpt_src, out_dir / "regressor.pt")

    arch_src = repo_root / "fermionic_pipeline/models/observable_regressor.py"
    if not arch_src.exists():
        raise SystemExit(f"missing arch source: {arch_src}")
    shutil.copy2(arch_src, out_dir / "observable_regressor.py")

    inference_src = repo_root / "scripts/hf/inference.py"
    shutil.copy2(inference_src, out_dir / "inference.py")

    if not model_card.exists():
        raise SystemExit(f"missing model card: {model_card}")
    shutil.copy2(model_card, out_dir / "README.md")

    if not data_h5.exists():
        raise SystemExit(f"missing data HDF5 (needed for orbital energies): {data_h5}")
    with h5py.File(data_h5, "r") as f:
        R_grid = f["R_values"][...]
        if "hf_orbital_energies" not in f:
            raise SystemExit(f"{data_h5} has no hf_orbital_energies group")
        orb = f["hf_orbital_energies"][...]
        npz_kwargs = dict(R_grid=R_grid, orbital_energies=orb)
        if "omega_op" in f:
            npz_kwargs["omega_op"] = f["omega_op"][...]
    np.savez(out_dir / "orbital_energies.npz", **npz_kwargs)

    eval_src = model_dir / "eval/regressor_eval.json"
    if eval_src.exists():
        eval_data = json.loads(eval_src.read_text())
        per_R = sorted(eval_data["results"], key=lambda r: r["R"])
        summary = {
            "checkpoint": eval_data.get("checkpoint"),
            "n_geometries": len(per_R),
            "pearson_mean": float(np.mean([r["pearson_mean"] for r in per_R])),
            "pearson_median": float(np.mean([r["pearson_median"] for r in per_R])),
            "range_ratio_mean": float(np.mean([r["range_ratio_mean"] for r in per_R])),
            "mse_mean": float(np.mean([r["mse"] for r in per_R])),
        }
        (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
        shutil.copy2(eval_src, out_dir / "eval_results.json")

    history_src = model_dir / "history.json"
    if history_src.exists():
        shutil.copy2(history_src, out_dir / "history.json")

    return {
        "n_orb": int(orb.shape[1]),
        "n_R": int(R_grid.shape[0]),
        "R_range": (float(R_grid.min()), float(R_grid.max())),
    }


def push_repo(staging_dir: Path, repo_id: str, version_tag: str | None,
              private: bool, commit_message: str, token: str) -> None:
    from huggingface_hub import HfApi, hf_hub_url

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[ok] repo ensured: {repo_id} (private={private})")

    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print(f"[ok] commit pushed to main: {commit_message!r}")

    if version_tag:
        try:
            api.create_tag(repo_id=repo_id, tag=version_tag,
                           repo_type="model", tag_message=commit_message)
            print(f"[ok] tag created: {version_tag}")
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg or "conflict" in msg or "409" in msg:
                print(f"[warn] tag {version_tag!r} already exists; "
                      f"delete it on the Hub first if you want it to point to the new commit.")
            else:
                raise

    print()
    print(f"  Hub URL:   https://huggingface.co/{repo_id}")
    if version_tag:
        print(f"  Pin via:   MolecularShadowsRegressor.from_hub("
              f"'{repo_id}', revision='{version_tag}', token=...)")
    else:
        print(f"  Load via:  MolecularShadowsRegressor.from_hub('{repo_id}', token=...)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, type=Path,
                    help="Directory containing regressor.pt (and optionally eval/, history.json)")
    ap.add_argument("--data_h5", required=True, type=Path,
                    help="HDF5 with hf_orbital_energies — R-grid for inference-time interpolation")
    ap.add_argument("--repo_id", required=True,
                    help="HF repo, e.g. aniketdesh/molecular-shadows-h2-v10")
    ap.add_argument("--model_card", required=True, type=Path,
                    help="Markdown model card (becomes README.md on the Hub)")
    ap.add_argument("--version_tag", default=None,
                    help="Optional tag (e.g. v10, v11). Pins the architecture version "
                         "to a specific commit so future pushes don't break collaborator's pins.")
    ap.add_argument("--public", action="store_true",
                    help="Make the repo public (default: private)")
    ap.add_argument("--commit_message", default=None,
                    help="Custom commit message (default: '<repo_id> <tag>')")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent
    token = load_hf_token()
    commit = args.commit_message or f"upload {args.repo_id.split('/')[-1]}"
    if args.version_tag:
        commit = f"{commit} ({args.version_tag})"

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "repo"
        info = stage_artifacts(args.model_dir, args.data_h5, args.model_card,
                               staging, repo_root)
        print(f"[ok] staged at {staging}")
        print(f"     n_orb={info['n_orb']}, n_R={info['n_R']}, "
              f"R={info['R_range'][0]:.2f}->{info['R_range'][1]:.2f} A")
        print(f"     files: {sorted(p.name for p in staging.iterdir())}")
        print()

        push_repo(staging, args.repo_id, args.version_tag, not args.public,
                  commit, token)


if __name__ == "__main__":
    main()
