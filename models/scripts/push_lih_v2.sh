#!/usr/bin/env bash
# Push LiH v2 (briefing #4): HF dataset+ckpt under lih_v2/, GitHub (plumbing) lightweight results.
# NOTE: v2 is a REGRESSION vs v1 ([1.0,3.2] r-bar 0.978 < v1 0.992); pushed for the record.
set -euo pipefail
cd "$(dirname "$0")/../.."              # -> repo root
set -a; . ./.env 2>/dev/null; set +a
: "${GH_TOKEN:?}"; : "${HF_TOKEN:?}"
GH_URL="https://x-access-token:${GH_TOKEN}@github.com/aniket-desh/fermionic-shadow-regressor.git"
RREL=models/results/fermionic_pipeline/regression
MDIR=$RREL/lih_regress_v2_orb_s42_model

echo "### HF upload: lih v2 -> aniketdesh/molecular-shadows-datasets:lih_v2/ ###"
python - <<'PY'
import os
from huggingface_hub import HfApi
tok=os.environ["HF_TOKEN"]; api=HfApi(); REPO="aniketdesh/molecular-shadows-datasets"
api.create_repo(REPO,repo_type="dataset",private=True,exist_ok=True,token=tok)
R="results/fermionic_pipeline/regression"
files=[(f"{R}/lih_regress_v2/regression_targets.h5","lih_v2/regression_targets.h5"),
       (f"{R}/lih_regress_v2_orb_s42_model/regressor.pt","lih_v2/regressor.pt"),
       (f"{R}/lih_regress_v2_orb_s42_model/eval/regressor_eval.json","lih_v2/regressor_eval.json")]
for l,r in files:
    if os.path.exists(l):
        print(f"  [up] {l} ({os.path.getsize(l)/1e6:.0f} MB) -> {r}")
        api.upload_file(path_or_fileobj=l,path_in_repo=r,repo_id=REPO,repo_type="dataset",token=tok,commit_message=f"add {r}")
print("  [done] lih_v2")
PY

echo "### GitHub push (plumbing; only new blobs uploaded) ###"
IDX=/tmp/rr_lihv2.idx; rm -f "$IDX"
git fetch -q "$GH_URL" runpod-results
BASE=$(git rev-parse FETCH_HEAD)
GIT_INDEX_FILE="$IDX" git read-tree "$BASE"
add() { [ -f "$1" ] || { echo "  [skip] $1"; return; }; local b; b=$(git hash-object -w "$1"); GIT_INDEX_FILE="$IDX" git update-index --add --cacheinfo "100644,$b,$1"; echo "  [add] $1"; }
add "$MDIR/eval/regressor_eval.json"
add "$MDIR/plots/coherence_heatmap.pdf"
add "$MDIR/plots/coherence_grid.npz"
add "$MDIR/plots/regression_summary.pdf"
add models/runpod_logs/preflight_lih_edge.txt
TREE=$(GIT_INDEX_FILE="$IDX" git write-tree)
MSG="LiH v2 (buffered box [0.8,3.4]) — REGRESSION, keep v1

v2 [1.0,3.2] held-out r-bar=0.978 < v1 0.992. My v1 re-run has NO R=3.2 spike (R3.20=0.992;
briefing's 0.45 is from the original off-pod ckpt). Buffer [0.8,1.0] adds a hard low-R region
(R0.95-1.02: 0.42-0.81; preflight bw99 jump 0.43->0.15 @R0.95-1.0). Recommend keeping v1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
COMMIT=$(git -c user.name="aniket-desh" -c user.email="aniket4@illinois.edu" commit-tree "$TREE" -p "$BASE" -m "$MSG")
git push "$GH_URL" "$COMMIT:refs/heads/runpod-results"
echo "### LIH V2 PUSH DONE ###"
