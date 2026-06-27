#!/usr/bin/env bash
# Efficient Step-3 push: build a commit on top of origin/runpod-results via git
# plumbing (no 461MB working-tree checkout). Only NEW blobs (~3.6MB) are hashed
# locally and sent; the existing 461MB of PDFs already on the remote are not re-uploaded.
set -euo pipefail
cd "$(dirname "$0")/../.."              # -> repo root
set -a; . ./.env 2>/dev/null; set +a
: "${GH_TOKEN:?}"
GH_URL="https://x-access-token:${GH_TOKEN}@github.com/aniket-desh/fermionic-shadow-regressor.git"
RREL=models/results/fermionic_pipeline/regression
IDX=/tmp/rr_step3.idx; rm -f "$IDX"

git fetch -q "$GH_URL" runpod-results
BASE=$(git rev-parse FETCH_HEAD)
echo "base = $BASE"
GIT_INDEX_FILE="$IDX" git read-tree "$BASE"

add() {  # $1 = path (relative to repo root); hashes blob and stages at that path
  [ -f "$1" ] || { echo "  [skip missing] $1"; return; }
  local blob; blob=$(git hash-object -w "$1")
  GIT_INDEX_FILE="$IDX" git update-index --add --cacheinfo "100644,$blob,$1"
  echo "  [add] $1"
}

LIH=$RREL/lih_regress_v1_orb_s42_model
add "$LIH/plots/coherence_heatmap.pdf"
add "$LIH/plots/coherence_grid.npz"
add "$LIH/plots_extrap/coherence_heatmap.pdf"
add "$LIH/plots_extrap/coherence_grid.npz"
add "$LIH/plots/regression_summary.pdf"
add "$LIH/eval/regressor_eval.json"
add "$LIH/eval/composition_diagnostic.json"
add "$LIH/history.json"
for m in n2 beh2; do
  add "$RREL/${m}_regress_v1_orb_s42_model/plots_extrap/coherence_heatmap.pdf"
  add "$RREL/${m}_regress_v1_orb_s42_model/plots_extrap/coherence_grid.npz"
done
add models/runpod_logs/R_eq_results.txt
add models/runpod_logs/compute_req.log
add models/runpod_logs/req_wide.log

TREE=$(GIT_INDEX_FILE="$IDX" git write-tree)
MSG="Briefing #3: LiH heatmaps+grids, N2/BeH2 grids, R_eq (corrected)

LiH in-box r-bar=0.992 (median 0.997). coherence_grid.npz for lih(in+extrap), n2, beh2.
R_eq corrected (committed compute_req.py scan windows too narrow -> all boundary artifacts):
R_EQ={h4:1.68, lih:2.86, beh2:2.44, n2:2.24}. See R_eq_results.txt.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
COMMIT=$(git -c user.name="aniket-desh" -c user.email="aniket4@illinois.edu" commit-tree "$TREE" -p "$BASE" -m "$MSG")
echo "new commit = $COMMIT"
git push "$GH_URL" "$COMMIT:refs/heads/runpod-results"
echo "### STEP3 FAST PUSH DONE ###"
