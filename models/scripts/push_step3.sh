#!/usr/bin/env bash
# Briefing #3 Step 3 GitHub push (isolated clone; never branch-switch live tree):
#   lih  : plots/ + plots_extrap/ coherence_heatmap.pdf + coherence_grid.npz,
#          regression_summary.pdf, eval/*.json
#   n2,beh2 : plots_extrap/ coherence_heatmap.pdf + coherence_grid.npz
#   + R_eq_results.txt and the compute_req/req_wide/lih logs
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."                 # -> models/
REPO_ROOT=$(cd .. && pwd)
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
: "${GH_TOKEN:?}"
GH_SLUG="aniket-desh/fermionic-shadow-regressor"
GH_URL="https://x-access-token:${GH_TOKEN}@github.com/${GH_SLUG}.git"
GH_BRANCH="runpod-results"
RREL=models/results/fermionic_pipeline/regression

TMP=/tmp/fsr-push-step3; rm -rf "$TMP"
git clone --quiet --branch "$GH_BRANCH" --single-branch --depth 1 "$GH_URL" "$TMP"
cd "$REPO_ROOT"

lihd=$RREL/lih_regress_v1_orb_s42_model
for f in "$lihd"/plots/coherence_heatmap.pdf "$lihd"/plots/coherence_grid.npz \
         "$lihd"/plots_extrap/coherence_heatmap.pdf "$lihd"/plots_extrap/coherence_grid.npz \
         "$lihd"/plots/regression_summary.pdf "$lihd"/eval/*.json "$lihd"/*.json; do
  [ -f "$f" ] && cp --parents "$f" "$TMP/"
done
for m in n2 beh2; do
  md=$RREL/${m}_regress_v1_orb_s42_model
  for f in "$md"/plots_extrap/coherence_heatmap.pdf "$md"/plots_extrap/coherence_grid.npz; do
    [ -f "$f" ] && cp --parents "$f" "$TMP/"
  done
done
for f in models/runpod_logs/R_eq_results.txt models/runpod_logs/compute_req.log \
         models/runpod_logs/req_wide.log models/runpod_logs/instance_lih.log; do
  [ -f "$f" ] && cp --parents "$f" "$TMP/"
done

cd "$TMP"
git add -f models/ 2>/dev/null || true
git -c user.name="aniket-desh" -c user.email="aniket4@illinois.edu" \
    commit -q -m "Briefing #3: LiH heatmaps+grids, N2/BeH2 grids, R_eq (corrected)

LiH in-box r-bar=0.992 (median 0.997). coherence_grid.npz added for lih(in+extrap),
n2, beh2. R_eq corrected (committed compute_req.py scan windows too narrow -> boundary
artifacts): R_EQ={h4:1.68, lih:2.86, beh2:2.44, n2:2.24}. See R_eq_results.txt.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" || { echo "(nothing to commit)"; cd /; rm -rf "$TMP"; exit 0; }
git push --quiet "$GH_URL" "$GH_BRANCH"
cd /; rm -rf "$TMP"
echo "### STEP3 PUSH DONE ###"
