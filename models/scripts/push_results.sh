#!/usr/bin/env bash
# Push results incrementally as molecules finish — WITHOUT touching the main
# working tree's branch (the run writes gitignored logs/results live, so we must
# never `git checkout` the main tree). GitHub side uses an isolated /tmp clone.
#   GitHub      -> branch 'runpod-results' : logs + plot PDFs + eval/history JSONs
#   HuggingFace -> private dataset repo    : per-molecule .h5 + checkpoint + eval json
# Usage: bash scripts/push_results.sh <mol> [<mol> ...]
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."                 # -> models/
REPO_ROOT=$(cd .. && pwd)
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
: "${GH_TOKEN:?GH_TOKEN missing}"; : "${HF_TOKEN:?HF_TOKEN missing}"

GH_SLUG="aniket-desh/fermionic-shadow-regressor"
GH_URL="https://x-access-token:${GH_TOKEN}@github.com/${GH_SLUG}.git"
GH_BRANCH="runpod-results"
RREL=models/results/fermionic_pipeline/regression
MOLS=("$@")
[ ${#MOLS[@]} -gt 0 ] || { echo "no molecules given"; exit 1; }

echo "### HF upload (private dataset repo) for: ${MOLS[*]} ###"
for m in "${MOLS[@]}"; do
  python "$REPO_ROOT/models/scripts/_hf_upload_mol.py" "$m"
done

echo "### GitHub push (isolated clone) for: ${MOLS[*]} ###"
TMP=/tmp/fsr-push-clone
rm -rf "$TMP"
if git ls-remote --exit-code --heads "$GH_URL" "$GH_BRANCH" >/dev/null 2>&1; then
  git clone --quiet --branch "$GH_BRANCH" --single-branch --depth 1 "$GH_URL" "$TMP"
else
  git clone --quiet --single-branch --depth 1 "$GH_URL" "$TMP"
  ( cd "$TMP" && git checkout -q -b "$GH_BRANCH" )
fi
# copy this wave's artifacts + a fresh snapshot of all logs into the clone
cd "$REPO_ROOT"
for m in "${MOLS[@]}"; do
  d=$RREL/${m}_regress_v1_orb_s42_model
  for f in "$d"/*.json "$d"/eval/*.json "$d"/plots/*.pdf "$d"/plots_extrap/*.pdf; do
    [ -f "$f" ] && cp --parents "$f" "$TMP/"
  done
done
for f in models/runpod_logs/*.log; do cp --parents "$f" "$TMP/"; done
cd "$TMP"
git add -f models/ 2>/dev/null || true
git -c user.name="aniket-desh" -c user.email="aniket4@illinois.edu" \
    commit -q -m "RunPod results: ${MOLS[*]} eval JSONs + plots + logs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" || { echo "(nothing new to commit)"; rm -rf "$TMP"; echo "### DONE ${MOLS[*]} ###"; exit 0; }
git push --quiet "$GH_URL" "$GH_BRANCH"
cd /; rm -rf "$TMP"
echo "### DONE ${MOLS[*]} ###"
