#!/usr/bin/env bash
# Push H2O v2 (negative result) + pre-flight:
#   GitHub runpod-results : v2 eval JSONs + plots + pre-flight stdout + v2 run log
#   HF dataset repo       : v2 .h5 + checkpoint + eval json under h2o_v2/
# Isolated /tmp clone for GitHub (never branch-switch the live tree).
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."                 # -> models/
REPO_ROOT=$(cd .. && pwd)
set -a; . "$REPO_ROOT/.env" 2>/dev/null; set +a
: "${GH_TOKEN:?}"; : "${HF_TOKEN:?}"
GH_SLUG="aniket-desh/fermionic-shadow-regressor"
GH_URL="https://x-access-token:${GH_TOKEN}@github.com/${GH_SLUG}.git"
GH_BRANCH="runpod-results"
RREL=models/results/fermionic_pipeline/regression
TAG=h2o_regress_v2
MDIR=$RREL/${TAG}_orb_s42_model

echo "### HF upload: $TAG -> aniketdesh/molecular-shadows-datasets:h2o_v2/ ###"
python - <<PY
import os
from huggingface_hub import HfApi
tok=os.environ["HF_TOKEN"]; api=HfApi(); REPO="aniketdesh/molecular-shadows-datasets"
api.create_repo(REPO,repo_type="dataset",private=True,exist_ok=True,token=tok)
R="results/fermionic_pipeline/regression"
files=[(f"{R}/$TAG/regression_targets.h5","h2o_v2/regression_targets.h5"),
       (f"{R}/${TAG}_orb_s42_model/regressor.pt","h2o_v2/regressor.pt"),
       (f"{R}/${TAG}_orb_s42_model/eval/regressor_eval.json","h2o_v2/regressor_eval.json")]
for l,r in files:
    if os.path.exists(l):
        print(f"  [up] {l} ({os.path.getsize(l)/1e6:.0f} MB) -> {r}")
        api.upload_file(path_or_fileobj=l,path_in_repo=r,repo_id=REPO,repo_type="dataset",token=tok,commit_message=f"add {r}")
    else: print(f"  [skip] {l}")
print("  [done] h2o_v2")
PY

echo "### GitHub push (isolated clone) ###"
TMP=/tmp/fsr-push-v2; rm -rf "$TMP"
git clone --quiet --branch "$GH_BRANCH" --single-branch --depth 1 "$GH_URL" "$TMP"
cd "$REPO_ROOT"
for f in "$MDIR"/*.json "$MDIR"/eval/*.json "$MDIR"/plots/*.pdf; do [ -f "$f" ] && cp --parents "$f" "$TMP/"; done
for f in models/runpod_logs/preflight_h2o_*.txt models/runpod_logs/h2o_v2.log; do [ -f "$f" ] && cp --parents "$f" "$TMP/"; done
cd "$TMP"
git add -f models/ 2>/dev/null || true
git -c user.name="aniket-desh" -c user.email="aniket4@illinois.edu" \
    commit -q -m "H2O v2 (t_max=2500): NEGATIVE result + line-spectrum pre-flight

v2 held-out Pearson ~0 (model collapsed to flat/mean; worse than v1's 0.385).
Pre-flight: >=5% lines resolved at t=800; longer horizon did not fix h2o. NO-GO.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" || { echo "(nothing to commit)"; cd /; rm -rf "$TMP"; exit 0; }
git push --quiet "$GH_URL" "$GH_BRANCH"
cd /; rm -rf "$TMP"
echo "### DONE h2o_v2 push ###"
