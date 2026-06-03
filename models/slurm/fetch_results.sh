#!/bin/bash
# Run LOCALLY to pull fermionic pipeline results + logs from Trillium.
#
# Excludes large dataset files (*.h5) by default — those are generated on
# Trillium and rarely needed locally. Pulls everything else: checkpoints
# (*.pt), eval JSON, plots, history. Pass --include-h5 to override.
#
# --which TOKEN restricts BOTH results and logs to paths whose name contains
# TOKEN (substring, at any depth). Use it when local disk is tight and you only
# want one run — e.g. `--which v17` pulls just results/.../*v17*/ dirs and
# logs/*v17* files, not the whole tree. TOKEN is a literal substring, so `v17`
# matches `h4_regress_v17_v17_orb_s42_model` but never `v1`/`v10`/`v16`.
#
# Usage:
#   bash slurm/fetch_results.sh                       # all tags, no .h5
#   bash slurm/fetch_results.sh fast                  # specific top-level tag, no .h5
#   bash slurm/fetch_results.sh fast --logs           # also fetch logs
#   bash slurm/fetch_results.sh --which v17           # only *v17* results
#   bash slurm/fetch_results.sh --which v17 --logs    # only *v17* results + logs
#   bash slurm/fetch_results.sh --include-h5          # include big datasets
#   bash slurm/fetch_results.sh fast --include-h5     # tag + include .h5

set -e

REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
REMOTE_DIR="\$SCRATCH/generative-quantum-states"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TAG=""
WHICH=""
FETCH_LOGS=false
INCLUDE_H5=false
while [ $# -gt 0 ]; do
    case "$1" in
        --logs)        FETCH_LOGS=true ;;
        --include-h5)  INCLUDE_H5=true ;;
        --which)       WHICH="$2"; shift ;;
        --which=*)     WHICH="${1#*=}" ;;
        --*)           echo "unknown flag: $1"; exit 1 ;;
        *)             [ -z "$TAG" ] && TAG="$1" ;;
    esac
    shift
done

if [ -n "$WHICH" ] && [ -n "$TAG" ]; then
    echo "note: --which '$WHICH' overrides positional tag '$TAG' for results selection"
fi

# Default rsync filters: skip large dataset files unless --include-h5. These are
# listed FIRST so the .h5 exclude wins before any --which include rule.
RSYNC_FILTERS=(--max-size=200M)
if [ "$INCLUDE_H5" = false ]; then
    RSYNC_FILTERS+=(--exclude='*.h5')
fi

echo "=== Fetching results from Trillium ==="
echo "  include .h5: ${INCLUDE_H5}    max file size: 200M (override w/ --include-h5)"

if [ -n "$WHICH" ]; then
    # Substring-filter the whole tree: descend into every dir (--include='*/'),
    # keep only files under a dir whose name contains TOKEN (the '*/***' tail
    # matches that dir and everything beneath it), drop the rest, and prune the
    # empty dir skeletons left behind (-m).
    echo "  which: *${WHICH}*  (substring match on dir names, recursive)"
    rsync -avzm "${RSYNC_FILTERS[@]}" \
        --include='*/' \
        --include="*${WHICH}*/***" \
        --exclude='*' \
        "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/" \
        "${LOCAL_DIR}/results/fermionic_pipeline/"
elif [ -n "$TAG" ]; then
    echo "  tag: $TAG"
    rsync -avz "${RSYNC_FILTERS[@]}" \
        "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/${TAG}/" \
        "${LOCAL_DIR}/results/fermionic_pipeline/${TAG}/"
else
    echo "  all tags"
    rsync -avz "${RSYNC_FILTERS[@]}" \
        "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/" \
        "${LOCAL_DIR}/results/fermionic_pipeline/"
fi

if [ "$FETCH_LOGS" = true ]; then
    echo ""
    echo "=== Fetching logs ==="
    [ -n "$WHICH" ] && echo "  which: *${WHICH}* (only log files containing the token)"
    mkdir -p "${LOCAL_DIR}/logs"
    BEFORE=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f | wc -l | tr -d ' ')
    # Per-prefix include patterns; when --which is set, also require the token.
    LOG_INCLUDES=()
    for prefix in ferm-pipeline_ exact_gen_ clf_ eval_ reg_; do
        if [ -n "$WHICH" ]; then
            LOG_INCLUDES+=(--include="${prefix}*${WHICH}*")
        else
            LOG_INCLUDES+=(--include="${prefix}*")
        fi
    done
    rsync -avz \
        "${LOG_INCLUDES[@]}" \
        --exclude='*' \
        "${REMOTE}:${REMOTE_DIR}/logs/" \
        "${LOCAL_DIR}/logs/"
    AFTER=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f | wc -l | tr -d ' ')
    echo "  log files: ${BEFORE} -> ${AFTER} ($((AFTER - BEFORE)) new)"
    for prefix in ferm-pipeline_ exact_gen_ clf_ eval_ reg_; do
        n=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -name "${prefix}*" -type f | wc -l | tr -d ' ')
        printf "    %-16s %d\n" "${prefix}*" "${n}"
    done
fi

echo ""
echo "=== Fetch complete ==="
LOCAL_SIZE=$(du -sh "${LOCAL_DIR}/results/fermionic_pipeline/" 2>/dev/null | cut -f1)
echo "  local results size: ${LOCAL_SIZE}"
