#!/bin/bash
# Run LOCALLY to pull fermionic pipeline results + logs from Trillium.
#
# Excludes large dataset files (*.h5) by default — those are generated on
# Trillium and rarely needed locally. Pulls everything else: checkpoints
# (*.pt), eval JSON, plots, history. Pass --include-h5 to override.
#
# --which TOKEN [TOKEN ...] restricts BOTH results and logs to paths whose name
# contains ANY of the tokens (substring, at any depth; OR-matched). Use it when
# local disk is tight and you only want some runs — e.g. `--which v17` pulls just
# results/.../*v17*/ dirs and logs/*v17* files; `--which v13 extrap` pulls both.
# A token is a literal substring, so `v17` matches `h4_regress_v17_v17_orb_s42_model`
# but never `v1`/`v10`/`v16`.
#
# Usage:
#   bash slurm/fetch_results.sh                       # all tags, no .h5
#   bash slurm/fetch_results.sh fast                  # specific top-level tag, no .h5
#   bash slurm/fetch_results.sh fast --logs           # also fetch logs
#   bash slurm/fetch_results.sh --which v17           # only *v17* results
#   bash slurm/fetch_results.sh --which v17 --logs    # only *v17* results + logs
#   bash slurm/fetch_results.sh --include-h5          # include big datasets
#   bash slurm/fetch_results.sh fast --include-h5     # tag + include .h5
#   bash slurm/fetch_results.sh --which v13 --data-only  # ONLY *.h5 (no pdfs/json/pt)
#   bash slurm/fetch_results.sh --which v13 --light --logs     # dipole npz + logs (no h5/pt)
#   bash slurm/fetch_results.sh --which extrap --light --logs  # heatmap pdf + logs (no h5/pt)
#   bash slurm/fetch_results.sh --which v13 extrap --light --logs  # both, in one pull

set -e

REMOTE="aniketrd@trillium-gpu.scinet.utoronto.ca"
REMOTE_DIR="\$SCRATCH/generative-quantum-states"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TAG=""
WHICH_TOKENS=()       # one or more substring tokens; results/logs matching ANY are pulled
FETCH_LOGS=false
INCLUDE_H5=false
DATA_ONLY=false
LIGHT=false
while [ $# -gt 0 ]; do
    case "$1" in
        --logs)        FETCH_LOGS=true ;;
        --include-h5)  INCLUDE_H5=true ;;
        --data-only)   DATA_ONLY=true; INCLUDE_H5=true ;;  # only *.h5; implies no size cap
        --light)       LIGHT=true ;;   # deliverables only: skip *.h5 AND *.pt (datasets + checkpoints)
        --which)       # consume all following non-flag args as tokens: --which v13 extrap
                       shift
                       while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do WHICH_TOKENS+=("$1"); shift; done
                       continue ;;
        --which=*)     WHICH_TOKENS+=("${1#*=}") ;;
        --*)           echo "unknown flag: $1"; exit 1 ;;
        *)             [ -z "$TAG" ] && TAG="$1" ;;
    esac
    shift
done

if [ ${#WHICH_TOKENS[@]} -gt 0 ] && [ -n "$TAG" ]; then
    echo "note: --which (${WHICH_TOKENS[*]}) overrides positional tag '$TAG' for results selection"
fi

# Default rsync filters: skip large dataset files unless --include-h5. The .h5
# (and, under --light, .pt) excludes are listed FIRST so they win before any
# --which include rule. NOTE: --include-h5 must ALSO drop the 200M size cap, else
# big datasets (~1.4 GB) are silently dropped by --max-size even though the .h5
# exclude was lifted. --light = deliverables only (plots + small data + logs):
# skip both the datasets (*.h5) and the checkpoints (*.pt).
if [ "$INCLUDE_H5" = false ]; then
    RSYNC_FILTERS=(--max-size=200M --exclude='*.h5')
    [ "$LIGHT" = true ] && RSYNC_FILTERS+=(--exclude='*.pt')
else
    RSYNC_FILTERS=()
fi

echo "=== Fetching results from Trillium ==="
if [ "$DATA_ONLY" = true ]; then
    echo "  data-only: true   (only *.h5 — no pdfs/json/checkpoints; no size cap)"
elif [ "$LIGHT" = true ]; then
    echo "  light: true   (deliverables only — skip *.h5 datasets and *.pt checkpoints)"
elif [ "$INCLUDE_H5" = false ]; then
    echo "  include .h5: false    max file size: 200M (override w/ --include-h5)"
else
    echo "  include .h5: true     max file size: unlimited"
fi

if [ ${#WHICH_TOKENS[@]} -gt 0 ]; then
    # Substring-filter the whole tree: descend into every dir (--include='*/'),
    # keep only files under a dir whose name contains ANY token (the '*/***' tail
    # matches that dir and everything beneath it), drop the rest, and prune the
    # empty dir skeletons left behind (-m). Under --data-only, keep ONLY *.h5
    # files sitting in a token-matching dir. Multiple tokens are OR-matched.
    echo "  which: ${WHICH_TOKENS[*]}  (substring match on dir names; any-of, recursive)"
    WHICH_INC=()
    for tok in "${WHICH_TOKENS[@]}"; do
        if [ "$DATA_ONLY" = true ]; then
            WHICH_INC+=(--include="*${tok}*/*.h5")
        else
            WHICH_INC+=(--include="*${tok}*/***")
        fi
    done
    rsync -avzm "${RSYNC_FILTERS[@]}" \
        --include='*/' \
        "${WHICH_INC[@]}" \
        --exclude='*' \
        "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/" \
        "${LOCAL_DIR}/results/fermionic_pipeline/"
elif [ -n "$TAG" ]; then
    echo "  tag: $TAG"
    if [ "$DATA_ONLY" = true ]; then
        rsync -avzm "${RSYNC_FILTERS[@]}" \
            --include='*/' --include='*.h5' --exclude='*' \
            "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/${TAG}/" \
            "${LOCAL_DIR}/results/fermionic_pipeline/${TAG}/"
    else
        rsync -avz "${RSYNC_FILTERS[@]}" \
            "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/${TAG}/" \
            "${LOCAL_DIR}/results/fermionic_pipeline/${TAG}/"
    fi
else
    echo "  all tags"
    if [ "$DATA_ONLY" = true ]; then
        rsync -avzm "${RSYNC_FILTERS[@]}" \
            --include='*/' --include='*.h5' --exclude='*' \
            "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/" \
            "${LOCAL_DIR}/results/fermionic_pipeline/"
    else
        rsync -avz "${RSYNC_FILTERS[@]}" \
            "${REMOTE}:${REMOTE_DIR}/results/fermionic_pipeline/" \
            "${LOCAL_DIR}/results/fermionic_pipeline/"
    fi
fi

if [ "$FETCH_LOGS" = true ]; then
    echo ""
    echo "=== Fetching logs ==="
    [ ${#WHICH_TOKENS[@]} -gt 0 ] && echo "  which: ${WHICH_TOKENS[*]} (only log files containing any token)"
    mkdir -p "${LOCAL_DIR}/logs"
    BEFORE=$(find "${LOCAL_DIR}/logs/" -maxdepth 1 -type f | wc -l | tr -d ' ')
    # Per-prefix include patterns; when --which is set, also require a token (any-of).
    LOG_INCLUDES=()
    for prefix in ferm-pipeline_ exact_gen_ clf_ eval_ reg_; do
        if [ ${#WHICH_TOKENS[@]} -gt 0 ]; then
            for tok in "${WHICH_TOKENS[@]}"; do
                LOG_INCLUDES+=(--include="${prefix}*${tok}*")
            done
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
