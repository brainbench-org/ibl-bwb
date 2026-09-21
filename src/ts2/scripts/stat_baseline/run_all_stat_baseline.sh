#!/usr/bin/env bash
# Runs on: any machine (plain bash, no Slurm)
# Run one TS2 statistical baseline LOCALLY, without slurm, for every eval
# recording id, sequentially on the current GPU. The models are zero-parameter
# (everything is fit in link_datasets), so a full sweep is cheap.
#
# Each method is its own hydra trainer config, so METHOD selects the model and
# also lands in wandb via hydra_choices.
#   co_smoothing : mean_rate | pop_coupling | rrr | rrr_isi
#   forecasting  : mean_rate | trailing_mean | shrinkage | ridge_ar
# A method run on the task it does not apply to degenerates to the per-unit mean.
#
# Env overrides:
#   METHOD     baseline to run (default: rrr)
#   TASK       task to run (default: co_smoothing)
#   PROJECT    wandb project (default: ts2-stat-baseline)
#   DATA_ROOT  processed data root (default: from .env)
#   EIDS_FILE  newline-separated recording ids (default: eval_recording_ids.txt)
#   LOGDIR     per-session log directory (default: logs/ts2/stat_baseline/<method>)
# Any extra args are passed through to train.py (e.g. model.rank=8).
#
# Usage:
#   # reduced-rank regression, co_smoothing (defaults)
#   ./run_all_stat_baseline.sh
#   # the per-unit mean floor
#   METHOD=mean_rate ./run_all_stat_baseline.sh
#   # ridge autoregression on forecasting
#   METHOD=ridge_ar TASK=forecasting ./run_all_stat_baseline.sh
#   # a build other than the one .env points at
#   DATA_ROOT=/path/to/processed/selected_units ./run_all_stat_baseline.sh
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO" || exit 1

# Run directly, without src/setup.sh, so .env has to be loaded by hand.
if [[ -f .env ]]; then set -a; source .env; set +a; fi

METHOD="${METHOD:-rrr}"
TASK="${TASK:-co_smoothing}"
PROJECT="${PROJECT:-ts2-stat-baseline}"
# data_root comes from the hydra config, which reads BWB_DATA_ROOT_SELECTED_UNITS -> BWB_DATA_ROOT
# out of .env; DATA_ROOT here overrides both.
DATA_ROOT_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then DATA_ROOT_ARGS=(data_root="$DATA_ROOT"); fi
EIDS_FILE="${EIDS_FILE:-$REPO/src/ibl_bwb_eval/data/eval_recording_ids.txt}"
LOGDIR="${LOGDIR:-$REPO/logs/ts2/stat_baseline/$METHOD}"
mkdir -p "$LOGDIR"

mapfile -t EIDS < <(grep -vE '^[[:space:]]*(#|$)' "$EIDS_FILE")
n=${#EIDS[@]}
echo "Running $n sessions | method=$METHOD task=$TASK project=$PROJECT"
echo "data_root=${DATA_ROOT:-${BWB_DATA_ROOT_SELECTED_UNITS:-${BWB_DATA_ROOT:-<unset>}}}"
echo "Extra train.py args: $*"
echo "Per-session logs in: $LOGDIR"

pass=0; fail=0; failed_eids=()
i=0
for eid in "${EIDS[@]}"; do
    i=$((i+1))
    log="$LOGDIR/${eid}.log"
    echo "=== [$i/$n] $(date +%H:%M:%S) eid=$eid -> $log ==="
    if python src/ts2/train.py \
            trainer="$METHOD" \
            "${DATA_ROOT_ARGS[@]}" \
            task="$TASK" \
            recording_id="$eid" \
            wandb.project="$PROJECT" \
            "$@" \
            > "$log" 2>&1; then
        pass=$((pass+1)); echo "  OK"
    else
        fail=$((fail+1)); failed_eids+=("$eid"); echo "  FAILED (see $log)"
    fi
done

echo "=== done: $pass ok, $fail failed ==="
if (( fail )); then printf 'FAILED eid: %s\n' "${failed_eids[@]}"; fi
exit $(( fail > 0 ))
