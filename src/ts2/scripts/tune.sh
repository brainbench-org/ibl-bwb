#!/bin/bash
# Runs on: any machine (plain bash, no Slurm)

# No venv path: setup.sh resolves VENV_DIR from .env, else /tmp/<user>/<checkout>/venv.
source "src/setup.sh"

# start ray head
NUM_CPUS=$(nproc)
NUM_GPUS="${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}"

RAY_TMP_DIR="${RAY_TMPDIR:-/tmp/$(id -un)/$(basename "$PWD")/ray}"
RAY_OUTPUT=$(ray start --head --port=0 --num-cpus=$NUM_CPUS --num-gpus=$NUM_GPUS --temp-dir="$RAY_TMP_DIR")
export RAY_ADDRESS=$(echo "$RAY_OUTPUT" | grep -oP "(?<=--address=\').*(?=\')")
if [ -z "$RAY_ADDRESS" ]; then
    echo "[tune.sh] [ERROR] Failed to parse RAY_ADDRESS. Ray may not have started correctly."
    exit 1
fi
echo "[tune.sh] [INFO] RAY_ADDRESS=$RAY_ADDRESS"
echo "[tune.sh] [INFO] NUM_CPUS=$NUM_CPUS"
echo "[tune.sh] [INFO] NUM_GPUS=$NUM_GPUS"

# gpu padder
GPU_PAD_TARGET=${GPU_PAD_TARGET:-0.4}
PADDER_PIDS=()
mkdir -p gpu_padder
for (( i=0; i<NUM_GPUS; i++ )); do
    python src/core/utils/gpu_padder.py --target "$GPU_PAD_TARGET" --gpu "$i" \
        >> "gpu_padder/gpu_padder_gpu${i}.log" 2>&1 &
    PADDER_PIDS+=($!)
    echo "[tune.sh] [INFO] Started GPU padder for gpu=$i (pid=${PADDER_PIDS[-1]}, target=${GPU_PAD_TARGET})"
done

cleanup() {
    echo "[tune.sh] [INFO] Stopping GPU padder(s)..."
    for pid in "${PADDER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    echo "[tune.sh] [INFO] Stopping Ray..."
    ray stop
}
trap cleanup EXIT

# extract task from args if provided (e.g. task=foo), otherwise run all tasks
TASK=$(echo "$@" | grep -oP '(?<=task=)\S+' || true)

echo "[tune.sh] [INFO] Parsed args: $@"
echo "[tune.sh] [INFO] Parsed task: ${TASK:-<all tasks>}"

if [ -n "$TASK" ]; then
    # If TASK env var is set, just tune that single task
    echo "[tune.sh] [INFO] Tuning Linear for task: $TASK"
    python src/ts2/tune.py "$@"
else
    # Otherwise, run over all tasks
    for task in $(python -c 'from ibl_bwb_eval.tasks import SUITE_TASKS; print(" ".join(SUITE_TASKS["ts2"]))'); do
        echo "[tune.sh] [INFO] Tuning Linear for task: $task"
        python src/ts2/tune.py "$@" task="$task" &
    done
    echo "[tune.sh] [INFO] Waiting for all tasks to complete..."
    wait
    echo "[tune.sh] [INFO] All tasks completed"
fi
