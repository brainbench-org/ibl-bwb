#!/bin/bash
# Runs on: any machine (plain bash, no Slurm)

RUN_DIAGNOSTICS=0
VENV_ARG=""

for arg in "$@"; do
    if [ "$arg" = "--diagnostics" ]; then
        RUN_DIAGNOSTICS=1
    elif [ "${arg#--}" != "$arg" ]; then
        continue
    elif [ -z "$VENV_ARG" ]; then
        VENV_ARG="$arg"
    fi
done

# Sourced before the venv is chosen, so a VENV_DIR set here takes effect.
if [ -f ".env" ]; then
    set -a; source .env; set +a
    echo "✅ Loaded .env"
fi

# Derived from the checkout, not from the repo name, so a rename does not strand the venv.
VENV_DIR=${VENV_ARG:-${VENV_DIR:-/tmp/$(id -un)/$(basename "$PWD")/venv}}

# Export UV_NO_CACHE=1 to skip uv's wheel cache, which matters on a small home quota.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    # create venv if it doesn't exist
    uv venv "$VENV_DIR" -p python3.10 --seed
    source "$VENV_DIR/bin/activate"
    uv pip install -e ".[train]"
else
    # otherwise, create and install any missing deps
    source "$VENV_DIR/bin/activate"
    uv pip install -e ".[train]"
fi

echo
echo "✅ Environment setup complete! ($VENV_DIR)"
echo

if [ "$RUN_DIAGNOSTICS" -eq 1 ]; then
    # run diagnostic on xformers + FlashAttention (bf16)
    $VENV_DIR/bin/python "src/core/utils/check_xformers_fa.py"
fi