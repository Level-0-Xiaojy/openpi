#!/bin/bash
# Run one experiment cell via `claude -p` (Claude Code subscription).
#
# Workflow:
#   1. Clean a per-cell workdir, start repl_driver.py in background.
#   2. Wait until state_00.json appears (Pi0.5 load ~90s).
#   3. Substitute the agent_task_prompt.md template with experiment vars.
#   4. Invoke `claude -p ...` with that prompt and Bash/Read/Write allowed.
#   5. Send {"action":"exit"} to the driver, kill it, return.
#
# Usage:
#   bash run_one_cell.sh <experiment_name> [--config pi05_x2robot] [--checkpoint-dir /path/to/ckpt]
#
# Idempotent: skips if audit already exists.
# Outputs: $OUTPUT_DIR/{recipe_<exp>.jsonl, <exp>.json, claude_<exp>.txt}

set -e

EXPERIMENT=${1:?Usage: $0 <experiment_name>}
CONFIG=${CONFIG:-pi05_x2robot}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-}
TCP_SERVER=${TCP_SERVER:-}  # set to "1" for server mode
TCP_IP=${TCP_IP:-192.168.77.58}
TCP_PORT=${TCP_PORT:-57770}
MODEL=${MODEL:-sonnet}
MAX_TURNS=${MAX_TURNS:-60}
CELL_TIMEOUT_S=${CELL_TIMEOUT_S:-600}
MAX_BUDGET_USD=${MAX_BUDGET_USD:-10}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MEMORY_DIR="${MEMORY_DIR:-$OPENPI_ROOT/workspace/memory_snapshot}"

WORKDIR_ROOT=${WORKDIR_ROOT:-/tmp}
WORKDIR=${WORKDIR_ROOT}/hybrid_repl_${EXPERIMENT}
OUTPUT_DIR=${OUTPUT_DIR:-$OPENPI_ROOT/workspace/results}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-$SCRIPT_DIR/agent_task_prompt.md}
DRIVER_LOG=/tmp/cc_driver_${EXPERIMENT}.log

mkdir -p "$OUTPUT_DIR"

# === 0. skip if audit exists ===
if [ -f "$OUTPUT_DIR/${EXPERIMENT}.json" ]; then
    echo "[$(date +%T)] [$EXPERIMENT] SKIP: audit already exists"
    exit 0
fi

# === 1. start driver ===
echo "[$(date +%T)] [$EXPERIMENT] starting driver"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

DRIVER_CMD=(
    "$OPENPI_ROOT/.venv/bin/python"
    -m openpi.primitives.repl_driver
    --config "$CONFIG"
    --workdir "$WORKDIR"
    --max-steps "$MAX_TURNS"
    --tcp-ip "$TCP_IP"
    --tcp-port "$TCP_PORT"
)
if [ -n "$CHECKPOINT_DIR" ]; then
    DRIVER_CMD+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [ "$TCP_SERVER" = "1" ]; then
    DRIVER_CMD+=(--tcp-server)
fi

"${DRIVER_CMD[@]}" > "$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!

# === 2. wait for driver ready ===
echo "[$(date +%T)] [$EXPERIMENT] driver pid=$DRIVER_PID; waiting for state_00.json ..."
T0=$(date +%s)
while [ ! -f "$WORKDIR/state_00.json" ]; do
    sleep 5
    if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
        echo "[$(date +%T)] [$EXPERIMENT] driver died before ready"
        tail -50 "$DRIVER_LOG"
        exit 2
    fi
    if [ $(( $(date +%s) - T0 )) -gt 300 ]; then
        echo "[$(date +%T)] [$EXPERIMENT] driver not ready after 300s"
        kill -9 "$DRIVER_PID" 2>/dev/null || true
        exit 3
    fi
done
echo "[$(date +%T)] [$EXPERIMENT] driver ready in $(( $(date +%s) - T0 ))s"

# === 3. build prompt ===
PROMPT_FILE=$(mktemp /tmp/cc_prompt_${EXPERIMENT}.XXXXXX.md)
sed \
    -e "s|{EXPERIMENT}|$EXPERIMENT|g" \
    -e "s|{WORKDIR}|$WORKDIR|g" \
    -e "s|{OUTPUT_DIR}|$OUTPUT_DIR|g" \
    "$PROMPT_TEMPLATE" > "$PROMPT_FILE"

# === 4. run claude -p ===
echo "[$(date +%T)] [$EXPERIMENT] invoking claude -p (model=$MODEL)"
CLAUDE_OUT="$OUTPUT_DIR/claude_${EXPERIMENT}.txt"
T_CLAUDE=$(date +%s)

cd "$OPENPI_ROOT"
set +e
timeout --kill-after=15 "$CELL_TIMEOUT_S" \
    claude -p "$(cat "$PROMPT_FILE")" \
        --model "$MODEL" \
        --output-format text \
        --add-dir "$WORKDIR" \
        --add-dir "$MEMORY_DIR" \
        --allowedTools "Bash Read Write Glob Grep" \
        --max-budget-usd "$MAX_BUDGET_USD" \
        > "$CLAUDE_OUT" 2>&1
CC_RC=$?
set -e
if [ "$CC_RC" = 124 ]; then
    echo "[$(date +%T)] [$EXPERIMENT] claude -p TIMEOUT after ${CELL_TIMEOUT_S}s"
fi
echo "[$(date +%T)] [$EXPERIMENT] claude -p finished in $(( $(date +%s) - T_CLAUDE ))s rc=$CC_RC"
rm -f "$PROMPT_FILE"

# === 5. stop driver ===
echo '{"action": "exit"}' > "$WORKDIR/command.json"
sleep 3
kill -9 "$DRIVER_PID" 2>/dev/null || true

# === 6. report ===
AUDIT="$OUTPUT_DIR/${EXPERIMENT}.json"
if [ -f "$AUDIT" ]; then
    echo "[$(date +%T)] [$EXPERIMENT] DONE  audit=$AUDIT"
else
    echo "[$(date +%T)] [$EXPERIMENT] NO AUDIT"
fi
