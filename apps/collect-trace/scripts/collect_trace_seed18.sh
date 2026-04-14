# Check if OPENAI_API_KEY is set
export OPENAI_API_KEY=""
export EVAL_OPENAI_API_KEY=""
export EVAL_OPENAI_BASE_URL=""
export EVAL_MODEL=""

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set."
    exit 1
else
    echo "OPENAI_API_KEY detected."
fi

# Get the directory where the current script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Current script directory: $SCRIPT_DIR"

DATASET=mmbc  # This script is only used for demo, we do not collect trace for training on mmbc
LLM=seed-1.8
AGENT=lmm-searcher_turn40
# Enter the apps/miroflow-agent directory
TARGET_DIR="$SCRIPT_DIR/../../miroflow-agent"
echo "Target directory: $TARGET_DIR"
cd $TARGET_DIR

mkdir -p ../../logs
LOG_DIR="../../logs/collect_trace_${AGENT}_${DATASET}_boxed_pass4_${LLM}"
echo "Log directory: $LOG_DIR"
mkdir -p $LOG_DIR

# Collect traces
uv run python benchmarks/common_benchmark.py \
    benchmark=$DATASET \
    llm=$LLM \
    llm.async_client=true \
    benchmark.execution.max_tasks=null \
    benchmark.execution.max_concurrent=25 \
    benchmark.execution.pass_at_k=4 \
    agent=$AGENT \
    hydra.run.dir=$LOG_DIR \
    2>&1 | tee "$LOG_DIR/output.log"

# Enter the apps/collect-trace directory
TARGET_DIR="$SCRIPT_DIR/../"
echo "Target directory: $TARGET_DIR"
cd $TARGET_DIR

# Process traces
uv run python $TARGET_DIR/utils/process_logs.py $LOG_DIR/benchmark_results.jsonl


