#!/bin/bash
export OPENAI_API_KEY=""
export EVAL_OPENAI_API_KEY=""
export EVAL_MODEL=""
export OPENAI_BACKEND="openai"

# Configuration parameters
NUM_RUNS=3
BENCHMARK_NAME="mmbc"
LLM_PROVIDER="openai"
AGENT_SET="lmm-searcher"
MAX_CONTEXT_LENGTH=262144
MAX_CONCURRENT=10
PASS_AT_K=1
TEMPERATURE=1.0
API_KEY=""
BASE_URL=""
LLM="seed-1.8"
LLM_MODEL=""

# Set results directory
RESULTS_DIR="../../logs/${BENCHMARK_NAME}/${LLM_PROVIDER}_${LLM}_${AGENT_SET}"

echo "Starting $NUM_RUNS runs of the evaluation..."
echo "Results will be saved in: $RESULTS_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Launch all parallel tasks
for i in $(seq 1 $NUM_RUNS); do
    echo "=========================================="
    echo "Launching experiment $i/$NUM_RUNS"
    echo "Output log: please view $RESULTS_DIR/run_${i}_output.log"
    echo "=========================================="
    
    # Set specific identifier for this run
    RUN_ID="run_$i"
    
    # Run experiment (background execution)
    (
        uv run python benchmarks/common_benchmark.py \
            benchmark=$BENCHMARK_NAME \
            llm=$LLM \
            llm.provider=$LLM_PROVIDER \
            llm.model_name=$LLM_MODEL \
            llm.base_url=$BASE_URL \
            llm.async_client=true \
            llm.temperature=$TEMPERATURE \
            llm.max_context_length=$MAX_CONTEXT_LENGTH \
            llm.api_key=$API_KEY \
            benchmark.execution.max_tasks=null \
            benchmark.execution.max_concurrent=$MAX_CONCURRENT \
            benchmark.execution.pass_at_k=$PASS_AT_K \
            agent=$AGENT_SET \
            hydra.run.dir=${RESULTS_DIR}/$RUN_ID \
            2>&1 | tee "$RESULTS_DIR/${RUN_ID}_output.log" 
        
        # Check if run was successful
        if [ $? -eq 0 ]; then
            echo "Run $i completed successfully"
            RESULT_FILE=$(find "${RESULTS_DIR}/$RUN_ID" -name "*accuracy.txt" 2>/dev/null | head -1)
            if [ -f "$RESULT_FILE" ]; then
                echo "Results saved to $RESULT_FILE"
            else
                echo "Warning: Result file not found for run $i"
            fi
        else
            echo "Run $i failed!"
        fi
    ) &
    
    # Small delay between launches to avoid simultaneous requests
    sleep 2
done

echo "All $NUM_RUNS runs have been launched in parallel"
echo "Waiting for all runs to complete..."

# Wait for all background tasks to complete
wait

echo "=========================================="
echo "All $NUM_RUNS runs completed!"
echo "=========================================="

# Calculate average scores
echo "Calculating average scores..."
uv run python benchmarks/evaluators/calculate_average_score.py "$RESULTS_DIR"

echo "=========================================="
echo "Multiple runs evaluation completed!"
echo "Check results in: $RESULTS_DIR"
echo "Check individual run logs: $RESULTS_DIR/run_*_output.log"
echo "==========================================" 
