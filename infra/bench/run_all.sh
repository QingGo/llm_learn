#!/bin/bash
MODEL="Qwen/Qwen3.5-4B"
BASE_DIR="./bench_results"
mkdir -p $BASE_DIR

# 虚拟环境路径（根据实际情况调整）
SGLANG_VENV="../sglang/.venv"
VLLM_VENV="../vllm/.venv"

# 编译缓存目录（分别设置，避免冲突）
SGLANG_CACHE="/root/autodl-tmp/inductor_cache_sglang"
VLLM_CACHE="/root/autodl-tmp/inductor_cache_vllm"
mkdir -p $SGLANG_CACHE $VLLM_CACHE

# 压测通用参数
NUM_REQUESTS=200
CONCURRENCY=20          # 可自行调整并发数
INPUT_LEN=512
OUTPUT_LEN=128
SEED=42

# 测试函数
run_test() {
    FRAMEWORK=$1      # sglang 或 vllm
    MODE=$2           # single / dp / tp
    PORT=$3
    SERVE_CMD=$4      # 在对应虚拟环境中执行的启动命令
    LABEL="${FRAMEWORK}_${MODE}"

    echo "===== Starting ${LABEL} ====="
    echo "Command: $SERVE_CMD"

    # 设置环境
    if [ "$FRAMEWORK" == "sglang" ]; then
        VENV_PATH="$SGLANG_VENV"
        export TORCHINDUCTOR_CACHE_DIR="$SGLANG_CACHE"
    else
        VENV_PATH="$VLLM_VENV"
        export TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE"
    fi

    # 在子 shell 中激活虚拟环境，保持隔离
    (
        source "$VENV_PATH/bin/activate"
        echo "Activated $VENV_PATH"

        # 后台启动服务
        $SERVE_CMD &
        SERVER_PID=$!
        echo "Server PID: $SERVER_PID"

        # 等待服务就绪（最长 600 秒，若超时则退出并报错）
        echo "Waiting for server on port $PORT..."
        server_ready=0
        for i in $(seq 1 600); do
            if curl -s http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; then
                echo "Server ready after ${i}s"
                server_ready=1
                break
            fi
            sleep 1
        done

        if [ $server_ready -eq 0 ]; then
            echo "Error: Server did not become ready in 600s"
            kill $SERVER_PID 2>/dev/null
            wait $SERVER_PID 2>/dev/null
            exit 1
        fi

        # 执行压测
        URL="http://127.0.0.1:$PORT/v1/chat/completions"
        python bench_one.py \
            --url "$URL" \
            --label "$LABEL" \
            --output-file "${BASE_DIR}/${LABEL}.json" \
            --num-requests $NUM_REQUESTS \
            --concurrency $CONCURRENCY \
            --input-len $INPUT_LEN \
            --output-len $OUTPUT_LEN \
            --seed $SEED

        # 停止服务
        echo "Stopping server..."
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
    )

    # 等待显存完全释放
    sleep 5
}

# ===== SGLang 三种配置 =====
# 基准命令（对齐全套优化：torch.compile + 分段 CUDA Graph + 显存限制 + 上下文长度）
SGLANG_BASE="sglang serve --model-path $MODEL \
    --host 127.0.0.1 --port 30000 \
    --context-length 16384 \
    --mem-fraction-static 0.80 \
    --enable-torch-compile"

run_test "sglang" "single" 30000 "$SGLANG_BASE"
run_test "sglang" "dp"     30000 "$SGLANG_BASE --dp 2"
run_test "sglang" "tp"     30000 "$SGLANG_BASE --tp 2"

# ===== vLLM 三种配置 =====
# 基准命令（对齐全套优化：-O2 包含 torch.compile + FULL_AND_PIECEWISE CUDA Graph，显存与上下文长度对齐）
VLLM_BASE="vllm serve $MODEL \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.80 \
    -O2"

run_test "vllm" "single" 8000 "$VLLM_BASE"
run_test "vllm" "dp"     8000 "$VLLM_BASE --data-parallel-size 2"
run_test "vllm" "tp"     8000 "$VLLM_BASE --tensor-parallel-size 2"

echo "All tests completed!"
echo "Results saved in $BASE_DIR"