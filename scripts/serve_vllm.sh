#!/bin/bash

# vLLM Serve Script for Qwen2.5 Models
# Usage: ./serve_vllm.sh [model_path] [port]

# Default values
MODEL_PATH="${1:-models/Qwen2.5-1.5B-Instruct-lora_16bit}"
PORT="${2:-8000}"
MAX_SEQ_LENGTH=8192
GPU_MEMORY_UTILIZATION=0.5  # Use 90% of GPU memory (adjust based on your needs)
TENSOR_PARALLEL_SIZE=1      # Number of GPUs for tensor parallelism
MAX_NUM_SEQS=64           # Max number of sequences processed in a batch
API_KEY="thanglq12@vpbank"  # API key for authentication

echo "Starting vLLM server with:"
echo "  Model: $MODEL_PATH"
echo "  Port: $PORT"
echo "  Max Sequence Length: $MAX_SEQ_LENGTH"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "  Max Batch Sequences: $MAX_NUM_SEQS"
echo "  Prefix Caching: Enabled"
echo "  API Key: ${API_KEY:0:10}***"

# Serve the model using vLLM
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_SEQ_LENGTH" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prefix-caching \
    --trust-remote-code \
    --dtype auto \
    --api-key "$API_KEY"
