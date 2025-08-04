#!/bin/bash
set -e

echo "=== GPU Bot Entrypoint Script ==="
echo "Reservation ID: $RESERVATION_ID"
echo "Reservation End Time: $RESERVATION_END_TIME"
echo "Container ID: ${CONTAINER_ID:-$VAST_CONTAINERLABEL}"

# Debug: Print all environment variables
echo "=== Environment Variables Debug ==="
env | grep -E "RESERVATION|VAST|CONTAINER" | sort
echo "=================================="

# Export RESERVATION_* variables to /etc/environment so they persist in SSH sessions
echo "Exporting RESERVATION_* variables to /etc/environment..."
env | grep ^RESERVATION_ >> /etc/environment || true
# Also export them for the current session
export $(env | grep ^RESERVATION_ | xargs) 2>/dev/null || true

# Install vast.ai CLI if not present
if ! command -v vastai &> /dev/null; then
    echo "Installing vast.ai CLI..."
    pip install vastai
fi

# Create self-termination script
echo "Creating self-termination script..."
cat > /root/self_terminate.sh << 'EOF'
#!/bin/bash
echo "$(date): Starting self-termination for reservation $RESERVATION_ID"

# Write termination status
CONTAINER="${CONTAINER_ID:-$VAST_CONTAINERLABEL}"
echo "{\"status\": \"terminating\", \"timestamp\": \"$(date --iso-8601=seconds)\", \"reservation_id\": \"$RESERVATION_ID\", \"container_id\": \"$CONTAINER\"}" > /root/instance_status.json

# Give 30 seconds for any cleanup
echo "Waiting 30 seconds for cleanup..."
sleep 30

# Terminate the instance
echo "Executing vastai destroy command..."
vastai destroy instance $CONTAINER

# Fallback if vastai command fails
if [ $? -ne 0 ]; then
    echo "vastai destroy failed, attempting poweroff"
    sudo poweroff
fi
EOF
chmod +x /root/self_terminate.sh

# Try to get RESERVATION_END_TIME from multiple sources
if [ -z "$RESERVATION_END_TIME" ]; then
    # Check if it's in a file (in case vast.ai writes env vars to a file)
    if [ -f "/etc/environment" ]; then
        source /etc/environment 2>/dev/null || true
    fi
    
    # Check if it's passed as a different env var name
    if [ -n "$EXTRA_ENV_RESERVATION_END_TIME" ]; then
        RESERVATION_END_TIME="$EXTRA_ENV_RESERVATION_END_TIME"
        echo "Found RESERVATION_END_TIME in EXTRA_ENV_RESERVATION_END_TIME"
    fi
fi

# Schedule termination
if [ -n "$RESERVATION_END_TIME" ]; then
    # Calculate minutes until termination (with 5 minute buffer)
    current_time=$(date +%s)
    end_time=$(date -d "$RESERVATION_END_TIME" +%s 2>/dev/null || echo 0)
    
    if [ $end_time -gt 0 ] && [ $end_time -gt $current_time ]; then
        minutes_until_end=$(( ($end_time - $current_time) / 60 + 5 ))
        
        echo "Scheduling termination for $RESERVATION_END_TIME (in $minutes_until_end minutes)"
        
        # Schedule using 'at' command
        if command -v at &> /dev/null; then
            echo "/root/self_terminate.sh" | at now + $minutes_until_end minutes 2>/dev/null || true
            echo "Scheduled with 'at' command"
        fi
        
        # Also add cron job as backup
        if command -v crontab &> /dev/null; then
            termination_time=$(date -d "$RESERVATION_END_TIME +5 minutes" +"%M %H %d %m")
            (crontab -l 2>/dev/null || true; echo "$termination_time * /root/self_terminate.sh") | crontab -
            echo "Added cron backup"
        fi
    else
        echo "WARNING: Invalid or past RESERVATION_END_TIME"
    fi
else
    echo "WARNING: No RESERVATION_END_TIME set - defaulting to 2 hour termination"
    # Default to 2 hours from now
    minutes_until_end=125  # 2 hours + 5 minute buffer
    
    echo "Scheduling default termination in $minutes_until_end minutes"
    
    # Schedule using 'at' command
    if command -v at &> /dev/null; then
        echo "/root/self_terminate.sh" | at now + $minutes_until_end minutes 2>/dev/null || true
        echo "Scheduled with 'at' command"
    fi
fi

# Create monitoring/heartbeat script
echo "Creating monitoring script..."
cat > /root/monitor.sh << 'EOF'
#!/bin/bash
while true; do
    # Write heartbeat
    CONTAINER="${CONTAINER_ID:-$VAST_CONTAINERLABEL}"
    echo "{\"status\": \"running\", \"timestamp\": \"$(date --iso-8601=seconds)\", \"reservation_id\": \"$RESERVATION_ID\", \"container_id\": \"$CONTAINER\", \"uptime\": \"$(uptime -p)\"}" > /root/instance_status.json
    
    # Check if past reservation end time
    if [ -n "$RESERVATION_END_TIME" ]; then
        current=$(date +%s)
        end=$(date -d "$RESERVATION_END_TIME" +%s 2>/dev/null || echo 0)
        if [ $end -gt 0 ] && [ $current -gt $end ]; then
            echo "Reservation time exceeded, triggering termination"
            /root/self_terminate.sh
            exit 0
        fi
    fi
    
    sleep 60
done
EOF
chmod +x /root/monitor.sh

# Start monitoring in background
echo "Starting monitoring process..."
nohup /root/monitor.sh > /var/log/reservation_monitor.log 2>&1 &

# Export vLLM configuration based on GPU count
echo "Configuring vLLM..."
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Detected $GPU_COUNT GPUs"

# Select model and configuration based on GPU count
if [ $GPU_COUNT -eq 8 ]; then
    echo "Using Qwen3-Coder-30B for 8x GPU configuration with tensor parallelism"
    export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    VLLM_ARGS=(
        --tensor-parallel-size 8
        --trust-remote-code
        --dtype float16
        --max-model-len 262144
        --gpu-memory-utilization 0.95
        --enable-chunked-prefill
        --enable-prefix-caching
        --api-key "${VLLM_API_KEY:-default-key}"
        --served-model-name qwen-coder
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
    )
elif [ $GPU_COUNT -eq 4 ]; then
    echo "Using Qwen3-Coder-30B for 4x GPU configuration with tensor parallelism"
    export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    VLLM_ARGS=(
        --tensor-parallel-size 4
        --dtype float16
        --max-model-len 262144
        --gpu-memory-utilization 0.95
        --max-num-batched-tokens 32768
        --enable-chunked-prefill
        --disable-log-requests
        --disable-custom-all-reduce
        --api-key "${VLLM_API_KEY:-default-key}"
        --served-model-name qwen-coder
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
    )
else
    echo "Using Qwen3-Coder-30B for 2x GPU configuration with tensor parallelism"
    export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    VLLM_ARGS=(
        --tensor-parallel-size 2
        --quantization fp8
        --kv-cache-dtype fp8
        --max-model-len 262144
        --rope-scaling '{"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":32768}'
        --gpu-memory-utilization 0.90
        --trust-remote-code
        --disable-custom-all-reduce
        --api-key "${VLLM_API_KEY:-default-key}"
        --served-model-name qwen-coder
        --enable-auto-tool-choice
        --tool-call-parser qwen3_coder
    )
fi

echo "Selected model: $VLLM_MODEL"
export RAY_ARGS="--head --port 6379 --dashboard-host 127.0.0.1 --dashboard-port 28265"
export PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8000:18000:/docs:vLLM API|localhost:8265:28265:/:Ray Dashboard|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:9090:19090:/metrics:Prometheus Metrics"

# List scheduled termination jobs
echo "=== Scheduled Termination Jobs ==="
if command -v atq &> /dev/null; then
    atq || echo "No 'at' jobs scheduled"
fi
if command -v crontab &> /dev/null; then
    crontab -l 2>/dev/null || echo "No cron jobs scheduled"
fi
echo "================================="

# Write initial status
echo "{\"status\": \"starting\", \"timestamp\": \"$(date --iso-8601=seconds)\", \"reservation_id\": \"$RESERVATION_ID\"}" > /root/instance_status.json

# Start vLLM directly (for vllm/vllm-openai image)
echo "Starting vLLM..."
cd /root

# Export environment variables to help with model download
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_MODELSCOPE=0
export HF_HUB_DISABLE_PROGRESS_BARS=0
# Use fork instead of spawn to avoid multiprocessing issues
export VLLM_WORKER_MULTIPROC_METHOD=fork
# Ensure vLLM uses the same cache directory as our pre-download
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface

# Environment setup for performance and stability
if [ $GPU_COUNT -eq 8 ]; then
    # Special environment setup for 8x GPU configuration
    export VLLM_SKIP_P2P_CHECK=1                         # Skip GPU peer-to-peer access check (avoids multi-GPU init errors)
    export VLLM_FLASH_ATTN_VERSION=2                     # Force FlashAttention v2 (H200/Blackwell support)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Use expandable segments to reduce CUDA fragmentation
    # Optional debugging flags (commented out by default)
    # export NCCL_ASYNC_ERROR_HANDLING=1                 # Enable async error handling in NCCL
    # export NCCL_P2P_DISABLE=1                          # Disable direct NCCL P2P if encountering NVLink issues
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7          # Explicitly set all 8 GPUs
elif [ $GPU_COUNT -eq 2 ]; then
    # Environment setup for 2x GPU with FP8
    export CUDA_LAUNCH_BLOCKING=1                        # Ensure correct sync for multi-GPU FP8
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"  # Optimize CUDA allocator to reduce fragmentation
    export VLLM_USE_TRITON_FLASH_ATTN=1                  # Use Triton-based FlashAttention for speed
    export VLLM_FLASH_ATTN_VERSION=3                     # Force FlashAttention-3 for long contexts
else
    # Default environment setup for 4x GPU
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Use expandable segments to reduce CUDA fragmentation
    export VLLM_FLASH_ATTN_VERSION=2                     # FlashAttention v2 for stability
fi

# Pre-download model if using tensor parallelism to avoid multiprocessing issues
if [[ "$VLLM_ARGS" == *"--tensor-parallel-size"* ]]; then
    echo "Tensor parallelism detected. Pre-downloading model to avoid multiprocessing conflicts..."
    echo "Using huggingface-cli for download (this may show high CPU usage but will complete)..."
    
    # Check if model already exists
    if huggingface-cli scan-cache --dir /root/.cache/huggingface | grep -q "$VLLM_MODEL"; then
        echo "Model $VLLM_MODEL already cached, skipping download"
    else
        echo "Downloading $VLLM_MODEL..."
        # Use hf (new command) or fall back to huggingface-cli
        if command -v hf &> /dev/null; then
            hf download "$VLLM_MODEL" --cache-dir /root/.cache/huggingface || {
                echo "Warning: Pre-download failed, continuing anyway..."
            }
        else
            huggingface-cli download "$VLLM_MODEL" --cache-dir /root/.cache/huggingface || {
                echo "Warning: Pre-download failed, continuing anyway..."
            }
        fi
    fi
fi

# Start vLLM with the configured model
if command -v vllm &> /dev/null; then
    echo "Starting vLLM with model: $VLLM_MODEL"
    echo "This may take several minutes to download the model on first run..."
    exec vllm serve "$VLLM_MODEL" \
        --host 0.0.0.0 \
        --port 18000 \
        "${VLLM_ARGS[@]}"
else
    echo "ERROR: vLLM command not found!"
    echo "This entrypoint script requires the vllm/vllm-openai Docker image"
    exit 1
fi