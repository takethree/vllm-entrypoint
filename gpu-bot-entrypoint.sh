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
    echo "Using Qwen3-Coder-30B (full BF16) for 8x GPU configuration"
    export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    # No need for DeepGEMM with BF16 model
    export VLLM_ARGS="--tensor-parallel-size 8 --max-model-len 131072 --enforce-eager --download-dir /workspace/models --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.9 --max-num-batched-tokens 16384 --max-num-seqs 256 --enable-prefix-caching --enable-chunked-prefill --api-key ${VLLM_API_KEY:-default-key} --served-model-name qwen-coder --enable-auto-tool-choice --tool-call-parser qwen3_coder"
elif [ $GPU_COUNT -eq 4 ]; then
    echo "Using Qwen3-Coder-30B-FP8 for 4x GPU configuration"
    # Enable DeepGEMM for FP8 models
    export VLLM_USE_DEEP_GEMM=1
    export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    export VLLM_ARGS="--tensor-parallel-size 4 --max-model-len 131072 --enforce-eager --download-dir /workspace/models --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192 --max-num-seqs 256 --enable-prefix-caching --enable-chunked-prefill --api-key ${VLLM_API_KEY:-default-key} --served-model-name qwen-coder --enable-auto-tool-choice --tool-call-parser qwen3_coder"
else
    echo "Using Qwen3-Coder-30B-FP8 for 2x GPU configuration"
    # Enable DeepGEMM for FP8 models
    export VLLM_USE_DEEP_GEMM=1
    export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
    export VLLM_ARGS="--tensor-parallel-size 2 --max-model-len 131072 --enforce-eager --download-dir /workspace/models --host 127.0.0.1 --port 18000 --gpu-memory-utilization 0.9 --max-num-batched-tokens 8192 --max-num-seqs 256 --enable-prefix-caching --enable-chunked-prefill --api-key ${VLLM_API_KEY:-default-key} --served-model-name qwen-coder --enable-auto-tool-choice --tool-call-parser qwen3_coder"
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

# Start vLLM with the configured model
if command -v vllm &> /dev/null; then
    echo "Starting vLLM with model: $VLLM_MODEL"
    echo "This may take several minutes to download the model on first run..."
    exec vllm serve "$VLLM_MODEL" \
        --host 0.0.0.0 \
        --port 8000 \
        $(echo $VLLM_ARGS)
else
    echo "ERROR: vLLM command not found!"
    echo "This entrypoint script requires the vllm/vllm-openai Docker image"
    exit 1
fi