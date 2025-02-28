export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
ulimit -n 32768

NNODES=2
NPUS_PER_NODE=8

# 主节点启动
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

while true; do
    ray_status_output=$(ray status)
    npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
    npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
    device_count=$((npu_count_int / $NPUS_PER_NODE))

    # 判断 device_count 是否与 NNODES 相等
    if [ "$device_count" -eq "$NNODES" ]; then
        echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
        ray status
        python ray_gpt.py --config-name grpo_trainer_qwen25_7b | tee logs/r1_zero_qwen25_7b_full.log
        break
    else
        echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
        sleep 5
    fi
done
