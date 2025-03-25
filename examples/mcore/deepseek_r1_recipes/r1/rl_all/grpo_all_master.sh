export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 主节点初始化ray
ray start --head --port 6344 --dashboard-host=0.0.0.0 --dashboard-port=8260 --resources='{"NPU": 8}'

# 子节点全部注册上ray后，查看是否状态正常
ray status

# 启动训练
python ray_gpt.py --config-name grpo_trainer_qwen25_7b_all | tee logs/grpo_trainer_qwen25_7b_all.log

