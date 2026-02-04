# Common environment variables for the fsdp2 backend
export TRAINING_BACKEND=mindspeed_fsdp
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DEBUG=0