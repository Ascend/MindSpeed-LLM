set -x
device_ids=${1-"0"}
max_memory=${2-35}
mode=${3-run}
tgi_options="MAX_MEMORY_GB=${max_memory} ASCEND_RT_VISIBLE_DEVICES=${device_ids} USE_HOST_CHOOSER=0"
atb_options="HCCL_BUFFSIZE=110 ATB_LAYER_INTERNAL_TENSOR_REUSE=1  ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1"
atb_options_beta="ATB_USE_TILING_COPY_STREAM=1" # 多stream 仅Atalas推理系列产品可用，功能不稳定
atb_options_async="TASK_QUEUE_ENABLE=1 ATB_OPERATION_EXECUTE_ASYNC=1" # 异步，收益不高，可能有细微的提升
env_options="PATH=$PATH:/root/.cargo/bin LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/wqh39/lib/"
lccl_options="BACKEND='lccl'" # Atlas 900 A2 PODc多卡限定
debug_options="LOG_LEVEL='debug' ATB_LOG_TO_STDOUT=1 ATB_LOG_LEVEL=INFO TASK_QUEUE_ENABLE=0 ASDOPS_LOG_TO_STDOUT=1 ASDOPS_LOG_LEVEL=INFO ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=1"
cmd="${tgi_options}  ${atb_options} ${env_options} text-generation-launcher --model-id ./ --port 8089"
if [ "x$mode" == "xdebug" ];then
cmd="${debug_options} ${cmd}"
fi
eval $cmd