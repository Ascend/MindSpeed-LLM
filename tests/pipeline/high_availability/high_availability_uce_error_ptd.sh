#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TTP_LOG_STDOUT=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="/data/ci/cache/temp-checkpoint-high-availability/"
DATA_PATH="/data/ci/datasets/processed/dataset-high-availability/llama_text_document"
TOKENIZER_MODEL="/data/ci/models/llama2/hf/llama-high-availability/tokenizer.model"
CKPT_LOAD_DIR="/data/ci/cache/temp-checkpoint-high-availability/"
TP=2
PP=1

basepath=$(cd `dirname $0`; cd ../../../; pwd)

training_file=$basepath/mindspeed_llm/training/training.py
stop_clean_file=$basepath/mindspeed_llm/core/high_availability/tft_stop_clean.py

cp $training_file $training_file.back
cp $stop_clean_file $stop_clean_file.back


sed -i '/def model_provider_func_wrapper/i\
GLB_CNT = 0\
def raise_uce_error(iteration):\
    global GLB_CNT\
    import os\
    cur_rank = torch.distributed.get_rank()\
    if iteration == 10 and GLB_CNT < 1:\
        GLB_CNT = GLB_CNT + 1\
        if cur_rank == 1:\
            print(f"############# rank:{cur_rank} start UCE ERROR")\
            raise RuntimeError("UCE ERROR")\
        else:\
            print(f"############# rank:{cur_rank} start FORCE STOP")\
            raise RuntimeError("FORCE STOP")' $training_file

sed -i '/args\.curr_iteration = iteration/a\
        raise_uce_error(iteration)' $training_file

sed -i 's/check_memory_result = torch_npu.npu.check_uce_in_memory(device)/check_memory_result = ha_constant.UCE_HIGH_LEVEL/g' $stop_clean_file

sed -i '/if clean_type == "retry":/,/torch.npu.restart_device(device)/s/^/#/' $stop_clean_file

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 4 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 12 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 20 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \

    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --enable-high-availability \
    --enable-hbmfault-repair \
    --use-distributed-optimizer \
    --bf16 \
    --overlap-grad-reduce
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --log-throughput \
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl

mv $training_file.back $training_file
mv $stop_clean_file.back $stop_clean_file


