#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

IPs=('your ranks ip')
NPUS_PER_NODE=8
MASTER_ADDR=${IPs[0]} 
MASTER_PORT=6000
NNODES=32
NODE_RANK=""
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

for i in "${!IPs[@]}";
do
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
#       echo "${IPs[$i]}"
        echo "Node Rank : ${i}"
        NODE_RANK=$i
        break
    fi
done

if [[ $NODE_RANK == "" ]];then
    echo "[Error] para \"NODE_RANK\" must be confing"
    exit 1
fi

CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"

TP=1
PP=8
CP=8
EP=32
NUM_LAYERS=64
CP_TYPE='megatron_cp_algo'
SEQ_LEN=32768
MBS=1
GBS=64

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 160 \  
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall \
    --moe-alltoall-overlap-comm \
    --moe-router-topk 5 \
    --moe-permutation-async-comm
"

GPT_ARGS="
    --use-mcore-models \
    --no-shared-storage \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --log-throughput \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-distributed-optimizer \
    --disable-bias-linear \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
    --expert-model-parallel-size ${EP} \    
    --lr-warmup-fraction 0.01 \
    --swap-attention \
    --recompute-num-layers 8 \
    --enable-recompute-layers-per-pp-rank \
    --use-fused-ring-attention-update \
    --use-fused-moe-token-permute-and-unpermute \
    --recompute-in-advance \
    --bf16
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --save ${CKPT_SAVE_DIR}
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS} \
    ${MOE_ARGS} \
    ${CKPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --distributed-backend nccl \
    | tee logs/pretrain_deepseek_500b_32k_256die_mcore_A2.log
