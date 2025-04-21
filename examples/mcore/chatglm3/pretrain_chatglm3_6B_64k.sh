#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Please enter ip of your server.
IPs=('IP1' 'IP2')
LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`
echo LOCAL_HOST $LOCAL_HOST

NPUS_PER_NODE=8
MASTER_ADDR=${IPs[0]}
MASTER_PORT=6010
NNODES=${#IPs[@]}
NODE_RANK=""
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

for i in "${!IPs[@]}";
do
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ];
    then
        echo "Node Rank : ${i}"
        NODE_RANK=$i
        break
    fi
done


CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=1
CP=16
MBS=1
GBS=64
SEQ_LEN=65536
CP_ALGO=hybrid_cp_algo

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 28 \
    --hidden-size 4096 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 32 \
    --ulysses-degree-in-cp 8 \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --context-parallel-algo ${CP_ALGO} \
    --context-parallel-size ${CP} \
    --max-position-embeddings ${SEQ_LEN} \
    --padded-vocab-size 65024 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --no-rope-fusion \
    --use-distributed-optimizer \
    --use-glm-rope \
    --rotary-percent 0.5 \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --normalization RMSNorm \
    --swiglu \
    --no-create-attention-mask-in-dataloader \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 1e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1e-8 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 512 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --fp16 \
    --num-workers 1 \
    --kv-head-repeat-before-uly-alltoall \
    --no-shared-storage \
    --use-cp-send-recv-overlap \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 10 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
"


python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/train_mcore_chatglm3_6B_64K.log
