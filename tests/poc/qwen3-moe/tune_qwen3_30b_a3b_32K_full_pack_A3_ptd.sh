# 验证所使用数据集下载自 https://huggingface.co/datasets/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT/tree/main
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=6000
export HCCL_EXEC_TIMEOUT=5400
export HCCL_IF_BASE_PORT=48600
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1


NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model huggingface ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your raw data path"
TOKENIZER_PATH="your tokenizer path"

TP=2
PP=1
EP=16
CP=1
CP_TYPE='ulysses_cp_algo'
SEQ_LENGTH=32768
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --expert-tensor-parallel-size 1 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 768 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-permute-fusion \
    --moe-alltoall-overlap-comm \
    --moe-token-dispatcher-type alltoall \
    --moe-router-load-balancing-type aux_loss \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001 
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --swap-optimizer \
    --swap-attention \
    --gemm-gradient-accumulation-fusion \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --fix-router \
"

TRAIN_ARGS="
    --reset-attention-mask \
    --attention-mask-type general \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --lr 1.25e-5 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --manual-gc \
    --manual-gc-interval 50 \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
"


DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --no-shared-storage \
    --handler-name AlpacaStyleInstructionHandler \
    --prompt-type qwen3 \
"


OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --log-throughput \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
"


TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --neat-pack \
    --pack \
"


CKPT_ARGS="
    --enable-hf2mg-convert \
    --model-type-hf qwen3-moe \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $CKPT_ARGS \
    --distributed-backend nccl \
    | tee logs/tune_qwen3_30b_a3b_32K_full_pack_A3_ptd.log