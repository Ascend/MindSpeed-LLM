#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6153
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_LOAD_DIR="/data/ci/models/phi35/mg/phi35-moe-tp1pp8l8-mcore-base"
DATA_PATH="/data/ci/datasets/processed/phi35-moe-data/alpaca_text_document"
TOKENIZER_MODEL="/data/ci/models/phi35/hf/Phi-3.5-MoE-instruct-hf"

TP=1
PP=8
EP=1
MBS=1
GBS=64
SEQ_LEN=4096
NUM_LAYERS=8
TRAIN_ITERS=20

LONG_FACTOR="1.0199999809265137,1.0299999713897705,1.0399999618530273,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.059999942779541,1.059999942779541,1.059999942779541,1.059999942779541,1.059999942779541,1.059999942779541,1.0999999046325684,1.1799999475479126,1.1799999475479126,1.3700000047683716,1.4899998903274536,2.109999895095825,2.8899998664855957,3.9499998092651367,4.299999713897705,6.429999828338623,8.09000015258789,10.690000534057617,12.050000190734863,18.229999542236328,18.84000015258789,19.899999618530273,21.420000076293945,26.200000762939453,34.28000259399414,34.590003967285156,38.730003356933594,40.22000503540039,42.54000473022461,44.000003814697266,47.590003967285156,54.750003814697266,56.19000244140625,57.44000244140625,57.4900016784668,61.20000076293945,61.540000915527344,61.75,61.779998779296875,62.06999969482422,63.11000061035156,63.43000030517578,63.560001373291016,63.71000289916992,63.92000198364258,63.94000244140625,63.94000244140625,63.96000289916992,63.980003356933594,64.0300064086914,64.0300064086914,64.0300064086914,64.04000854492188,64.10000610351562,64.19000244140625,64.20999908447266,64.75,64.95999908447266"
SHORT_FACTOR="1.0,1.0399999618530273,1.0399999618530273,1.0399999618530273,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.059999942779541,1.059999942779541,1.0699999332427979,1.0699999332427979,1.0699999332427979,1.0699999332427979,1.1399999856948853,1.159999966621399,1.159999966621399,1.159999966621399,1.159999966621399,1.1799999475479126,1.1999999284744263,1.3199999332427979,1.3399999141693115,1.3499999046325684,1.3999998569488525,1.4799998998641968,1.4999998807907104,1.589999794960022,1.6499998569488525,1.71999990940094,1.8999998569488525,1.9099998474121094,1.9099998474121094,1.9899998903274536,1.9999998807907104,1.9999998807907104,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.009999990463257,2.0999999046325684,2.319999933242798,2.419999837875366,2.5899999141693115,2.7899999618530273"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --expert-model-parallel-size ${EP} \
    --num-experts 16 \
    --moe-router-topk 2 \
    --moe-grouped-gemm \
    --moe-router-load-balancing-type sparsemixer_topk \
    --moe-input-jitter-eps 0.01 \
    --moe-token-dispatcher-type allgather \
    --moe-permutation-async-comm \
"

GPT_ARGS="
    --transformer-impl local \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.phi35_moe_spec layer_spec \
    --no-rope-fusion \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 131072 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 6400 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --padded-vocab-size 32064 \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 10000 \
    --train-iters ${TRAIN_ITERS} \
    --add-qkv-bias \
    --disable-bias-linear \
    --add-dense-bias \
    --add-output-layer-bias \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --rope-scaling-type longrope \
    --longrope-freqs-type outer \
    --rope-scaling-original-max-position-embeddings 4096 \
    --long-mscale 1.243163121016122 \
    --short-mscale 1.243163121016122 \
    --long-factor ${LONG_FACTOR} \
    --short-factor ${SHORT_FACTOR} \
    --normalization LayerNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --seed 42 \
    --bf16
"

OPTIM_ARGS="
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --attention-softmax-in-fp32 \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --no-load-optim \
    --no-load-rng
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --min-lr 1.25e-7 \
    --lr-decay-style cosine \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --finetune \
    --log-throughput \
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $MOE_ARGS \
    $GPT_ARGS \
    $OPTIM_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR}