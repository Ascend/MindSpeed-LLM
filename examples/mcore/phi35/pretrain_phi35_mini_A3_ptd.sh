#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# distributed config
NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
LONG_FACTOR="1.0800000429153442,1.1100000143051147,1.1399999856948853,1.340000033378601,1.5899999141693115,1.600000023841858,1.6200000047683716,2.620000123977661,3.2300000190734863,3.2300000190734863,4.789999961853027,7.400000095367432,7.700000286102295,9.09000015258789,12.199999809265137,17.670000076293945,24.46000099182129,28.57000160217285,30.420001983642578,30.840002059936523,32.590003967285156,32.93000411987305,42.320003509521484,44.96000289916992,50.340003967285156,50.45000457763672,57.55000305175781,57.93000411987305,58.21000289916992,60.1400032043457,62.61000442504883,62.62000274658203,62.71000289916992,63.1400032043457,63.1400032043457,63.77000427246094,63.93000411987305,63.96000289916992,63.970001220703125,64.02999877929688,64.06999969482422,64.08000183105469,64.12000274658203,64.41000366210938,64.4800033569336,64.51000213623047,64.52999877929688,64.83999633789062"
SHORT_FACTOR="1.0,1.0199999809265137,1.0299999713897705,1.0299999713897705,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0499999523162842,1.0699999332427979,1.0999999046325684,1.1099998950958252,1.1599998474121094,1.1599998474121094,1.1699998378753662,1.2899998426437378,1.339999794960022,1.679999828338623,1.7899998426437378,1.8199998140335083,1.8499997854232788,1.8799997568130493,1.9099997282028198,1.9399996995925903,1.9899996519088745,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0199997425079346,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0299997329711914,2.0799996852874756,2.0899996757507324,2.189999580383301,2.2199995517730713,2.5899994373321533,2.729999542236328,2.749999523162842,2.8399994373321533"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \
    --rope-scaling-type longrope \
    --longrope-freqs-type outer \
    --rope-scaling-original-max-position-embeddings 4096 \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --swiglu \
    --num-layers 32 \
    --hidden-size 3072 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --micro-batch-size 2 \
    --global-batch-size 128 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --use-flash-attn \
    --rotary-base 10000 \
    --use-distributed-optimizer \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --padded-vocab-size 32064 \
    --untie-embeddings-and-output-weights \
    --seed 42 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --long-factor ${LONG_FACTOR} \
    --short-factor ${SHORT_FACTOR} \
    | tee logs/train_phi35_mini_mcore.log