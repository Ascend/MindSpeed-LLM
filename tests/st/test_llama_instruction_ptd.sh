# This is an example: train llama using PTD,
# the number of parameters is not aligned

export LD_LIBRARY_PATH=/usr/local:/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WITHOUT_JIT_COMPILE=1
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/ckpt

rm -rf $CHECKPOINT_PATH

mkdir $CHECKPOINT_PATH
DATA_PATH=/data/tune_dataset/alpaca
TOKENIZER_PATH=/data/llama-2-7b-hf/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
basepath=$(cd `dirname $0`; cd ../../; pwd)
export PYTHONPATH=${basepath}:$PYTHONPATH

python3.8 -m torch.distributed.launch $DISTRIBUTED_ARGS \
      ${basepath}/pretrain_gpt.py \
      --tensor-model-parallel-size 2 \
      --pipeline-model-parallel-size 2 \
      --sequence-parallel \
      --num-layers 4 \
      --hidden-size 4096 \
      --ffn-hidden-size 11008 \
      --num-attention-heads 32 \
      --tokenizer-type PretrainedFromHF  \
      --tokenizer-name-or-path $TOKENIZER_PATH \
      --tokenizer-not-use-fast \
      --seq-length 4096 \
      --max-position-embeddings 4096 \
      --micro-batch-size 1 \
      --global-batch-size 16 \
      --make-vocab-size-divisible-by 1 \
      --lr 1.25e-6 \
      --train-iters 5 \
      --lr-decay-style cosine \
      --untie-embeddings-and-output-weights \
      --disable-bias-linear \
      --attention-dropout 0.0 \
      --init-method-std 0.01 \
      --hidden-dropout 0.0 \
      --position-embedding-type rope \
      --normalization RMSNorm \
      --swiglu \
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
      --use-distributed-optimizer \
      --no-load-optim \
      --no-load-rng \
      --fp16 \
      --finetune \
      --is-instruction-dataset \
      --position-embedding-type alibi \
      --square-alibi-mask \
      --distributed-backend nccl \
      --data-path $DATA_PATH \
      --split 90,5,5 \
      --log-interval 1 \
      --eval-interval 1000 \
      --use-deter-comp \
      --eval-iters 1

exit $?
