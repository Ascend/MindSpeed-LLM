# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

SHELL_SCRIPTS_DIR="$BASE_DIR/shell_scripts"
export BASELINE_DIR="$BASE_DIR/baseline_results"
export EXEC_PY_DIR=$(dirname "$BASE_DIR")

export GENERATE_LOG_DIR=$BASE_DIR/run_ms_logs
export GENERATE_JSON_DIR=$BASE_DIR/run_ms_jsons
mkdir -p $GENERATE_LOG_DIR
mkdir -p $GENERATE_JSON_DIR

rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*

# error flag
export ERROR_FLAG="$BASE_DIR/ci_ms_test.error"
rm -f "$ERROR_FLAG"

# step 2: enable deterministic computation and insert modification points
MindSpeed_LLM_PATH=$BASE_DIR/../../../../MindSpeed-LLM
Megatron_LM_PATH=$BASE_DIR/../../../../Megatron-LM
addbias() {
    fname=$1
    lineNum=$(grep -n 'config.perform_initialization' ${fname} | cut -d: -f1)
    sed -i $((lineNum))'i\ \ \ \ \ \ \ \ self.bias = torch.zeros((self.config.num_moe_experts), dtype=torch.bfloat16)' $fname
}
addSeedAll() {
    fname=$1
    lineNumMain=$(grep -n '__main__' ${fname} | cut -d: -f1)
    echo deterministic
    sed -i $((lineNumMain + 1))'i\ \ \ \ seed_all()' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch_npu.npu.manual_seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch_npu.npu.manual_seed_all(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch.use_deterministic_algorithms(True)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ torch.manual_seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ np.random.seed(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ os.environ["PYTHONHASHSEED"] = str(seed)' $fname
    sed -i $((lineNumMain - 1))'i\ \ \ \ random.seed(seed)' $fname
    sed -i $((lineNumMain - 1))'idef seed_all(seed=42):' $fname
    sed -i $((lineNumMain - 1))'iimport torch_npu' $fname
    sed -i $((lineNumMain - 1))'iimport torch' $fname
    sed -i $((lineNumMain - 1))'iimport numpy as np' $fname
    sed -i $((lineNumMain - 1))'iimport random' $fname
    sed -i $((lineNumMain - 1))'iimport os' $fname
}
modifyTrainingLogs() {
    fname=$1
    echo "Modifying training log precision..."
    # replace log_string += ' {}: {:.6E} |'.format(key, avg)
    sed -i 's/log_string += '\'' {}: {:.6E} |'\''.format(key, avg)/log_string += '\'' {}: {:.16f} |'\''.format(key, avg)/g' "$fname"
    # replace log_string += ' grad norm: {:.3f} |'.format(grad_norm)
    sed -i 's/log_string += '\'' grad norm: {:.3f} |'\''.format(grad_norm)/log_string += '\'' grad norm: {:.16f} |'\''.format(grad_norm)/g' "$fname"
    # replace log_string += ' params norm: {:.3f} |'.format(params_norm)
    sed -i 's/log_string += '\'' params norm: {:.3f} |'\''.format(params_norm)/log_string += '\'' params norm: {:.16f} |'\''.format(params_norm)/g' "$fname"
    echo "Log precision has been updated to 16 decimal places in $fname"
}
modifyTrainingLogs ${Megatron_LM_PATH}/megatron/training/training.py
addSeedAll ${MindSpeed_LLM_PATH}/pretrain_gpt.py
addSeedAll ${MindSpeed_LLM_PATH}/posttrain_gpt.py
#sed -i 's/\ \ \ \ \ \ \ \ logits = F.linear(input, self.weight)/\ \ \ \ \ \ \ \ logits = F.linear(input, self.weight, self.bias)/g' ${MindSpeed_LLM_PATH}/mindspeed_llm/core/transformer/moe/router.py
#addbias ${Megatron_LM_PATH}/megatron/core/transformer/moe/router.py
#sed -i 's/        device=freqs.device, dtype=torch.float32/        dtype=torch.float32/g' ${MindSpeed_LLM_PATH}/mindspeed_llm/core/models/common/embeddings/rotary_pos_embedding.py
export HCCL_DETERMINISTIC=true  
export ASCEND_LAUNCH_BLOCKING=1  
export NCCL_DETERMINISTIC=1

# step 3: running scripts of 8 NPUs and execute `test_ci_pipeline.py`
MAX_PARALLEL=1

find "$SHELL_SCRIPTS_DIR" -name "*.sh" \
    ! -exec grep -qE "(NPUS_PER_NODE|NPUS_PER_NODE)=(4|2|1)" {} \; \
    -print | xargs -n 1 -P $MAX_PARALLEL -I {} bash -c '

    if [[ -f "$ERROR_FLAG" ]]; then
        exit 0
    fi

    test_case={}
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")
    echo "$file_name_prefix"

    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    echo "$file_name_prefix using NPUs: $ASCEND_RT_VISIBLE_DEVICES"

    # create empty JSON file to receive results parsed from log
    touch "$GENERATE_JSON_DIR/$file_name_prefix.json"

    (
        echo "Log of $file_name_prefix:"
        
        # if executing shell script fails, exit directly without comparison
        bash $test_case | tee "$GENERATE_LOG_DIR/$file_name_prefix.log"
        SCRIPT_EXITCODE=${PIPESTATUS[0]}
        if [ $SCRIPT_EXITCODE -ne 0 ]; then
            echo "Training $file_name_prefix has failed. Exit!"
            touch "$ERROR_FLAG"
            exit 1
        fi
        # begin to execute the logic of compare
        pytest -x $EXEC_PY_DIR/test_tools/test_ci_st.py \
            --baseline-json $BASELINE_DIR/$file_name_prefix.json \
            --generate-log $GENERATE_LOG_DIR/$file_name_prefix.log \
            --generate-json $GENERATE_JSON_DIR/$file_name_prefix.json
        PYTEST_EXITCODE=$?
        if [ $PYTEST_EXITCODE -ne 0 ]; then
            echo "$file_name_prefix compare to baseline has failed, check it!"
            touch "$ERROR_FLAG"
            exit 1
        else
            echo "Pretrain $file_name_prefix execution success."
        fi
    ) > $BASE_DIR/$file_name_prefix.log 2>&1
    cat $BASE_DIR/$file_name_prefix.log
    rm -f $BASE_DIR/$file_name_prefix.log
'


if [[ -f "$ERROR_FLAG" ]]; then
    echo "Some tests failed! Kill parallel processes..."
    pkill -f python
    exit 1
else
    echo "All tests passed!"
    exit 0
fi