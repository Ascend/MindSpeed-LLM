# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

SHELL_SCRIPTS_DIR="$BASE_DIR/shell_scripts"
BASELINE_DIR="$BASE_DIR/baseline_results"
EXEC_PY_DIR=$(dirname "$BASE_DIR")

GENERATE_LOG_DIR=/data/ci/run_logs
GENERATE_JSON_DIR=/data/ci/run_jsons

rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*

# 删缓存 预编译，提高用例执行稳定性
rm -rf /root/.cache
python -c "import mindspeed; from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GMMV2OpBuilder; GMMV2OpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MatmulAddOpBuilder; MatmulAddOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MoeTokenPermuteOpBuilder; MoeTokenPermuteOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MoeTokenUnpermuteOpBuilder; MoeTokenUnpermuteOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder; RotaryPositionEmbeddingOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GroupMatmulAddOpBuilder; GroupMatmulAddOpBuilder().load()"


# step 2: running scripts and execute `test_ci_pipeline.py`
for test_case in "$SHELL_SCRIPTS_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")
    echo "$file_name_prefix"

    # create empty json file to receive the result parsered from log
    touch "$GENERATE_JSON_DIR/$file_name_prefix.json"

    # if executing the shell has failed, then just exit, no need to compare.
    bash $test_case | tee "$GENERATE_LOG_DIR/$file_name_prefix.log"
    SCRIPT_EXITCODE=${PIPESTATUS[0]}
    if [ $SCRIPT_EXITCODE -ne 0 ]; then
        echo "Script has failed. Exit!"
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
        exit 1
    else
        echo "Pretrain $file_name_prefix execution success."
    fi

done

