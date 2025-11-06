# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
export PYTHONPATH=$BASE_DIR:$PYTHONPATH

SHELL_SCRIPTS_DIR="$BASE_DIR/shell_scripts"
export BASELINE_DIR="$BASE_DIR/baseline_results"
export EXEC_PY_DIR=$(dirname "$BASE_DIR")

GENERATE_LOG_DIR=/data/ci/run_logs
GENERATE_JSON_DIR=/data/ci/run_jsons

rm -rf $GENERATE_LOG_DIR/*
rm -rf $GENERATE_JSON_DIR/*

# error flag
export ERROR_FLAG="/tmp/ci_test.error"
rm -f "$ERROR_FLAG"


# step 2: running scripts of 8 NPUs and execute `test_ci_pipeline.py`
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
    ) > /tmp/$file_name_prefix.log 2>&1
    cat /tmp/$file_name_prefix.log
    rm -f /tmp/$file_name_prefix.log
'


# step 3: running scripts of 4 NPUs and execute `test_ci_pipeline.py`
NPU_GROUP_A=(0 1 2 3)
NPU_GROUP_B=(4 5 6 7)

export LOCKFILE_A="/tmp/npu_group_a.lock"
export LOCKFILE_B="/tmp/npu_group_b.lock"

MAX_PARALLEL=2

find "$SHELL_SCRIPTS_DIR" -name "*.sh" \
    -exec grep -qE "(NPUS_PER_NODE|NPUS_PER_NODE)=4" {} \; \
    -print | xargs -n 1 -P $MAX_PARALLEL -I {} bash -c '

    if [[ -f "$ERROR_FLAG" ]]; then
        exit 0
    fi

    test_case={}
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")

    # Use file lock to select available NPU group
    acquire_npu_group() {
        # flock -n: non-blocking, -w 10: wait 10min
        if flock -n -w 600 200; then
            echo "A"
            return 0
        fi

        # try to get group B
        if flock -n -w 600 201; then
            echo "B"
            return 0
        fi

        echo "No available NPU group"
        return 1
    }

    # create file descriptor for lock file
    exec 200>"$LOCKFILE_A"
    exec 201>"$LOCKFILE_B"

    # get NPU group
    if ! available_group=$(acquire_npu_group); then
        echo "Failed to acquire NPU group"
        touch "$ERROR_FLAG"
        exit 1
    fi

    # set NPU according to the assigned group
    case "$available_group" in
        "A") selected_npus=(0 1 2 3) ;;
        "B") selected_npus=(4 5 6 7) ;;
        *) touch "$ERROR_FLAG"; exit 1 ;;
    esac

    export ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${selected_npus[*]}")

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
    ) > /tmp/$file_name_prefix.log 2>&1
    cat /tmp/$file_name_prefix.log
    rm -f /tmp/$file_name_prefix.log
'


# step 4: running scripts of 2 NPUs and execute `test_ci_pipeline.py`
NPU_GROUP_A=(0 1)
NPU_GROUP_B=(2 3)
NPU_GROUP_C=(4 5)
NPU_GROUP_D=(6 7)

export LOCKFILE_A="/tmp/npu_group_a.lock"
export LOCKFILE_B="/tmp/npu_group_b.lock"
export LOCKFILE_C="/tmp/npu_group_c.lock"
export LOCKFILE_D="/tmp/npu_group_d.lock"

MAX_PARALLEL=4

find "$SHELL_SCRIPTS_DIR" -name "*.sh" \
    -exec grep -qE "(NPUS_PER_NODE|NPUS_PER_NODE)=(2|1)" {} \; \
    -print | xargs -n 1 -P $MAX_PARALLEL -I {} bash -c '

    if [[ -f "$ERROR_FLAG" ]]; then
        exit 0
    fi

    test_case={}
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    file_name_prefix=$(basename "${file_name%.*}")

    # Use file lock to select available NPU group
    acquire_npu_group() {
        # flock -n: non-blocking, -w 10: wait 10min
        if flock -n -w 600 200; then
            echo "A"
            return 0
        fi

        # try to get group B
        if flock -n -w 600 201; then
            echo "B"
            return 0
        fi

        # try to get group C
        if flock -n -w 600 202; then
            echo "C"
            return 0
        fi

        # try to get group D
        if flock -n -w 600 203; then
            echo "D"
            return 0
        fi

        echo "No available NPU group"
        return 1
    }

    # create file descriptor for lock file
    exec 200>"$LOCKFILE_A"
    exec 201>"$LOCKFILE_B"
    exec 202>"$LOCKFILE_C"
    exec 203>"$LOCKFILE_D"

    # get NPU group
    if ! available_group=$(acquire_npu_group); then
        echo "Failed to acquire NPU group"
        touch "$ERROR_FLAG"
        exit 1
    fi

    # set NPU according to the assigned group
    case "$available_group" in
        "A") selected_npus=(0 1) ;;
        "B") selected_npus=(2 3) ;;
        "C") selected_npus=(4 5) ;;
        "D") selected_npus=(6 7) ;;
        *) touch "$ERROR_FLAG"; exit 1 ;;
    esac

    export ASCEND_RT_VISIBLE_DEVICES=$(IFS=,; echo "${selected_npus[*]}")

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
    ) > /tmp/$file_name_prefix.log 2>&1
    cat /tmp/$file_name_prefix.log
    rm -f /tmp/$file_name_prefix.log
'

if [[ -f "$ERROR_FLAG" ]]; then
    echo "Some tests failed! Kill parallel processes..."
    pkill -f python
    exit 1
else
    echo "All tests passed!"
    exit 0
fi