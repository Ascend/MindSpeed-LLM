# Setting
# set cann envirment
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# install MindSpeed
git clone -b master https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed
git checkout master
pip install -r requirements.txt
pip3 install -e .
cd ..

git clone -b master https://gitcode.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
chmod 777 -R ./
pip install -r requirements.txt

# megatron core_v0.12.1
cp -rf /home/master_branch/Megatron-LM/megatron ./

# define dirs
export TEST_MODEL=true
BASE_DIR=$(dirname "$(readlink -f "$0")")
CURRENT_TIME=$(date "+%Y-%m-%d")

# ============================
# Parse filter arguments
# ============================
FILTER_KEYWORD=""
FILTER_PATTERN=""
RUN_PIPELINE_ST=false
RUN_PIPELINE_UT=false
RUN_ST=false
RUN_UT=false
HAS_TYPE_FLAG=false

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -k, --keyword <str>     Only run test cases matching keyword (like pytest -k)"
    echo "  -f, --filter <pattern>  Only run test files matching glob pattern"
    echo "  --pipeline-st           Run pipeline ST tests (combinable)"
    echo "  --pipeline-ut           Run pipeline UT tests (combinable)"
    echo "  --st                    Run ST tests (combinable)"
    echo "  --ut                    Run UT tests (combinable)"
    echo "  Default: run all if no type flag specified."
    echo "  -h, --help              Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--keyword)
            FILTER_KEYWORD="$2"
            shift 2
            ;;
        -f|--filter)
            FILTER_PATTERN="$2"
            shift 2
            ;;
        --pipeline-st)
            RUN_PIPELINE_ST=true; HAS_TYPE_FLAG=true
            shift
            ;;
        --pipeline-ut)
            RUN_PIPELINE_UT=true; HAS_TYPE_FLAG=true
            shift
            ;;
        --st)
            RUN_ST=true; HAS_TYPE_FLAG=true
            shift
            ;;
        --ut)
            RUN_UT=true; HAS_TYPE_FLAG=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# If no type flag specified, run all
if [[ "$HAS_TYPE_FLAG" == false ]]; then
    RUN_PIPELINE_ST=true
    RUN_PIPELINE_UT=true
    RUN_ST=true
    RUN_UT=true
fi
GENERATE_LOG_BASE_DIR="/$(echo "$BASE_DIR" | cut -d'/' -f2)/pipeline_log_v2"
GENERATE_LOG_DIR="$GENERATE_LOG_BASE_DIR/$CURRENT_TIME"

PIPELINE_DIR="$BASE_DIR/pipeline"
PIPELINE_ST_DIR="$PIPELINE_DIR/st"
PIPELINE_ST_BASELINE_DIR="$PIPELINE_DIR/st/baseline"
PIPELINE_UT_DIR="$PIPELINE_DIR/ut"

UT_DIR="$BASE_DIR/ut"
ST_DIR="$BASE_DIR/st"
ST_BASELINE_DIR="$BASE_DIR/st/baseline_results"

#mkdir cache to store product and will be removed after test
mkdir -p "$GENERATE_LOG_DIR"
touch "$GENERATE_LOG_DIR/exec_error.log"
chmod a+w "$GENERATE_LOG_DIR/exec_error.log"


# coverage config
coverage_config() {
    COVERAGE_DIR="$GENERATE_LOG_DIR/coverage"
    SOURCE_DIR=$(realpath "$BASE_DIR/../mindspeed_llm")
    mkdir -p "$COVERAGE_DIR"

    rm -f .coverage
    rm -f .coverage*
    cat > ".coveragerc" << EOF
[run]
branch = False
parallel = False
source = $SOURCE_DIR

[report]
show_missing = True
skip_covered = False

exclude_lines =
    pragma: no cover
    ^\s*import\s
    ^\s*from\s
EOF
}

# run coverage task only on Tuesday/Thursday/Saturday
WEEK_DAY=$(date +%u)
if [[ $WEEK_DAY == 2 || $WEEK_DAY == 4 || $WEEK_DAY == 6 ]]; then
    echo "Testcase Execution Results With Coverage" > $GENERATE_LOG_DIR/exec_error.log
    export START_COVERAGE=true
else
    echo "Testcase Execution Results Without Coverage" > $GENERATE_LOG_DIR/exec_error.log
    export START_COVERAGE=false
fi
if [[ $START_COVERAGE == true ]]; then
    echo "setting coverage config"
    coverage_config
fi


echo "===========================================Test Results=====================================================" >> $GENERATE_LOG_DIR/exec_error.log
# run pipeline st testcase
if [[ "$RUN_PIPELINE_ST" == true ]]; then
    find "$PIPELINE_ST_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
        if [ -d "$dir" ]; then
            find "$dir" -type f -name "*.sh" | while read -r file; do
                filename=$(basename "$file")
                extension="${filename##*.}"
                name="${filename%.$extension}"
                # filter by pattern
                if [[ -n "$FILTER_PATTERN" ]] && [[ ! "$file" =~ $FILTER_PATTERN ]]; then
                    continue
                fi
                if [[ -n "$FILTER_KEYWORD" ]] && [[ ! "$name" =~ $FILTER_KEYWORD ]]; then
                    continue
                fi
                echo "running $file"
                bash $file 2>&1 | tee "$GENERATE_LOG_DIR/[PIPELINE_ST]$name.log"
                SCRIPT_EXITCODE=${PIPESTATUS[0]}
                if [ $SCRIPT_EXITCODE -eq 0 ]; then
                    # begin to execute the logic of compare
                    echo "$BASE_DIR/test_tools/test_ci_st.py"
                    pytest -x $BASE_DIR/test_tools/test_ci_st.py \
                        --baseline-json $PIPELINE_ST_BASELINE_DIR/$name.json \
                        --generate-log $GENERATE_LOG_DIR/[PIPELINE_ST]$name.log \
                        --generate-json $GENERATE_LOG_DIR/$name.json
                    PYTEST_EXITCODE=$?
                    if [ $PYTEST_EXITCODE -ne 0 ]; then
                        echo "[PIPELINE_ST]${name}.sh compare to baseline has failed, check it!" >> $GENERATE_LOG_DIR/exec_error.log
                    fi
                else
                    echo "[PIPELINE_ST]${name}.sh Script has failed. Exit!" >> $GENERATE_LOG_DIR/exec_error.log
                fi
            done
        fi
    done
fi

# run pipeline ut testcase
if [[ "$RUN_PIPELINE_UT" == true ]]; then
    find "$PIPELINE_UT_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
        if [ -d "$dir" ]; then
            find "$dir" -type f -name "*.py" | while read -r file; do
                tmp_file_name="${file#*MindSpeed-LLM/}"
                file_name="${tmp_file_name//\//_}"
                # filter by pattern
                if [[ -n "$FILTER_PATTERN" ]] && [[ ! "$file" =~ $FILTER_PATTERN ]]; then
                    continue
                fi
                if [[ -n "$FILTER_KEYWORD" ]] && [[ ! "$file_name" =~ $FILTER_KEYWORD ]]; then
                    continue
                fi
                echo "running $file"
                pytest --log-level=INFO "$file" 2>&1 | tee "${GENERATE_LOG_DIR}/[PIPELINE_UT]${file_name}.log"
                PYTEST_EXITCODE=${PIPESTATUS[0]}
                if [ $PYTEST_EXITCODE -ne 0 ]; then
                    echo "[PIPELINE_UT]$file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
                fi
                coverage run -p --source=$SOURCE_DIR $file
            done
        fi
    done
fi

# run st testcase
if [[ "$RUN_ST" == true ]]; then
    find "$ST_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
        if [ -d "$dir" ]; then
            find "$dir" -type f -name "*.sh" | while read -r file; do
                filename=$(basename "$file")
                extension="${filename##*.}"
                name="${filename%.$extension}"
                # filter by pattern
                if [[ -n "$FILTER_PATTERN" ]] && [[ ! "$file" =~ $FILTER_PATTERN ]]; then
                    continue
                fi
                if [[ -n "$FILTER_KEYWORD" ]] && [[ ! "$name" =~ $FILTER_KEYWORD ]]; then
                    continue
                fi
                echo "running $file"
                bash $file 2>&1 | tee "$GENERATE_LOG_DIR/[ST]$name.log"
                SCRIPT_EXITCODE=${PIPESTATUS[0]}
                if [ $SCRIPT_EXITCODE -eq 0 ]; then
                    # begin to execute the logic of compare
                    echo "$BASE_DIR/test_tools/test_ci_st.py"
                    pytest -x $BASE_DIR/test_tools/test_ci_st.py \
                        --baseline-json $ST_BASELINE_DIR/$name.json \
                        --generate-log $GENERATE_LOG_DIR/[ST]$name.log \
                        --generate-json $GENERATE_LOG_DIR/$name.json
                    PYTEST_EXITCODE=$?
                    if [ $PYTEST_EXITCODE -ne 0 ]; then
                        echo "[ST]${name}.sh compare to baseline has failed, check it!" >> $GENERATE_LOG_DIR/exec_error.log
                    fi
                else
                    echo "[ST]${name}.sh Script has failed. Exit!" >> $GENERATE_LOG_DIR/exec_error.log
                fi
            done
        fi
    done
fi

# run ut testcase
if [[ "$RUN_UT" == true ]]; then
    find "$UT_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
        if [ -d "$dir" ]; then
            find "$dir" -type f -name "*.py" | while read -r file; do
                tmp_file_name="${file#*MindSpeed-LLM/}"
                file_name="${tmp_file_name//\//_}"
                # filter by pattern
                if [[ -n "$FILTER_PATTERN" ]] && [[ ! "$file" =~ $FILTER_PATTERN ]]; then
                    continue
                fi
                if [[ -n "$FILTER_KEYWORD" ]] && [[ ! "$file_name" =~ $FILTER_KEYWORD ]]; then
                    continue
                fi
                echo "running $file"
                pytest --log-level=INFO "$file" 2>&1 | tee "${GENERATE_LOG_DIR}/[UT]${file_name}.log"
                PYTEST_EXITCODE=${PIPESTATUS[0]}
                if [ $PYTEST_EXITCODE -ne 0 ]; then
                    echo "[UT]$file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
                fi
                coverage run -p --source=$SOURCE_DIR $file
            done
        fi
    done
fi

# generate the coverage report
if [[ $START_COVERAGE == true ]]; then
    coverage combine
    coverage html -d "$COVERAGE_DIR/htmlcov"
    coverage xml -o "$COVERAGE_DIR/coverage.xml"
    coverage json -o "$COVERAGE_DIR/coverage.json"
    coverage_output=$(coverage report --format=total 2>&1)
    exit_code=$?
    echo "==========================================Coverage Results=====================================================" >> $GENERATE_LOG_DIR/exec_error.log
    if [ $exit_code -ne 0 ]; then
        echo "Coverage report failed!: $coverage_output">> $GENERATE_LOG_DIR/exec_error.log
    else
        echo "Pipeline Coverage Percentage is $coverage_output%" >> $GENERATE_LOG_DIR/exec_error.log
        echo "For detailed information, please refer to ./coverage/html/index.html" >> $GENERATE_LOG_DIR/exec_error.log
    fi
fi

rm -rf /data/ci/cache/*
echo "=================tar error log=================="
tar -czvf "${GENERATE_LOG_BASE_DIR}/${CURRENT_TIME}.tar.gz" "${GENERATE_LOG_DIR}/"
