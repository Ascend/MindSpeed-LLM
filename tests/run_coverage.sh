# This script is used to run the coverage test for the pipeline, unit tests and shell scripts.

# 设定环境变量 用于部分代码段重构
export PYTHONPATH=$PYTHONPATH:${PWD}
export TEST_MODEL=true
export START_COVERAGE=true

# define the base directory
BASE_DIR=$(dirname $(dirname "$(readlink -f "$0")"))
SOURCE_DIR="$BASE_DIR/mindspeed_llm"
PIPELINE_DIR="empty"
UT_DIR="empty"
ST_DIR="empty"

# 创建日志目录
COVERAGE_DIR="COVERAGE"
rm -rf "$COVERAGE_DIR"
mkdir -p "$COVERAGE_DIR"
GENERATE_LOG_DIR="$COVERAGE_DIR/logs"
mkdir -p "$GENERATE_LOG_DIR"
touch "$GENERATE_LOG_DIR/exec_error.log"
echo "core0.12.1 Execution Results" > $GENERATE_LOG_DIR/exec_error.log

REPORT_DIR="$COVERAGE_DIR/report"
mkdir -p "$REPORT_DIR"


# 带参1用于区分运行场景
if [ -z "$1" ]; then
    echo "请提供一个参数（ST、PIPELINE、UT、all）"
    exit 1
fi

BRANCH_TEST=$1

if [ ${BRANCH_TEST} = "all" ]; then
    PIPELINE_DIR="$BASE_DIR/tests/pipeline"
    UT_DIR="$BASE_DIR/tests/ut"
    ST_DIR="$BASE_DIR/tests/st/shell_scripts"
elif  [ ${BRANCH_TEST} = "ST" ]; then
    ST_DIR="$BASE_DIR/tests/st/shell_scripts"
elif  [ ${BRANCH_TEST} = "PIPELINE" ]; then
    PIPELINE_DIR="$BASE_DIR/tests/pipeline"
elif  [ ${BRANCH_TEST} = "UT" ]; then
    UT_DIR="$BASE_DIR/tests/ut"
fi

# 带参2只用于ut，非必要
BRANCH_UT=$2
if [ -z "$2" ]; then
    echo "第二参量，BRANCH_UT，未提供，默认全量UT"
elif [ $BRANCH_UT != "all" ]; then
        UT_DIR="$BASE_DIR/tests/coverage/${BRANCH_UT}"
fi

echo "PIPELINE_DIR is ${PIPELINE_DIR}"
echo "UT_DIR is ${UT_DIR}"
echo "ST_DIR is ${ST_DIR}"

# remove the existing coverage files
rm -f .coverage
rm -f .coverage*

# create the coverage configuration file
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

# run the coverage for python files in the unit tests
find "$UT_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.py" | while read -r file; do
            echo "Running [UT] ${file}"
            filename=$(basename "$file")
            extension="${filename##*.}"
            name="${filename%.$extension}"
            pytest -s $file | tee "$GENERATE_LOG_DIR/[UT]$name.log" 2>&1
            PYTEST_EXITCODE=${PIPESTATUS[0]}
            if [ $PYTEST_EXITCODE -ne 0 ]; then
                echo "[UT] $file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
            fi
            coverage run -p --source=$SOURCE_DIR $file
        done
    fi
done

# run the coverage for shell scripts in the st
for test_case in "$ST_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running [ST] $file_name..."
    extension="${file_name##*.}"
    name="${file_name%.$extension}"
    bash $test_case | tee "$GENERATE_LOG_DIR/[ST]$name.log" 2>&1
    PYTEST_EXITCODE=${PIPESTATUS[0]}
    if [ $PYTEST_EXITCODE -ne 0 ]; then
        echo "[ST] $test_case has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
    fi
done

# run the coverage for python files in the pipeline
find "$PIPELINE_DIR/ut" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.py" | while read -r file; do
            echo "Running [PIPELINE_UT] ${file}"
            filename=$(basename "$file")
            extension="${filename##*.}"
            name="${filename%.$extension}"
            pytest -s $file | tee "$GENERATE_LOG_DIR/[PIPELINE_UT]$name.log" 2>&1
            PYTEST_EXITCODE=${PIPESTATUS[0]}
            if [ $PYTEST_EXITCODE -ne 0 ]; then
                echo "[PIPELINE_UT] $file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
            fi
            coverage run -p --source=$SOURCE_DIR $file
        done
    fi
done

# run the coverage for shell scripts in the pipeline
find "$PIPELINE_DIR/st" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.sh" | while read -r file; do
            echo "Running [PIPELINE_ST] ${file}"
            filename=$(basename "$file")
            extension="${filename##*.}"
            name="${filename%.$extension}"
            bash $file | tee "$GENERATE_LOG_DIR/[PIPELINE_ST]$name.log" 2>&1
            PYTEST_EXITCODE=${PIPESTATUS[0]}
            if [ $PYTEST_EXITCODE -ne 0 ]; then
                echo "[PIPELINE_ST] $file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
            fi
        done
    fi
done

# generate the coverage report
coverage combine
coverage html -d "$REPORT_DIR/htmlcov"
coverage xml -o "$REPORT_DIR/coverage.xml"
coverage json -o "$REPORT_DIR/coverage.json"

# 压缩目录
echo "Compressing directory '$COVERAGE_DIR'..."
tar -czf $COVERAGE_DIR.tgz $COVERAGE_DIR

# 检查压缩是否成功
if [ $? -eq 0 ]; then
    # 删除原目录
    echo "Removing original directory $COVERAGE_DIR ..."
    rm -rf $COVERAGE_DIR
else
    echo "Compression failed."
fi
