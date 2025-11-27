# This script is used to run the coverage test for the pipeline, unit tests and shell scripts.

# 设定环境变量 用于部分代码段重构
export PYTHONPATH=$PYTHONPATH:${PWD}
export TEST_MODEL=true

# define the base directory
BASE_DIR=$(dirname $(dirname "$(readlink -f "$0")"))
SOURCE_DIR="$BASE_DIR/mindspeed_llm"
PIPELINE_DIR="empty"
UT_DIR="empty"
ST_DIR="empty"

# 创建日志目录
GENERATE_LOG_DIR="$UT_DIR/logs"
mkdir -p "$GENERATE_LOG_DIR"
touch "$GENERATE_LOG_DIR/exec_error.log"
echo "core0.12.1 Execution Results" > $GENERATE_LOG_DIR/exec_error.log


# 带参1用于区分运行场景
if [ -z "$1" ]; then
    echo "请提供一个参数（ST、PIPELINE、UT、all）"
    exit 1
fi

BRANCH_TEST=$1

if [ ${BRANCH_TEST} = "all" ]; then
    PIPELINE_DIR="$BASE_DIR/tests/pipeline"
    UT_DIR="$BASE_DIR/tests/coverage"
    ST_DIR="$BASE_DIR/tests/st/shell_scripts"
elif  [ ${BRANCH_TEST} = "ST" ]; then
    ST_DIR="$BASE_DIR/tests/st/shell_scripts"
elif  [ ${BRANCH_TEST} = "PIPELINE" ]; then
    PIPELINE_DIR="$BASE_DIR/tests/pipeline"
elif  [ ${BRANCH_TEST} = "UT" ]; then
    UT_DIR="$BASE_DIR/tests/coverage"
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
rm -f .coverage.*
rm -rf htmlcov

# create the coverage configuration file
cat > ".coveragerc" << EOF
[run]
branch = False
parallel = False
source = $SOURCE_DIR

[report]
show_missing = True
skip_covered = False
EOF

add_coverage() {
    sed -i "1a\import random" pretrain_gpt.py
    sed -i "2a\import time" pretrain_gpt.py
    sed -i "3a\import coverage" pretrain_gpt.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' pretrain_gpt.py
    sed -i "5a\cov.start()" pretrain_gpt.py

    sed -i "/    main()/a\    cov.stop()" pretrain_gpt.py
    sed -i "/    cov.stop()/a\    cov.save()" pretrain_gpt.py

    sed -i "1a\import random" pretrain_mamba.py
    sed -i "2a\import time" pretrain_mamba.py
    sed -i "3a\import coverage" pretrain_mamba.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' pretrain_mamba.py
    sed -i "5a\cov.start()" pretrain_mamba.py

    sed -i "/    main()/a\    cov.stop()" pretrain_mamba.py
    sed -i "/    cov.stop()/a\    cov.save()" pretrain_mamba.py

    sed -i "1a\import random" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "2a\import time" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "3a\import coverage" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py

    sed -i "/    run()/i\    cov.start()" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "/    run()/a\    cov.stop()" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "/    cov.stop()/a\    cov.save()" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py

    sed -i "1a\import random" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "2a\import time" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "3a\import coverage" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' mindspeed_llm/tasks/checkpoint/convert_param.py

    sed -i "/    main()/i\    cov.start()" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "/    main()/a\    cov.stop()" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "/    cov.stop()/a\    cov.save()" mindspeed_llm/tasks/checkpoint/convert_param.py

    sed -i "1a\import random" convert_ckpt.py
    sed -i "2a\import time" convert_ckpt.py
    sed -i "3a\import coverage" convert_ckpt.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' convert_ckpt.py

    sed -i "/    main()/i\    cov.start()" convert_ckpt.py
    sed -i "/    main()/a\    cov.stop()" convert_ckpt.py
    sed -i "/    cov.stop()/a\    cov.save()" convert_ckpt.py

    sed -i "1a\import random" convert_ckpt_v2.py
    sed -i "2a\import time" convert_ckpt_v2.py
    sed -i "3a\import coverage" convert_ckpt_v2.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' convert_ckpt_v2.py

    sed -i "/    main()/i\    cov.start()" convert_ckpt_v2.py
    sed -i "/    main()/a\    cov.stop()" convert_ckpt_v2.py
    sed -i "/    cov.stop()/a\    cov.save()" convert_ckpt_v2.py

    sed -i "1a\import random" evaluation.py
    sed -i "2a\import time" evaluation.py
    sed -i "3a\import coverage" evaluation.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' evaluation.py

    sed -i "/def main():/a\    cov.start()" evaluation.py
    sed -i "/            logger.info(f'NeedleBench_eval Running Time: {time.time() - a}')/a\    cov.stop()" evaluation.py
    sed -i "/    cov.stop()/a\    cov.save()" evaluation.py

    sed -i "1a\import random" inference.py
    sed -i "2a\import time" inference.py
    sed -i "3a\import coverage" inference.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' inference.py

    sed -i "/def main():/a\    cov.start()" inference.py
    sed -i "/    task_factory(args, model)/a\    cov.stop()" inference.py
    sed -i "/    cov.stop()/a\    cov.save()" inference.py

    sed -i "1a\import random" posttrain_gpt.py
    sed -i "2a\import time" posttrain_gpt.py
    sed -i "3a\import coverage" posttrain_gpt.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' posttrain_gpt.py
    sed -i "5a\cov.start()" posttrain_gpt.py

    sed -i "/    launch()/a\    cov.stop()" posttrain_gpt.py
    sed -i "/    cov.stop()/a\    cov.save()" posttrain_gpt.py

    sed -i "1a\import random" preprocess_data.py
    sed -i "2a\import time" preprocess_data.py
    sed -i "3a\import coverage" preprocess_data.py
    sed -i '4a\cov = coverage.Coverage(data_suffix=f"usecase-{time.time_ns()}_{random.randint(0, 100)}")' preprocess_data.py

    sed -i "/def main():/a\    cov.start()" preprocess_data.py
    sed -i "/                os.remove(idx_file.replace('.idx', '.bin'))/a\    cov.stop()" preprocess_data.py
    sed -i "/    cov.stop()/a\    cov.save()" preprocess_data.py
}

remove_coverage() {
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py
    sed -i "2d" pretrain_gpt.py

    sed -i "/    cov.stop()/d" pretrain_gpt.py
    sed -i "/    cov.save()/d" pretrain_gpt.py

    sed -i "2d" pretrain_mamba.py
    sed -i "2d" pretrain_mamba.py
    sed -i "2d" pretrain_mamba.py
    sed -i "2d" pretrain_mamba.py
    sed -i "2d" pretrain_mamba.py

    sed -i "/    cov.stop()/d" pretrain_mamba.py
    sed -i "/    cov.save()/d" pretrain_mamba.py

    sed -i "2d" convert_ckpt.py
    sed -i "2d" convert_ckpt.py
    sed -i "2d" convert_ckpt.py
    sed -i "2d" convert_ckpt.py

    sed -i "/    cov.start()/d" convert_ckpt.py
    sed -i "/    cov.stop()/d" convert_ckpt.py
    sed -i "/    cov.save()/d" convert_ckpt.py

    sed -i "2d" convert_ckpt_v2.py
    sed -i "2d" convert_ckpt_v2.py
    sed -i "2d" convert_ckpt_v2.py
    sed -i "2d" convert_ckpt_v2.py

    sed -i "/    cov.start()/d" convert_ckpt_v2.py
    sed -i "/    cov.stop()/d" convert_ckpt_v2.py
    sed -i "/    cov.save()/d" convert_ckpt_v2.py

    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py

    sed -i "/    cov.start()/d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "/    cov.stop()/d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py
    sed -i "/    cov.save()/d" mindspeed_llm/tasks/checkpoint/convert_ckpt_mamba2.py

    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "2d" mindspeed_llm/tasks/checkpoint/convert_param.py

    sed -i "/    cov.start()/d" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "/    cov.stop()/d" mindspeed_llm/tasks/checkpoint/convert_param.py
    sed -i "/    cov.save()/d" mindspeed_llm/tasks/checkpoint/convert_param.py


    sed -i "2d" evaluation.py
    sed -i "2d" evaluation.py
    sed -i "2d" evaluation.py
    sed -i "2d" evaluation.py

    sed -i "/    cov.start()/d" evaluation.py
    sed -i "/    cov.stop()/d" evaluation.py
    sed -i "/    cov.save()/d" evaluation.py

    sed -i "2d" inference.py
    sed -i "2d" inference.py
    sed -i "2d" inference.py
    sed -i "2d" inference.py

    sed -i "/    cov.start()/d" inference.py
    sed -i "/    cov.stop()/d" inference.py
    sed -i "/    cov.save()/d" inference.py

    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py
    sed -i "2d" posttrain_gpt.py

    sed -i "/    cov.stop()/d" posttrain_gpt.py
    sed -i "/    cov.save()/d" posttrain_gpt.py

    sed -i "2d" preprocess_data.py
    sed -i "2d" preprocess_data.py
    sed -i "2d" preprocess_data.py
    sed -i "2d" preprocess_data.py

    sed -i "/    cov.start()/d" preprocess_data.py
    sed -i "/    cov.stop()/d" preprocess_data.py
    sed -i "/    cov.save()/d" preprocess_data.py
}

add_coverage

# run the coverage for python files in the pipeline
find "$PIPELINE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.py" | while read -r file; do
            coverage run -p --source=$SOURCE_DIR $file
        done
    fi
done

# run the coverage for python files in the unit tests
find "$UT_DIR" -mindepth 0 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.py" | while read -r file; do
            echo "running ${file}"
            filename=$(basename "$file")
            extension="${filename##*.}"
            name="${filename%.$extension}"
            pytest -xs $file | tee "$GENERATE_LOG_DIR/$name.log" 2>&1
            PYTEST_EXITCODE=${PIPESTATUS[0]}
            if [ $PYTEST_EXITCODE -ne 0 ]; then
                echo "$file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
            fi
            coverage run -p --source=$SOURCE_DIR $file
        done
    fi
done

# run the coverage for shell scripts in the st
for test_case in "$ST_DIR"/*.sh; do
    file_name=$(basename "${test_case}")
    echo "Running $file_name..."
    bash $test_case
done

# run the coverage for shell scripts in the pipeline
find "$PIPELINE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.sh" | while read -r file; do
            bash $file
        done
    fi
done

remove_coverage

# generate the coverage report
coverage combine
coverage html
coverage xml

# 压缩目录
echo "Compressing directory '$TARGET_DIR'..."
tar -czf htmlcov.tgz ${PWD}/htmlcov

# 检查压缩是否成功
if [ $? -eq 0 ]; then
    # 删除原目录
    echo "Removing original directory ${PWD}/htmlcov ..."
    rm -rf ${PWD}/htmlcov
else
    echo "Compression failed."
fi
