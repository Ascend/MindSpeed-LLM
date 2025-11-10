# Setting
# source ascend-toolkit 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# source atb库 环境变量
source /usr/local/Ascend/nnal/atb/set_env.sh

# 安装加速库
git clone -b master https://gitcode.com/ascend/MindSpeed.git
cd MindSpeed

# checkout commit from MindSpeed master
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

# Main
# step 1: define dir
BASE_DIR=$(dirname "$(readlink -f "$0")")
CURRENT_TIME=$(date "+%Y-%m-%d")
BASELINE_DIR="$BASE_DIR/baseline"
GENERATE_LOG_BASE_DIR="/$(echo "$BASE_DIR" | cut -d'/' -f2)/pipeline_log_v2"
GENERATE_LOG_DIR="$GENERATE_LOG_BASE_DIR/$CURRENT_TIME"

#mkdir cache to store product and will be removed after test
mkdir -p "$GENERATE_LOG_DIR"
touch "$GENERATE_LOG_DIR/exec_error.log"
chmod a+w "$GENERATE_LOG_DIR/exec_error.log"
echo "core0.12.0 Execution Results" > $GENERATE_LOG_DIR/exec_error.log

# step 2: running scripts and execute `test_ci_pipeline.py` && running pytest
find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
    if [ -d "$dir" ]; then
        find "$dir" -type f -name "*.sh" | while read -r file; do
            filename=$(basename "$file")
            extension="${filename##*.}"
            name="${filename%.$extension}"
            echo "running $file"
            bash $file 2>&1 | tee "$GENERATE_LOG_DIR/$name.log"
            SCRIPT_EXITCODE=${PIPESTATUS[0]}
            if [ $SCRIPT_EXITCODE -eq 0 ]; then
                # begin to execute the logic of compare
                echo "$(dirname "$BASE_DIR")/test_tools/test_ci_st.py"
                pytest -x $(dirname "$BASE_DIR")/test_tools/test_ci_st.py \
                    --baseline-json $BASELINE_DIR/$name.json \
                    --generate-log $GENERATE_LOG_DIR/$name.log \
                    --generate-json $GENERATE_LOG_DIR/$name.json
                PYTEST_EXITCODE=$?
                if [ $PYTEST_EXITCODE -ne 0 ]; then
                    echo "${name}.sh compare to baseline has failed, check it!" >> $GENERATE_LOG_DIR/exec_error.log
                fi
            else
                echo "${name}.sh Script has failed. Exit!" >> $GENERATE_LOG_DIR/exec_error.log
            fi
        done

        # python test testing
        find "$dir" -type f -name "*.py" | while read -r file; do
            echo "running $file"
            tmp_file_name="${file#*MindSpeed-LLM/}"
            file_name="${tmp_file_name//\//_}"
            pytest -x --log-level=INFO "$file" 2>&1 | tee "${GENERATE_LOG_DIR}/${file_name}.log"
            PYTEST_EXITCODE=${PIPESTATUS[0]}
            if [ $PYTEST_EXITCODE -ne 0 ]; then
                echo "$file has failed, check it!" >> "$GENERATE_LOG_DIR/exec_error.log"
            fi
        done
    fi
done

rm -rf /data/ci/cache/*
echo "=================tar error log=================="
tar -czvf "${GENERATE_LOG_BASE_DIR}/${CURRENT_TIME}.tar.gz" "${GENERATE_LOG_DIR}/"