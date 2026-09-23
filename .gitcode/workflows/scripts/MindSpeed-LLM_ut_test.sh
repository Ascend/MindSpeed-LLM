#!/bin/bash
# 用于 ModelLink 门禁UT 脚本
# shellcheck source=/dev/null
set -e
WORKSPACE=$1
pr_id=$2
branch=$3
# 进入工作目录
cd "${WORKSPACE}/CODE"
echo "Mindspeed-LLM Workspace is ${WORKSPACE}"
git diff-tree -r --name-only --no-commit-id origin/${branch} HEAD > ${WORKSPACE}/modify.txt
cat ${WORKSPACE}/modify.txt
cd "${WORKSPACE}"

git clone https://gitcode.com/GitHub_Trending/me/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron "${WORKSPACE}"/CODE/
cd ..

if [ ${branch} == "master" ]; then
    git clone https://gitcode.com/Ascend/FSDPTurbo.git
    cd FSDPTurbo
    git checkout main
    cp -r fsdp_turbo "${WORKSPACE}"/CODE/
    cd ..

    git clone https://gitcode.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout master
    cp -r mindspeed "${WORKSPACE}"/CODE/

elif [ ${branch} == "26.1.0" ]; then
    git clone https://gitcode.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 26.1.0_core_r0.12.1
    cp -r mindspeed "${WORKSPACE}"/CODE/

elif [ ${branch} == "26.0.0" ]; then
    git clone https://gitcode.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 26.0.0_core_r0.12.1
    cp -r mindspeed "${WORKSPACE}"/CODE/
fi

echo "install MindSpeed requirements"
pip install -r requirements.txt
cd -

# source cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source ATB
source /usr/local/Ascend/nnal/atb/set_env.sh
# cd "${WORKSPACE}"/CODE/ci

cd "${WORKSPACE}"/CODE
echo "install MindSpeed-LLM requirements"
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:${WORKSPACE}/CODE
echo $PYTHONPATH
ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs -r kill -9
python ci/access_control_test.py
