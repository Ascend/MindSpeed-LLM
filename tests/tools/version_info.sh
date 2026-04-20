#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
# Script to collect and display environment version information including CANN, torch_npu, HDK, and related framework versions (transformers, triton-ascend, peft, etc.)

if ! which atc >/dev/null 2>&1; then
    echo "Error: atc command not found. Please source CANN environment first."
    exit 1
fi

if ! pip show torch_npu >/dev/null 2>&1; then
    echo "Error: torch_npu not found. Check whether you are inside a Docker container or a Conda environment."
    exit 1
fi

platform=`lscpu|grep -E "x86_64|aarch64"|awk '{print $2}'`
cann_path=$(dirname $(dirname $(dirname "$(which atc)")))
echo "Cann path:$cann_path"


device_model=$(npu-smi info | head -n 7 | tail -n 1 | awk '{ print "Ascend " $3 }')
hdk_version=$(npu-smi info | head -n 2 | tail -n 1 | awk '{ print $3 }')
enable_bin_op=$(python -c "import torch;import torch_npu;torch.npu.is_available();print(torch.npu.is_jit_compile_false())" | tail -n 1)
framework_version=$(python -c "import torch_npu;print('.'.join(torch_npu.__version__.split('.')[:-1]))")
cann_version=$(cat $cann_path/ascend-toolkit/latest/$platform-linux/ascend_toolkit_install.info | head -n 2|tail -n 1| awk '{ print $1 }')

echo "############## Code information ##############"
echo "----------------MindSpeed-LLM ----------------"
git log -1
echo "------------------MindSpeed ------------------"
(cd ../MindSpeed && git log -1)
echo ""

echo "##### Brief version information #####"
echo "device_model==$device_model"
echo "hdk_version==$hdk_version"
echo "enable_bin_op==$enable_bin_op"
echo "framework_version==$framework_version"
echo "cann_version==$cann_version"
echo ""

export PIP_DISABLE_PIP_VERSION_CHECK=1
driver_full_version=$(cat /usr/local/Ascend/driver/version.info)
cann_full_version=$(cat $cann_path/ascend-toolkit/latest/$platform-linux/ascend_toolkit_install.info | head -n 3| awk '{ print $1 }')
pta_full_version=$(pip list | grep torch_npu | awk '{ print $2 }')
transformers_version=$(pip list | grep transformers | awk '{ print $2 }') | head -1
triton_ascend_version=$(pip list | grep triton-ascend | awk '{ print $2 }')
peft_version=$(pip list | grep peft | awk '{ print $2 }')


echo "##### Detailed version information #####"
echo "------------------ HDK -------------------"
echo "$driver_full_version"
echo "------------------ CANN ------------------"
echo "$cann_full_version"
echo "------------------ PTA -------------------"
echo "$pta_full_version"
echo "-------------- transformers --------------"
echo "$transformers_version"
echo "------------- triton-ascend --------------"
echo "$triton_ascend_version"
echo "----------------- peft -------------------"
echo "$peft_version"
echo ""