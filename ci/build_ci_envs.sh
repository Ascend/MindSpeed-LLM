#!/bin/bash
# ============================================
# MindSpeed LLM CI 环境构建脚本
# ============================================
# 使用场景：CI 流水线中自动构建 Docker 镜像用于 UT/ST 测试
# 用法：bash build_ci_envs.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="${SCRIPT_DIR}/../docker/image_build.sh"

# 校验构建脚本是否存在
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "[ERROR] Build script not found: ${BUILD_SCRIPT}"
    exit 1
fi

# ============================================================
# 默认 CI 参数（可根据实际 CI 环境调整）
# ============================================================
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-910b-openeuler24.03-py3.12"
TORCH_VERSION="2.10.0"
TORCH_NPU_VERSION="2.10.0.post4"
TRITON_ASCEND_VERSION="3.2.2"
MINDSPEED_BRANCH="master"
MEGATRON_BRANCH="core_v0.12.1"
MINDSPEED_LLM_VERSION="master"

echo "=========================================="
echo "CI Environment Build Configuration"
echo "=========================================="
echo "Base Image:              ${BASE_IMAGE}"
echo "Torch Version:           ${TORCH_VERSION}"
echo "Torch-NPU Version:       ${TORCH_NPU_VERSION}"
echo "Triton-Ascend Version:   ${TRITON_ASCEND_VERSION}"
echo "MindSpeed-LLM Ver:       ${MINDSPEED_LLM_VERSION}"
echo "MindSpeed Branch:        ${MINDSPEED_BRANCH}"
echo "Megatron Branch:         ${MEGATRON_BRANCH}"
echo "=========================================="

# 切换到 docker 目录执行构建
cd "$SCRIPT_DIR/../docker"

# 构建参数数组
BUILD_ARGS=(
    --base-image "$BASE_IMAGE"
    --torch-version "$TORCH_VERSION"
    --torch-npu-version "$TORCH_NPU_VERSION"
    --triton-ascend-version "$TRITON_ASCEND_VERSION"
    --mindspeed-llm-branch "$MINDSPEED_LLM_VERSION"
    --mindspeed-branch "$MINDSPEED_BRANCH"
    --megatron-branch "$MEGATRON_BRANCH"
    --cleanup-on-fail
)

echo ""
echo "[CI] Starting image build..."
echo ""

set +e
bash "$BUILD_SCRIPT" "${BUILD_ARGS[@]}"
BUILD_RESULT=$?
set -e

if [ $BUILD_RESULT -eq 0 ]; then
    echo ""
    echo "[CI] Build completed successfully."
else
    echo ""
    echo "[CI] Build failed with exit code: ${BUILD_RESULT}"
    exit $BUILD_RESULT
fi
