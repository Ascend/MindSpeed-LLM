#!/bin/bash
# ============================================
# MindSpeed LLM Docker Image Build Script
# ============================================

cleanup_dangling() {
    echo ">>> Cleaning up <none> tagged images and corresponding containers..."

    local dangling_images=$(docker images -f "dangling=true" -q 2>/dev/null)
    if [ -n "$dangling_images" ]; then
        for img_id in $dangling_images; do
            local containers=$(docker ps -a -q --filter "ancestor=$img_id" 2>/dev/null)
            if [ -n "$containers" ]; then
                echo ">>> Removing containers from dangling image: $img_id"
                docker rm -f $containers 2>/dev/null || true
            fi
        done
        echo ">>> Removing dangling images..."
        docker rmi $dangling_images 2>/dev/null || true
    else
        echo ">>> No dangling images found"
    fi

    echo ">>> Cleanup complete"
}

OS="openEuler24.03"
NPU_TYPE="910B"
TORCH_VERSION="2.9.0"
TORCH_NPU_VERSION="2.9.0"
BASE_IMAGE_VERSION="9.0.0-beta.2"
MINDSPEED_LLM_VERSION="26.0.0"
CLEANUP_ON_FAIL=false

IMAGE_NAME="mindspeed_llm"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build MindSpeed LLM Docker Image

Required:
    -t, --npu-type TYPE      NPU type: A3 or 910B (auto-detected from --base-image if not specified)

Optional:
    -i, --image-name NAME    Image name (default: mindspeed-llm:{version}-{chip}-{os}-py{py_ver}-{arch})
    -o, --os OS              OS (default: openEuler24.03)
    -v, --version VERSION    MindSpeed LLM version (default: 26.0.0, determines model install scripts)
    --torch-version VER      PyTorch version (default: 2.9.0, for online install)
    --torch-npu-version VER  torch-npu version (default: 2.9.0, for online install)
    --base-image-version VER Base image CANN version (default: 9.0.0-beta.2)
    -h, --help               Show help

Dockerfile naming convention: Dockerfile (unified, supports all NPU types and OS)
    NPU type and OS are passed as build arguments

Image tag naming convention: {version}-{chip}-{os}-py{python_version}-{architecture}
    e.g. 26.0.0-a3-openeuler24.03-py3.11-aarch64
         26.0.0-910b-openeuler24.03-py3.11-aarch64

Examples:
    bash $0 -t A3
    bash $0 -t A3 -o openEuler24.03
    bash $0 -t 910B --torch-version 2.9.0 --torch-npu-version 2.9.0
    bash $0 -t A3 --base-image-version 9.0.0
    bash $0 -t A3 -i myproject/mindspeed-llm
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--npu-type)      NPU_TYPE="$2"; shift 2 ;;
        -i|--image-name)    IMAGE_NAME="$2"; shift 2 ;;
        -o|--os)            OS="$2"; shift 2 ;;
        -v|--version)       MINDSPEED_LLM_VERSION="$2"; shift 2 ;;
        --torch-version)    TORCH_VERSION="$2"; shift 2 ;;
        --torch-npu-version) TORCH_NPU_VERSION="$2"; shift 2 ;;
        --base-image-version) BASE_IMAGE_VERSION="$2"; shift 2 ;;
        --cleanup-on-fail)  CLEANUP_ON_FAIL=true; shift ;;
        -h|--help)          show_help; exit 0 ;;
        *)                  echo "Unknown argument: $1"; show_help; exit 1 ;;
    esac
done

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: Dockerfile not found: $DOCKERFILE"
    exit 1
fi

NPU_TYPE=$(echo "$NPU_TYPE" | tr '[:upper:]' '[:lower:]')
OS=$(echo "$OS" | tr '[:upper:]' '[:lower:]')

OS_FAMILY="openeuler"

REPO_SCRIPT="configure_yum_repo.sh"

cp "${SCRIPT_DIR}/${REPO_SCRIPT}" configure_repo.sh

echo "=========================================="
echo "Build Configuration"
echo "=========================================="
echo "NPU Type:           ${NPU_TYPE}"
echo "OS:                 ${OS}"
echo "OS_FAMILY:          ${OS_FAMILY}"
echo "Dockerfile:         ${DOCKERFILE}"
echo "Image Name:         ${IMAGE_NAME}"
echo "Base Image Version: ${BASE_IMAGE_VERSION}"
echo "PyTorch Version:    ${TORCH_VERSION}"
echo "torch-npu Version:  ${TORCH_NPU_VERSION}"
echo "MindSpeed LLM Ver:   ${MINDSPEED_LLM_VERSION}"

BUILD_ARGS="--build-arg OS=${OS}"
BUILD_ARGS="$BUILD_ARGS --build-arg OS_FAMILY=${OS_FAMILY}"
BUILD_ARGS="$BUILD_ARGS --build-arg NPU_TYPE=${NPU_TYPE}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_VERSION=${TORCH_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg TORCH_NPU_VERSION=${TORCH_NPU_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg BASE_IMAGE_VERSION=${BASE_IMAGE_VERSION}"
BUILD_ARGS="$BUILD_ARGS --build-arg MINDSPEED_LLM_VERSION=${MINDSPEED_LLM_VERSION}"


echo ""
echo "Starting image build..."
echo ""

# Temporarily disable set -e to handle build failure gracefully
set +e

docker build \
    -t "$IMAGE_NAME" \
    -f "$DOCKERFILE" \
    $BUILD_ARGS \
    --network=host \
    .

BUILD_RESULT=$?

# Restore set -e
set -e

rm -f "configure_repo.sh"

# Check build result and handle accordingly
if [ $BUILD_RESULT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Build Complete!"
    echo "Image: ${IMAGE_NAME}"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  docker run -it --rm ${IMAGE_NAME} bash"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "Build Failed!"
    echo "=========================================="
    if [ "$CLEANUP_ON_FAIL" = true ]; then
        echo ""
        echo ">>> Cleaning up dangling images and containers..."
        cleanup_dangling
    fi
    exit $BUILD_RESULT
fi
