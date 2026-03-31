#!/bin/bash
#=============================================
# Author: Xuguoliang
# Date: 2026-03-17
# Description: ST for feature layerwise disaggregated training
# Remarks: 
#=============================================

set -u
set +e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

declare -a CHILD_PIDS=()
EXIT_CODE=0
CONVERSION_SUCESS=false

log_info()  { echo -e "${GREEN}[INFO]${NC} $(date '+%H:%M:%S') - $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') - $1"; }
log_error()  { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') - $1"; }
log_step()  { echo -e "${BLUE}[STEP]${NC} $(date '+%H:%M:%S') - $1"; }


cleanup() {
    log_info "Received termination signal, starting cleanup of background processes"
    for pid in "${CHILD_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log_info "Terminating background process $pid"
            kill -TERM "$pid" 2>/dev/null
        fi
    done
    exit 0
}

# Register termination signal handler (Ctrl+C or kill command)
trap cleanup INT TERM EXIT

# Configuration parameters
# Model conversion configuration
HF_MODEL_DIR="/data/ci/models/qwen25/hf/Qwen2.5-7B-Instruct/"
TOKENIZER_PATH="/data/ci/models/qwen25/hf/Qwen2.5-7B-Instruct/tokenizer.json"
MG_SAVE_DIR="/data/ci/cache/qwen2.5_7b_tp1pp5/"
VPP_EDGE_SAVE_DIR="/data/ci/cache/qwen2.5_7b_tp1pp4_vpp_edge/"
VPP_CLOUD_SAVE_DIR="/data/ci/cache/qwen2.5_7b_tp1pp4_vpp_cloud/"
VPP_2_MG_SAVE_DIR="/data/ci/cache/qwen2.5_7b_tp1pp5_vpp_2_mg/"

# Parallel training script configuration
TRAIN_SCRIPTS=(
    "./tests/pipeline/st/layerwise_disaggregated_training/tune_qwen25_7b_tp1pp4_full_ptd_edge"
    "./tests/pipeline/st/layerwise_disaggregated_training/tune_qwen25_7b_tp1pp4_full_ptd_cloud"
)

check_environment() {
    log_step "Starting environment check..."

    if [ ! -d "$HF_MODEL_DIR" ]; then
        log_error "HuggingFace model directory not found: $HF_MODEL_DIR"
        return 1
    fi

    if ! command -v python &> /dev/null; then
        log_error "Python interpreter not found, please ensure the correct virtual environment is activated"
        return 1
    fi

    for script in "${TRAIN_SCRIPTS[@]}"; do
        if [ ! -f "$script" ]; then
            log_error "Training script not found: $script"
            return 1
        fi
    done

    log_info "Environment check passed"
    return 0
}

run_model_conversion() {
    log_step "Starting model conversion process..."
    
    # Creating directories
    mkdir -p "$MG_SAVE_DIR"
    mkdir -p "$VPP_EDGE_SAVE_DIR"
    mkdir -p "$VPP_CLOUD_SAVE_DIR"

    # --- Step 1: HF -> Megatron ---
    log_step "Executing Step 1: HF -> Megatron"
    python convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 1 \
       --target-pipeline-parallel-size 5 \
       --num-layer-list "2,8,8,8,2" \
       --add-qkv-bias \
       --load-dir "$HF_MODEL_DIR" \
       --save-dir "$MG_SAVE_DIR" \
       --tokenizer-model "$TOKENIZER_PATH" \
       --model-type-hf llama2 \
       --params-dtype bf16
    
    if [ $? -ne 0 ]; then
        log_error "Step 1: HF -> Megatron model conversion failed"
        return 1
    fi

    # --- Step 2: Megatron -> VPP Edge ---
    log_step "Executing Step 2: Megatron -> VPP Edge"
    python mindspeed_llm/tasks/posttrain/ldt_sft/convert_ckpt_pp_vpp.py merge \
       --load-dir "$MG_SAVE_DIR" \
       --save-dir-edge "$VPP_EDGE_SAVE_DIR" \
       --save-dir-cloud "$VPP_CLOUD_SAVE_DIR" \
       --merge-stages 0,4 \
       --middle-stages 1,2,3
    
    if [ $? -ne 0 ]; then
        log_error "Step 2: Megatron -> VPP Edge model conversion failed"
        return 1
    fi

    log_info "Model conversion process completed"
    CONVERSION_SUCESS=true
    return 0
}

run_parallel_tasks() {
    if [ "$CONVERSION_SUCESS" != true ]; then
        log_error "Model conversion process not completed successfully, cannot execute parallel training tasks"
        return 1
    fi

    log_step "Starting parallel training tasks..."

    local count=${#TRAIN_SCRIPTS[@]}
    log_info "Found $count parallel training tasks"
    
    for i in "${!TRAIN_SCRIPTS[@]}"; do
        local script="${TRAIN_SCRIPTS[$i]}"
        local pid

        log_info "Starting script [$i]: $script"
        bash "$script" &
        pid=$!
        log_info "Script [$i] started successfully, PID: $pid, log output printed to terminal"

        CHILD_PIDS+=($pid)
    done

    log_info "All parallel tasks started, monitoring running status..."

    # Waiting for all child processes to complete
    local failed_count=0
    for pid in "${CHILD_PIDS[@]}"; do
        wait "$pid"
        local status=$?
        if [ $status -ne 0 ]; then
            log_error "Process PID [$pid] exited with status code: $status"
            ((failed_count++))
        fi
    done

    if [ $failed_count -gt 0 ]; then
        log_error "$failed_count scripts failed during parallel task execution"
        return 1
    else
        log_info "All parallel tasks completed"
        return 0
    fi
}

run_model_conversion_2() {
    log_step "Starting model conversion process..."
    
    # Creating directories
    mkdir -p "$VPP_2_MG_SAVE_DIR"

    # --- Step 1: VPP Edge -> Megatron ---
    log_step "Executing Step 1: VPP Edge -> Megatron"
    python mindspeed_llm/tasks/posttrain/ldt_sft/convert_ckpt_pp_vpp.py split \
       --load-dir-edge "$VPP_EDGE_SAVE_DIR" \
       --load-dir-cloud "$VPP_CLOUD_SAVE_DIR" \
       --save-dir "$VPP_2_MG_SAVE_DIR" \
       --split-rank 0 \
       --middle-ranks 1,2,3
    
    if [ $? -ne 0 ]; then
        log_error "Step 1: VPP Edge -> Megatron model conversion failed"
        return 1
    fi

    log_info "Model conversion process completed"
    CONVERSION_SUCESS=true
    return 0
}

main() {
    # change work dir
    basepath=$(cd `dirname $0`; cd ../../../; pwd)
    cd $basepath
    log_info "Changed to working directory: $basepath"

    if ! check_environment; then
        log_error "Environment check failed, task terminated"
        EXIT_CODE=1
        return
    fi

    if ! run_model_conversion; then
        log_error "Model conversion (HF->MG->VPP) process failed, task terminated"
        EXIT_CODE=1
        return
    fi

    if ! run_parallel_tasks; then
        log_error "Parallel training tasks failed, task terminated"        
        EXIT_CODE=1
        return
    fi

    if ! run_model_conversion_2; then
        log_error "Model conversion (VPP->MG) process failed, task terminated"
        EXIT_CODE=1
        return
    fi

    # clean cache dir
    log_step "Starting cleanup of cache directories..."
    rm -rf "$MG_SAVE_DIR"
    rm -rf "$VPP_EDGE_SAVE_DIR"
    rm -rf "$VPP_CLOUD_SAVE_DIR"
    rm -rf "$VPP_2_MG_SAVE_DIR"
    rm -rf /data/ci/cache/save_dir
    log_info "Cache directory cleanup completed"

    EXIT_CODE=0
}

# Script execution entry point
main
log_info "Script execution completed, exit code: $EXIT_CODE"
exit $EXIT_CODE
