# ============================================
# MindSpeed LLM Docker Image
# NPU Type: Configurable (910b, a3, etc.)
# Supports: x86_64 and aarch64 (ARM)
# Supports: openEuler and Ubuntu
# ============================================

# Global arguments for base image selection
ARG OS=openeuler24.03
ARG BASE_IMAGE_VERSION=8.5.2
ARG NPU_TYPE=910b
ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=""

# ------------------------------
# Stage 1: Base Image
# ------------------------------
FROM ${BASE_IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${BASE_IMAGE_VERSION}-${NPU_TYPE}-${OS}-py${PYTHON_VERSION}} AS base

# Switch to root user
USER root

ARG OS_FAMILY=openeuler

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Detect architecture (x86_64 or aarch64)
RUN ARCH=$(uname -m) && \
    echo "Detected CPU architecture: ${ARCH}" && \
    if [ "$ARCH" = "x86_64" ]; then \
        echo "Architecture type: x86"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        echo "Architecture type: ARM"; \
    else \
        echo "ERROR: Unsupported architecture: $ARCH"; \
        exit 1; \
    fi

# Configure DNS for stable network connection
RUN if [ -f /etc/resolv.conf ]; then \
        echo "Existing resolv.conf:" && cat /etc/resolv.conf; \
    else \
        echo "No resolv.conf found, creating one..."; \
    fi && \
    echo "nameserver 114.114.114.114" > /etc/resolv.conf && \
    echo "nameserver 8.8.8.8" >> /etc/resolv.conf && \
    echo "nameserver 223.5.5.5" >> /etc/resolv.conf && \
    echo "Updated resolv.conf:" && cat /etc/resolv.conf

# Configure repository based on OS_FAMILY (openeuler/ubuntu)
COPY configure_repo.sh /tmp/configure_repo.sh
RUN chmod +x /tmp/configure_repo.sh && \
    bash /tmp/configure_repo.sh && \
    rm /tmp/configure_repo.sh

# Clean YUM cache and temporary files
RUN yum clean all && \
    rm -rf /var/cache/yum && \
    rm -rf /tmp/* /var/tmp/* /var/log/*

# Install system dependencies
RUN echo "Installing system dependencies..." && \
    if [ "$OS_FAMILY" = "openeuler" ]; then \
        yum install -y \
            git \
            iproute \
            wget \
            curl \
            gcc \
            gcc-c++ \
            make \
            cmake \
        && yum clean all; \
    elif [ "$OS_FAMILY" = "ubuntu" ]; then \
        apt-get install -y \
            git \
            iproute2 \
            wget \
            curl \
            gcc \
            g++ \
            make \
            cmake \
        && apt-get clean; \
    else \
        echo "ERROR: Unsupported OS: $OS_FAMILY"; \
        exit 1; \
    fi

# ------------------------------
# Stage 2: Builder (Python env + PyTorch)
# ------------------------------
FROM base AS builder

ARG TORCH_VERSION=2.7.1
ARG TORCH_NPU_VERSION=2.7.1

WORKDIR /tmp

# Install Miniconda (auto-detect architecture)
RUN echo "=== Detecting system architecture ===" && \
    arch=$(uname -m) && \
    echo "Current architecture: $arch" && \
    if [ "$arch" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_26.1.1-1-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-py311_26.1.1-1-Linux-aarch64.sh"; \
    fi && \
    echo "Download URL: $MINICONDA_URL" && \
    wget --no-check-certificate "$MINICONDA_URL" -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm -f miniconda.sh && \
    /opt/conda/bin/conda clean -ya && \
    ln -sf /opt/conda/bin/python /usr/bin/python && \
    ln -sf /opt/conda/bin/pip /usr/bin/pip

# Set environment variables for Conda
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Configure pip source
RUN pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple && \
    pip config set global.trusted-host "repo.huaweicloud.com" && \
    /opt/conda/bin/conda init bash

# Install PyTorch and torch_npu (ONLINE INSTALL ONLY)
RUN echo "Installing PyTorch and torch_npu from PyPI..." && \
        ARCH=$(uname -m) && \
        MAX_RETRIES=3 && \
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            if [ "$ARCH" = "x86_64" ]; then \
                echo "Installing PyTorch for x86_64..." && \
                pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && break; \
            elif [ "$ARCH" = "aarch64" ]; then \
                echo "Installing PyTorch for aarch64..." && \
                pip install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio && break; \
            fi; \
        done && \
        for retry in $(seq 1 $MAX_RETRIES); do \
            echo ">>> Installing torch-npu, attempt $retry of $MAX_RETRIES"; \
            pip cache purge 2>/dev/null || true; \
            rm -rf /root/.cache/pip; \
            pip install --no-cache-dir torch-npu==${TORCH_NPU_VERSION} && break; \
        done;

RUN conda clean -ya

# ------------------------------
# Stage 3: Final Runtime Image
# ------------------------------
FROM builder AS final

# Arguments for MindSpeed-LLM
ARG MINDSPEED_LLM_BRANCH=26.0.0
ARG MINDSPEED_BRANCH=26.0.0_core_r0.12.1
ARG MEGATRON_BRANCH=core_v0.12.1

WORKDIR /workspace

# Disable Git SSL verification (before cloning)
RUN git config --global http.sslVerify false

# Install MindSpeed
RUN git clone https://gitcode.com/ascend/MindSpeed.git && \
    cd MindSpeed && \
    git checkout ${MINDSPEED_BRANCH} && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

# Install MindSpeed-LLM and Megatron-LM
RUN git clone https://gitcode.com/ascend/MindSpeed-LLM.git && \
    git clone https://github.com/NVIDIA/Megatron-LM.git && \
    cd Megatron-LM && \
    git checkout ${MEGATRON_BRANCH} && \
    cp -r megatron ../MindSpeed-LLM/ && \
    cd ../MindSpeed-LLM && \
    git checkout ${MINDSPEED_LLM_BRANCH} && \
    mkdir logs && \
    pip3 install -r requirements.txt

# Set working directory
WORKDIR /workspace/MindSpeed-LLM

# Auto-run on login
RUN echo 'cd /workspace/MindSpeed-LLM' >> /root/.bashrc && \
    echo '. /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'conda activate base' >> /root/.bashrc

# Environment variables
ENV PATH=/opt/conda/bin:$PATH
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit