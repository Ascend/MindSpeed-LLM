# 简介

## 概述

MindSpeed LLM，作为昇腾大模型训练框架，旨在为华为昇腾硬件提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调以及对应的开发工具链。

MindSpeed LLM支持Transformer架构的大型语言模型（LLM，Large Language Model），并支持MoE模型的训练和调优。提供超过100个主流公版模型，以及开箱即用的模型训练脚本。

MindSpeed LLM是基于训练加速库MindSpeed的大语言模型分布式训练框架，原生对接MindSpeed Core训练加速库，从并行优化、内存优化、通信优化、计算优化四个方面，对基于昇腾硬件的大模型训练进行了极致优化。

## MindSpeed LLM架构

MindSpeed LLM架构关系如[图1](#架构图)所示，整体分为两个层次：
- **MindSpeed Core基础加速**  
    提供融合算子、融合优化器及自研加速模块等底层能力，围绕并行、内存、通信、计算进行深度优化。MindSpeed LLM调用上述加速特性以保障训练性能。
- **MindSpeed LLM**  
    - **模型**：基于统一Transformer基座，通过编辑模块快速构建QWen3、DeepSeekV3、LLaMA3、Mamba2等公版模型。
    - **训练**：分布式预训练（适配Megatron-LM）+ 分布式指令微调（集成Peft）。
    - **功能**：覆盖数据处理、权重转换（Megatron/Huggingface互转）、断点续训、推理、评估等全流程，并提供易用性工具。

**图 1**  MindSpeed LLM架构图<a id="架构图"></a>  
<img src="./pytorch/figures/introduction/architecture_mindspeed_llm.png" width="60%"/>

## 功能特性

- 主流LLM大语言模型：支持Qwen3/DeepSeek/Mamba2系列等100+主流LLM模型，涵盖Dense/MoE/SSM等LLM架构，提供针对昇腾架构的高性能训练脚本，开箱即用。

- 分布式预训练：支持分布式预训练，提供数据预处理方案与包含TP/PP/DP/CP/EP在内的多维并行策略。

- 分布式指令微调：支持业界主流的全参微调/LoRA/QLoRA微调训练算法，并提供微调性能/显存优化手段。

- 模型权重转换：支持Megatron/HuggingFace格式的权重转换和LoRA微调权重的独立/合并转换。

- 在线推理与评估：支持模型分布式在线推理与公版数据的在线评估。
