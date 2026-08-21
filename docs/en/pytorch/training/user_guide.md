# Model Usage Guide

MindSpeed LLM provides an end-to-end LLM training solution, covering distributed pretraining, distributed fine-tuning, and inference.

In [Quick Start (Megatron Training Backend)](./quick_start.md) or [Quick Start (FSDP2 Training Backend)](./fsdp2_quick_start.md), you can use the Qwen3-8B model to quickly master LLM pretraining and fine-tuning tasks. The following table describes in detail how to use MindSpeed LLM models.

**Table 1** Model training solutions and usage instructions

| Category | Content | Usage Scenario | Description |
| :--- | :--- | :--- | :--- |
| Pretraining | [Distributed Pretraining for LLMs](./pretrain/mcore/pretrain.md) | Provides data preprocessing and pretraining scripts for offline model pretraining | <ul><li>To load Hugging Face weights, perform weight conversion in advance.</li><li>To store logs in script files, create the <code>logs</code> folder in the running directory.</li></ul> |
| Pretraining | [Training with Online Data and Weight Loading](./pretrain/mcore/train_from_hf.md) | Integrates data preprocessing, weight conversion, and training into one process, providing a one-click solution from Hugging Face open-source data and weights to training | <ul><li>Currently supported Hugging Face model types include: <code>qwen3</code>, <code>qwen3-moe</code>, <code>deepseek3</code>, <code>glm45-air</code>, <code>bailing_mini</code>, <code>qwen3-next</code>, <code>seed-oss</code>, <code>deepseek32</code>, <code>magistral</code>, <code>deepseek2-lite</code>.</li><li>Currently, the automatic dataset conversion feature supports only the following raw data formats: <code>parquet, arrow, csv, json, jsonl, txt</code>. Other formats are not supported for now.</li><li>Currently, the <code>--enable-mg2hf-convert</code> weight conversion feature supports only single-node or shared storage environments and does not support Megatron-to-HF weight conversion for LoRA fine-tuned weights.</li></ul> |
| Fine-tuning | [Single-Sample Fine-Tuning](./finetune/mcore/single_sample_finetune.md) | Suitable for general instruction fine-tuning of single-turn tasks without historical dependencies | The parallel configuration of training parameters in the fine-tuning script (for example, TP/PP/EP/VPP) must be consistent with that used during weight conversion. |
| Inference | [MindSpeed LLM Streaming Inference](./inference/inference.md) | Supports streaming output of multiple generation strategies such as `greedy_search` and `beam_search` | Streaming inference currently uses a fixed `Instruction` input to compare model inference results. |
