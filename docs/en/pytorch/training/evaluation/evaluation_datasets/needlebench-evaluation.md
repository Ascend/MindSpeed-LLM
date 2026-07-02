# NeedleBench Accuracy Test

## Overview

NeedleBench is a framework developed by a research team at Shanghai Artificial Intelligence Laboratory and Tsinghua University to evaluate the retrieval and reasoning capabilities of LLMs when processing ultra-long text with context windows of up to one million tokens. It is designed specifically to stress-test the long-text processing capabilities of a model in bilingual Chinese-English contexts.

Download the NeedleBench test set: <https://huggingface.co/datasets/opencompass/NeedleBench>

Download the cache file required by `tiktoken`: <https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken>

Currently, the MindSpeed LLM repository supports only the Single-Needle-Retrieval (S-RT) test.

## Parameter Configuration

- Set `--max-new-tokens` to `64`.
- Set `--seq-length`, `--max-position-embeddings`, and `--max-tokens-to-oom` to `4096`, `8192`, `32768`, `131072`, `262144`, `524288`, or `1048576` to evaluate NeedleBench accuracy at the corresponding sequence length.
- Enable YaRN. Configure YaRN according to the [YaRN](../../../features/mcore/yarn.md) documentation.

The NeedleBench accuracy test results are as follows:

| Model | Task | MindSpeed LLM | Community Version ([OpenCompass](https://opencompass.readthedocs.io/en/latest/advanced_guides/needleinahaystack_eval.html)) |
|---|---|---|---|
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) | NeedleBench-128K-Single-Needle-Retrieval | 70.19% | 70.25% |
| [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) + YaRN | NeedleBench-128K-Single-Needle-Retrieval | 87.03% | 88.63% |
