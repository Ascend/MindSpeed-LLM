# Precision Alignment

## Overview

Precision alignment is a core verification step in model migration. It ensures that after a model is migrated from a GPU platform or another framework to MindSpeed LLM, the functional correctness, numerical precision, and training convergence remain consistent with the original model. LLM training involves many stages and technical layers, including data, models, frameworks, operators, and hardware. Therefore, precision issues may arise during training, and they generally fall into two categories: model precision issues and numerical precision issues.

Model precision issues mainly refer to problems in the data read from the dataset, the training hyperparameters, the model structure, or even the framework design or usage process itself. Model precision issues have a significant impact on convergence. Therefore, you need to eliminate and analyze each item carefully and adjust it based on the actual situation.

Numerical precision issues mainly refer to approximation errors caused by the finite word length effect of floating-point computation, the computation order, the communication order, or the mathematical expressions used in various computations. The approximate nature of computed values may affect model convergence with a certain probability, but you cannot simply assume that differences in the computation process will definitely cause convergence problems. The numerical precision of operators is the foundation of the computation process. Operator precision issues are generally considered one of the sources of LLM precision issues and require attention. However, due to differences in implementation processes, the same computation process on different hardware (for example, between GPU and CPU, or between different GPU versions) usually produces somewhat different numerical results. Within a specific tolerance range, these differences do not affect final model convergence.

## Problem Scenarios

In the model migration scenario, if the training process and results differ from those of the benchmark (another framework on GPU or NPU) and the deviation exceeds the tolerance threshold, the precision is considered misaligned. The specific scenarios can be further divided into the following phenomena:

- **First-step loss difference**: The loss at step 0 or in the first few steps differs from the benchmark, and the average error exceeds the tolerance threshold.
- **Long-term stable loss difference**: The loss matches the benchmark in the early stage, but the difference from the benchmark gradually grows in the later stage, and the average error exceeds the tolerance threshold.
- **Spikes**: The loss or grad norm rises sharply and then falls quickly, occurring more frequently than in the benchmark.

## Guidance

Before locating a precision issue, you need to first rule out interference from inconsistent factors and ensure that the issue is reproducible. Therefore, perform a comparative check on the following checklist:

|Check Item|Description|
|---|---|
|Version alignment|Ensure that third-party library versions are consistent. Use `pip list` to check whether the versions of `torch`, `torch_npu`, `transformers`, and so on are aligned, and use the `git` branch to check whether the repository versions are consistent.|
|Configuration alignment|Compare whether the hyperparameters and environment variable settings in the training logs or startup scripts are consistent, such as the learning rate `lr`, the global batch size `GBS`, and the optimizer type.|
|Model structure alignment|Print and compare the model structures of both sides during training.|
|Weight initialization alignment|Ensure that the initialization weights before training are consistent. Confirm that the same pretrained model is loaded or that the same initialization random seed is used.|
|Data loading alignment|Check whether the data read from the dataset and fed into model training is consistent.|

After the preliminary check is complete, you can use the [msProbe accuracy tool](https://gitcode.com/Ascend/msprobe) to analyze and locate different problem scenarios. The tool provides usage guidance and a [large model training accuracy locating guide](https://www.hiascend.com/document/detail/en/mindstudio/2610/practicalcases/LargeModelTrainingAccuracy/docs/en/best_practices/train_debug_guide.md).
