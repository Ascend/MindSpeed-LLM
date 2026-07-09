# Muon Optimizer

## Use Cases

Muon, short for Momentum + Orthogonalization Update, is an efficient optimizer for pretraining LLMs. Its core idea is to orthogonalize momentum gradients through Newton-Schulz iteration. Therefore, the parameter update matrix is approximately orthogonal. It suits training tasks that seek better convergence efficiency than Adam under the same compute budget.

## Usage Notes

Enable the Muon optimizer by setting `--optimizer muon` in the training script. The following table lists the full parameter reference.

### Basic Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| --optimizer muon | str | — | Enables the Muon optimizer. |
| --muon-momentum | float | 0.9 | Momentum coefficient for the internal SGD used by Muon. |
| --muon-use-nesterov | flag | False | Enables Nesterov momentum in the internal SGD. |
| --muon-no-split-qkv | flag | True | Disables independent orthogonalization of QKV parameters in blockwise mode. Blockwise mode is enabled by default. |
| --muon-extra-scale-factor | float | 1.0 | Applies an additional global scaling factor to the Muon update. |

### Newton-Schulz Iteration Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| --muon-num-ns-steps | int | 5 | Number of Newton-Schulz iteration steps. More steps produce more accurate orthogonalization, but increase compute cost. |
| --muon-fp32-matmul-prec | str | medium | FP32 matrix-multiplication precision in NS iterations. This affects the numerical stability of orthogonalization. |
| --muon-scale-mode | str | spectral | Scaling mode for the update after orthogonalization. |

### Tensor Parallel Mode Parameters

| Parameter | Type | Default Value | Description |
|------|------|--------|------|
| --muon-tp-mode | str | blockwise | How Newton-Schulz orthogonalization for tensor-parallel weights is calculated. |

## Usage Constraints

When you use the Muon optimizer, **do not enable the following features at the same time**.

| Incompatible Feature | Corresponding Parameter |
|------------|----------|
| Gradient reduction overlap | --overlap-grad-reduce |
| Parameter gather overlap | --overlap-param-gather |
| Distributed optimizer | --use-distributed-optimizer |
| Torch FSDP2 | --use-torch-fsdp2 |
