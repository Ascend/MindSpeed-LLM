# Edge-Cloud Collaborative Distributed Secure Training

## Use Cases

### Feature Overview

Edge-cloud collaborative distributed training is a feature designed for the secure computing power leasing service offered by telecom operators.

At present, enterprise customers in sectors such as finance and healthcare have two mainstream options for LLM fine-tuning:

- Build computing power in-house. This option requires major investment, such as buying training servers and building a data center, and it is difficult to promote among small and medium-sized enterprises.
- Rent computing power from an operator. This option requires uploading samples to the operator's cloud servers, which makes it difficult to satisfy data privacy and compliance requirements.

Edge-cloud collaborative distributed training is a training scheme that satisfies both limited on-site computing power and data-residency requirements. Based on conventional pipeline parallelism (PP), this scheme introduces a new model partitioning approach. A small number of model blocks that directly process raw samples run on the enterprise side, that is, the edge. A large number of model blocks that only need to process intermediate results run on the operator side, that is, the cloud. Under this deployment scheme, the edge needs only a small amount of computing power to process the first and last layers of the model, and it does not need to upload raw samples to the cloud.

![image](../../figures/ldt_sft/layerwise_disaggregated_training_stage.png 'layerwise_disaggregated_training_stage.png')

This feature supports the following capabilities:

- Raw samples do not go to the cloud. Pipeline parallelism (PP) supports U-shaped model partitioning. The first and last layers of the model deploy on the edge, and the cloud does not need to read the samples.
- Cross-domain collaborative training performance optimization. It optimizes pipeline scheduling and hides communication with computation to achieve efficient training in edge-cloud cross-domain connection scenarios.

### Design Principles

The principle is straightforward. PP supports U-shaped model partitioning, which means that the first pipeline stage deploys the parameters of the first and last layers together. In practice, you can specify the edge devices as the first pipeline stage, and then the first and last layers of the model are deployed on the edge at the same time.

During training, the process for a single sample is as follows:

- Forward pass on the edge: The edge reads the raw sample, processes it through the first layer of the model, converts it into activation values, and sends those activation values to the cloud.
- Forward pass on the cloud: After the cloud receives the activation values from the edge, it processes the intermediate hidden layers and sends the result back to the edge.
- Forward pass on the edge: The edge processes the last layer of the model and computes the loss. The forward pass then completes.

The backward pass is similar.

Effect: Throughout the training process, the edge sends only activation values during the forward pass and gradients during the backward pass to the cloud. Therefore, the raw samples do not need to go to the cloud.

Note: With U-shaped partitioning, each sample must complete four steps on the edge. These are the first-layer forward pass (`ForwardStart`, `FS`), the last-layer forward pass (`ForwardEnd`, `FE`), the last-layer backward pass (`BackwardStart`, `BS`), and the first-layer backward pass (`BackwardEnd`, `BE`).

### Cross-Domain Collaborative Training Optimization

Feature description: For U-shaped model partitioning, this feature optimizes pipeline scheduling and uses communication hiding to achieve high compute efficiency in cross-domain training.

Pipeline scheduling scheme: With U-shaped model partitioning, compared with conventional PP, where the first pipeline stage handles `FS` and `BE`, the first pipeline stage must also handle `FE` and `BS`. The pipeline scheduling scheme is designed as follows:

- Step 1. Split the first pipeline stage into two logical pipeline stages, one for the first layer and one for the last layer, and complete the pipeline schedule by referring to the `1F1B` schedule of conventional PP.
- Step 2. Merge the two logical pipeline stages. If the task queues of the two stages conflict, optimize the execution order of the tasks.

Example: `PP = 3`, `mbn = 4`

![image](../../figures/ldt_sft/pipeline_chart.png 'pipeline_chart.png')

The upper figure shows the two logical pipeline stages generated in Step 1, and the lower figure shows the final pipeline scheme after Step 2 merges them. When Step 2 merges the two edge logical pipeline stages, a task conflict appears. During optimization, reorder the tasks in `FS-FE-BS-BE` execution order. The basis for this optimization is that this execution order can increase the tolerable edge-cloud communication latency. For example, the communication for the forward pass of sample 3 can be hidden by the forward computation time of sample 5. This increases the tolerable communication latency and therefore reduces the compute efficiency loss in scenarios with long edge-cloud communication distances.

In summary, the edge pipeline scheduling rules are as follows. The cloud uses conventional PP scheduling for the middle layers.

| Stage | Operation | Count | Result in the Example |
| --- | --- | --- | --- |
| warmup | FS | PP + 1 | 4 |
| steady state 1 | FEBS | floor((PP-1)*2/3 - 1/2 + 2) | 2 |
| steady state 2 | FS-FE-BS-BE | mbn - floor((PP-1)*2/3 - 1/2 + 2) | 2 |
| cooldown | BE | floor((PP-1)*2/3 - 1/2 + 2) | 2 |

Effect: This pipeline scheduling scheme ensures that the steady state does not introduce extra bubbles. When the edge-cloud communication latency is less than `tf`, which is the forward computation time of a single microbatch on the edge, the steady state has no extra bubbles. The warmup and cooldown phases have only a small number of extra bubbles.

## How to Use

For detailed instructions, see [Usage Guide](../../training/finetune/mcore/layerwise_disaggregated_training.md).

| Key Parameter | Description |
|---------------------------------------|-----------------------------------------------------------------|
| --layerwise-disaggregated-training | Enables edge-cloud collaborative distributed secure training. |
| --num-layer-list [str] | Configures non-uniform PP partitioning. Pass the hidden-layer count for each pipeline stage as `L0,...,LPP`, where `L0` and `LPP` represent the number of hidden layers at the beginning and end, respectively. |
| --num-virtual-stages-per-pipeline-rank [int] | Configures the number of virtual pipeline stages. Set this parameter to `2`. |

## Constraints

### Supported Models

- The feature supports the following LLMs in the Qwen2.5 and Qwen3 series:

| Model Type | Specific Models |
|---------------------------------------|-----------------------------------------------------------------|
| LLM | Qwen3-32B, Qwen2.5-32B, Qwen2.5-72B |

- MoE models are not supported yet.

## Notes

- low-rank adaptation (LoRA) is not supported yet.
- Conventional virtual pipeline parallelism (VPP) is not supported. You must set `--num-virtual-stages-per-pipeline-rank` to `2` to enable joint deployment of the first and last layers.
