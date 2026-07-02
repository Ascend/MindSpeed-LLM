# Checkpoint-based Resumable Training

During large-scale model pretraining, training may stop because of hardware failures, resource scheduling, or similar issues. To support resuming training after an interruption, the system provides the **checkpoint-based resumable training** feature. This document briefly explains how to configure and use this feature.

---

## Prerequisites for Enabling Checkpoint-based Resumable Training

To support checkpoint-based resumable training, set the related parameters correctly when you launch the pretraining script. This ensures that the optimizer state, model parameters, and training progress are all saved completely.

### Correctly Set the `pretrain` Script Parameters

In the pretraining script, pay attention to `GPT_ARGS`:

```bash
GPT_ARGS="
    [other parameters...] \
    --use-distributed-optimizer \  # Use the distributed optimizer. Required.
"
```

#### Key Parameters

| Parameter | Description |
|------|------|
| `--use-distributed-optimizer` | This parameter must be enabled. This saves the optimizer state across data parallel ranks in a distributed manner, which makes later restoration easier. |
| `--finetune` | ❌ Do not set this option. Otherwise, the system skips loading the optimizer state. |
| `--no-load-optim` | ❌ Do not set this option. Otherwise, the system does not restore the optimizer state, such as the learning rate and momentum. |

> ⚠️ If you set `--finetune`, `--no-load-optim`, or `--no-load-rng`, the system does not restore the optimizer state or the random state. Therefore, it cannot truly resume training.

---

## Saving Training Checkpoints

During training, if you configure `--save $SAVE_PATH` and a save interval such as `--save-interval`, the system automatically saves full checkpoints at regular intervals. These checkpoints include:

- Model weights
- Optimizer states
- Training iterations
- Random states

Example:

```bash
--save /your/checkpoint/path \
--save-interval 500   # Save once every 500 steps.
```

Each save produces a structure like this:

```shell
/your/checkpoint/path/
├── latest_checkpointed_iteration.txt
├── iter_0000001/
│   ├── mp_rank_00_000
│   │   ├── distrib_optim.pt
│   │   └── model_optim_rng.pt
│   └── ...
└── iter_0000500/
    ├── mp_rank_00_000
    │   ├── distrib_optim.pt
    │   └── model_optim_rng.pt
    └── ...
```

---

## Loading Weights to Resume Training

To resume training after an interruption, specify `--load` in the pretraining script launch command and point it to the previous save path:

```bash
GPT_ARGS="
    [other parameters...] \
    --use-distributed-optimizer \  # Use the distributed optimizer. Required.
"

...

msrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    [other parameters...] \
    --load $CHECKPOINT_PATH \
```

The system automatically reads `latest_checkpointed_iteration.txt`, finds the latest iteration, and restores the model and optimizer states.

### Verifying the Resuming Result

- Model parameters
- Optimizer states, including Adam momentum and variance
- Learning rate scheduler state
- Completed training iterations, which prevents repeated training

After restoration, training continues from the checkpoint and prints the following log output:

```shell
successfully loaded checkpoint from xx at iteration x
(min, max) time across ranks(ms):
load-checkpoint ....................:(9289.88, 9288.22)
```

---

## Notes

1. **File integrity**: Ensure that the checkpoint files under `$CHECKPOINT_PATH` are complete and intact.
2. **Training configuration consistency**: When you resume training, the batch size, model structure, number of training iterations, optimizer type, and similar settings should match the saved run.
3. **Distributed environment match**: Keep the number of NPUs and the parallel strategy, such as TP/DP, unchanged. Otherwise, loading fails.
