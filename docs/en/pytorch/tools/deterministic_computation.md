# Ascend Deterministic Computation

## Use Cases

During training, random factors can make each run differ slightly, which prevents the loss curve, performance curve, and other results from matching exactly. However, repeated runs and comparison experiments sometimes require deterministic results to ensure reproducibility. To meet this need, this feature introduces deterministic computation, which lets users ensure consistent results across multiple training runs on Ascend chips and helps with performance tuning, comparison experiments, and other tasks.

## How to Use

To enable this feature, add `--npu-deterministic` to the script and set the environment variable `export HCCL_DETERMINISTIC=true`. The default seed is 1234, and you can also set a custom value.

## Effects

With deterministic computation enabled, multiple experiments with the same parameters produce the same results.

> If you configure `HCCL_LOGIC_SUPERPOD_ID`, changes in the cluster topology during RoCE communication can lead to different collective communication behavior. The communication algorithms then follow different logic, and the two deterministic schemes are not aligned.
