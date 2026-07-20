# 昇腾高可用性

## 说明

本文档仅提供特性介绍，若需使用完整的高可用特性请参考 MindCluster 官方指导文档：[[MindCluster指导文档](https://www.hiascend.com/software/mindcluster)]

## 使用场景

分布式优化器的思想是通过将优化器状态均匀地分布在数据并行组中来节省内存。基于该思想，MindIO 设计了将数据并行组切分成两个副本数据并行组的方案，副本优化器将优化器状态均匀分布在副本数据并行组，实现优化器状态均有备份。由于本特性对片上内存占用会有一定增加，推荐千卡及以上的大规模集群使用本特性，减少故障引起的机时损失。结合华为自研的高可用框架，可实现以下功能：

### TTP临终遗言功能

在训练过程中发生故障后，校验优化器中间状态数据的完整性和一致性，生成一次临终 checkpoint 数据，恢复训练时能够通过该 checkpoint 恢复到故障前一刻的状态，减少故障造成的训练迭代损失。

### UCE Step级重计算功能

昇腾芯片支持 NPU 卡内存发生 UCE 故障（内存不可修复）的实时检测，检测到 UCE 故障后，基于优化器状态副本机制并完成故障卡的在线修复并继续训练，减少训练损失。

### 弹性训练功能

在训练过程中发生故障后，在训练集群中没有空闲资源可替换时，基于优化器状态副本机制缩掉部分节点继续训练；当训练集群中有空闲资源可使用时，再基于优化器状态副本机制扩容回原有规模继续训练。
当前阶段仅支持 Data Parallel 级别的弹性训练，即按照 Data Parallel 粒度缩掉部分数据并行域进行扩容或缩容。

### 原理说明

Megatron原生的分布式优化器数据流及工作原理如下图：

![](../../figures/high_availability/grad_buffer_sharding.png)

副本优化器通过设计优化器参数均匀分布在副本 DP 组，完成优化器状态的备份，从而为 TTP/UCE 功能提供机制支持：

![](../../figures/high_availability/replica_optimizer.png)

副本优化器相比分布式优化器会有内存占用增加，相对占用如下：

|                                  | Non-distributed optim | Distributed optim | Replica optim |
|----------------------------------|-----------------------|-------------------|---------------|
| fp16/bf16 param, fp16/bf16 grads | 20                    | 4 + 16/d          | 4 + 32/d      |
| fp16/bf16 param, fp32 grads      | 18                    | 6 + 12/d          | 6 + 24/d      |

## 使用说明

### 环境准备

MindIO的功能以whl包的形式提供。

mindio_ttp下载地址：[MindIO TTP 下载软件包-昇腾社区](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/fault_recovery_acceleration/02_installation_and_deployment.md#%E5%87%86%E5%A4%87%E8%BD%AF%E4%BB%B6%E5%8C%85)

### 启动脚本中添加启动参数

`--enable-high-availability`  # 使能开启高可用功能的总开关，并使能TTP临终遗言功能，保存 checkpoint 时要求全局至少存在一份完整的优化器数据；

`--enable-hbmfault-repair` # 使能进行片上内存故障，Step 级重计算功能的开关；本功能将在线进行 worker 级修复，修复时要求全局至少存在一个故障卡的副本卡。

`--enable-worker-reboot` # 使能空中加油功能，配合支持相关功能的 MindCluster 组件共同使能后，在发生一般性故障时，进行进程级重启修复，继续训练。本功能会将故障卡所在节点进行重启，修复时要求未故障节点中至少存在一份完整的优化器数据。

`--distributed-optimizer-no-replica`  # 不使用副本优化器而使用 CKPT 文件进行重计算和空中加油修复，需要在故障时存在 CKPT 文件。

`--enable-elastic-training` # 使能弹性训练功能，配合支持相关功能的 MindCluster 组件共同使能后，在发生一般性故障且无空闲芯片资源时，缩掉部分节点后继续训练，待有可用芯片资源时扩容回原有规模继续训练。本功能会将故障卡所在 Data Parallel 域对应节点剔除，修复时要求未故障节点中至少存在一份完整的优化器数据。

### 启动脚本中添加环境变量

为避免在结合 mindx 使用时需配置多个组件的开关，添加环境变量，环境变量优先级高于 args，设置环境变量会被优先使用。

`export HIGH_AVAILABILITY=dump` 启用 `--enable-high-availability`

`export HIGH_AVAILABILITY=retry` 启用 `--enable-high-availability` `--enable-hbmfault-repair`

`export HIGH_AVAILABILITY=recover` 启用 `--enable-high-availability` `--enable-worker-reboot`

`export HIGH_AVAILABILITY=elastic-training` 启用 `--enable-high-availability` `--enable-elastic-training`

## 使用约束

1、由于原理限制，为了保证故障发生后，有完整的优化器状态数据，需要在  PTD（P+T+D 三维并行）切分时保障 Data Parallel Size 大于1，在使用MoE特性时还要求稠密层与稀疏层的 Data Parallel Size 均大于1，在使用长序列并行特性时还要求 dp_cp_size 大于1。

2、MindSpeed-llm 的高可用特性仅针对基础模型和基础训练特性进行适配以帮助用户快速体验该特性，暂未进行全量训练特性兼容，如用户对某个训练特性存在兼容需求，请在社区提出 issue 后我们将快速完成适配。

3、 高可用必须兼容的特性清单（前提需要模型本身支持）

   - `--bf16` （覆盖混精计算，范围广）
   - `--overlap-grad-reduce` （梯度、优化器都涉及）
   - `--load` （权重相关参数）
   - `--save` （权重相关参数）
   - `--ckpt-format torch` （权重相关参数）
   - `--use-distributed-optimizer`(优化器相关，开和未开都支持)

### 弹性训练功能使用约束

除上述使用约束外，针对弹性训练功能还需遵守以下使用约束：

1、当前仅支持开启 `enable-high-availability`、`use-distributed-optimizer`

2、当前仅支持不开启 `use-custom-fsdp`、 `reuse-fp32-param` 的场景

3、当前仅支持 `Data Parallel`、`Tensor Parallel`、`Pipeline Parallel` 并行

4、当前缩容后不可再次缩容，扩容仅支持直接扩容回原有规模

详见：[MindIO TTP 约束限制-昇腾社区](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/fault_recovery_acceleration/02_installation_and_deployment.md#%E7%BA%A6%E6%9D%9F%E9%99%90%E5%88%B6)

### Checkpoint 保存与加载优化

开启 `enable-high-availability` 时，若环境上安装了 MindIO ACP SDK ，则会使用mindio_acp的一级异步 checkpoint 保存与加载优化

详见：[MindIO TTP 约束限制-昇腾社区](https://gitcode.com/Ascend/mind-cluster/blob/branch_v26.0.0/docs/zh/scheduling/fault_recovery_acceleration/02_installation_and_deployment.md#%E7%BA%A6%E6%9D%9F%E9%99%90%E5%88%B6)
