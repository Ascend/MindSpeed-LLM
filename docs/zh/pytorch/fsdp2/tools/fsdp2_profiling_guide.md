# MindSpeed LLM FSDP2后端 Profiling 工具使用指南

## 使用场景  

本工具基于 `torch_npu.profiler` 实现，集成于 MindSpeed FSDP2 训练流程。通过配置 YAML 或命令行参数，即可在指定训练步数和指定 rank 上自动采集性能数据，并生成 profiling 文件。

## 使用方法  

### 1. 配置方式

在训练配置文件（YAML）的 `training` 字段下添加 profiling 参数：

```yaml
training:
  # ... 其他训练参数 ...

  # --- Profiling 配置 ---
  profile: true
  profile_step_start: 5
  profile_step_end: 10
  profile_ranks: [0]
  profile_level: level1
  profile_export_type: text
  profile_data_simplification: false
  profile_with_cpu: true
  profile_with_stack: true
  profile_with_memory: true
  profile_record_shapes: true
  profile_save_path: ./npu_profile
```

### 2. 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `profile` | bool | `false` | 是否启用 profiling |
| `profile_step_start` | int | `0` | 开始采集的 global step（包含） |
| `profile_step_end` | int | `-1` | 结束采集的 global step（不包含）；`-1` 表示采集到训练结束 |
| `profile_ranks` | List[int] | `[-1]` | 要采集的 rank 列表；`[-1]` 表示所有 rank |
| `profile_level` | str | `level0` | 采集级别：<br>• `level_none`：关闭<br>• `level0`：基础算子耗时<br>• `level1`：增加 AICore 利用率、通信算子（推荐）<br>• `level2`：更详细（含缓存、内存等） |
| `profile_export_type` | str | `text` | 导出格式：<br>• `text`：文本格式 <br>• `db`：数据库格式 |
| `profile_data_simplification` | bool | `false` | 是否启用数据简化（减小 trace 文件体积） |
| `profile_with_cpu` | bool | `false` | 是否同时采集 CPU 活动（如数据加载、调度） |
| `profile_with_stack` | bool | `false` | 是否记录函数调用栈（便于定位代码位置） |
| `profile_with_memory` | bool | `false` | 是否采集 NPU 显存分配/释放事件，用于分析显存峰值、碎片化及内存泄漏|
| `profile_record_shapes` | bool | `false` | 是否记录张量 shape（用于分析显存和计算量） |
| `profile_save_path` | str | `./profile` | trace 文件保存目录（每个 rank 独立文件） |

### 3. 常用使用方法

这里给出两种配置示例，展示比较常用的使用场景：

1. 初步分析性能时，可以只采集0号卡的CPU信息，查看通信和计算时间占比，各类算子占比以及算子调度信息，推荐配置如下：

    ```yaml
    training:
      # ... 其他训练参数 ...
    
      # --- Profiling: 初步性能分析 ---
      profile: true
      profile_step_start: 5
      profile_step_end: 6 # 采集[5, 6), 区间左闭右开
      profile_ranks: [0] # 只采集0号卡
      profile_level: level1
      profile_with_cpu: true
      profile_save_path: ./profile_dir
    ```

2. 如果想要进一步查看算子内存占用信息以及算子详细调用情况，可以加入 profile_with_stack、profile_with_memory 和profile_record_shapes 等参数，但是这会导致数据膨胀，性能劣化。具体配置如下:

    ```yaml
    training:
      # ... 其他训练参数 ...
    
      # --- Profiling: 深度分析（含堆栈/内存/shape）---
      profile: true
      profile_step_start: 5
      profile_step_end: 6 # 采集[5, 6), 区间左闭右开
      profile_ranks: [0] # 只采集0号卡
      profile_level: level1
      profile_with_cpu: true
      profile_with_stack: true # 采集算子详细调用情况
      profile_with_memory: true # 采集内存占用信息
      profile_record_shapes: true # 记录张量 shape
      profile_save_path: ./profile_dir_with_stack
    ```

### 4. 输出文件

训练结束后，指定路径下将生成 profiling 文件，命名格式示例：

```shell
localhost.localdomain_3687609_20260129150104894_ascend_pt
```

该文件的目录结构如下：  

```shell
 localhost.localdomain_3687609_20260129150104894_ascend_pt
    ├─ASCEND_PROFILER_OUTPUT
    ├─logs
    └─PROF_000001_20260129150104896_KRPBOALLPQHOIAOA
        ├─device_0
        │  └─data
        ├─host
        │  └─data
        ├─mindstudio_profiler_log
        └─mindstudio_profiler_output
```

### 5. 可视化性能分析

以 MindStudio Insight 工具为例，对采集到的profiling进行分析。  
参考文档指导[MindStudio Insight 工具部署文档](https://gitcode.com/Ascend/msinsight#%E7%8E%AF%E5%A2%83%E9%83%A8%E7%BD%B2  ) 进行工具部署。  
将路径生成的profiling文件导入到工具中，进行性能的拆解分析。  
在这里我们简单介绍MindStudio Insight 工具主要使用的几个界面。  

#### 时间线（Timeline）  

时间线界面包含工具栏（区域一）、时间线树状图（区域二）、图形化窗格（区域三）和数据窗格（区域四）四个部分组成，如图界面所示。

![时间线](../../figures/fsdp2_profiling_guide/timeline.png)

#### 内存（Memory）  

内存界面由参数配置栏（区域一）、算子内存折线图（区域二）、内存申请/释放详情表（区域三）三个部分组成，如图所示。

![内存](../../figures/fsdp2_profiling_guide/memory.png)

#### 算子（Operator） 

算子界面由参数配置栏（区域一）、耗时百分比饼状图（区域二）、耗时统计及详情数据表（区域三）三个部分组成，如图所示  

![算子](../../figures/fsdp2_profiling_guide/operator.png)

这里只对工具的界面进行了简单介绍，如果想要更多了解工具的使用方法，可以参考MindStudio Insight的使用方法指导官方文档 
[MindStudio Insight系统调优](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/system_tuning.md#%E5%9F%BA%E7%A1%80%E5%8A%9F%E8%83%BD  ) 。

#### 更加定制化的性能拆解  

如果您希望更加定制化的profiling采集方式，可参考 [性能调优工具文档 性能数据采集和自动解析](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/devaids/Profiling/atlasprofiling_16_0033.html  ) 对采集代码进行修改，从而进行自定义采集和拆解分析。

---
