MindSpeed-LLM支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析。使用时只需将相关参数添加至训练脚本中，运行脚本即可进行采集。主要参数及含义如下：


```shell
--profile                        # 打开profiling采集数据开关
--profile-export-type text       # 指定导出的性能数据结果文件格式, db, text, 默认text格式
--profile-data-simplification    # 使用数据精简模式
--profile-step-start  5          # 指定开启采集数据的步骤
--profile-step-end 6             # 指定结束采集数据的步骤，实际采集步数从start到end，包括第start步，不包括第end步
--profile-ranks 0 1 2 3 4        # 指定采集数据的卡号，默认为0，设置为-1时表示采集所有rank的profiling数据，可以设置为 0 1 2 3 4 5 6 7 8 9 列表指定全局卡号
--profile-level level2           # 数据采集水平，可选配置为：level0, level1, level2, 级别越高采集信息越多，默认为level0
--profile-with-cpu               # 是否采集CPU数据，加入参数采集
--profile-with-stack             # 采集指令运行堆栈，加入参数采集
--profile-with-memory            # 是否采集内存，加入参数采集
--profile-record-shapes          # 是否采集计算shape，加入参数采集
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```


常见使用场景有以下两种：

1. 初步分析性能时，可以只采集0号卡的CPU信息，查看通信和计算时间占比，各类算子占比以及算子调度信息，推荐配置如下：

```shell
--profile                        # 打开profiling采集数据开关
--profile-step-start  5          # 从第5步开始采集
--profile-step-end 6             # 从第6步结束，不包括第6步
--profile-ranks 0                # 采集0号卡的数据
--profile-level level1           # 采集上层应用数据，底层NPU数据，NPU计算算子耗时和通信算子耗时信息，CANN层AscendCL数据信息，NPU AI Core性能指标信息，通信小算子耗时信息
--profile-with-cpu               # 采集CPU数据，用于分析通信和调度
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```

2. 如果想要进一步查看算子内存占用信息以及算子详细调用情况，可以加入`--profile-with-stack`、`--profile-with-memory`和`--profile-record-shapes`等参数，但是这会导致数据膨胀，性能劣化。具体配置如下：

```shell
--profile                        # 打开profiling采集数据开关
--profile-step-start  5          # 从第5步开始采集
--profile-step-end 6             # 从第6步结束，不包括第6步
--profile-ranks 0                # 采集0号卡的数据
--profile-level level1           # 采集上层应用数据，底层NPU数据，NPU计算算子耗时和通信算子耗时信息，CANN层AscendCL数据信息，NPU AI Core性能指标信息，通信小算子耗时信息
--profile-with-cpu               # 采集CPU数据，用于分析通信和调度
--profile-with-stack             # 采集指令运行堆栈信息
--profile-with-memory            # 采集算子内存信息
--profile-record-shapes          # 采集算子数据维度信息
--profile-save-path ./profile_dir_with_stack    # profiling数据采集保存路径
```
