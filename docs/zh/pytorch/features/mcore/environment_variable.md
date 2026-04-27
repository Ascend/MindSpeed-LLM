# 模型脚本环境变量介绍

以上模型列表中脚本的环境变量说明具体如下：

| 环境变量名称        | 环境变量描述        |
|-----------------------------|-------------------------------------|
| [ASCEND_LAUNCH_BLOCKING](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/ASCEND_LAUNCH_BLOCKING.md) | 1：强制算子采用同步模式运行会导致性能下降，会屏蔽task_queue队列优化功能；<br>0：会增加内存消耗，有OOM的风险。|
| [ASCEND_SLOG_PRINT_TO_STDOUT](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0121.html) | 0：关闭日志打屏，日志采用默认输出方式，将日志保存在log文件中；<br>1：开启日志打屏，日志将不会保存在log文件中，直接打屏显示。|
| [HCCL_WHITELIST_DISABLE](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0156.html) | HCCL白名单开关，1-关闭/0-开启。|
| [HCCL_CONNECT_TIMEOUT](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0077.html) | 设置HCCL超时时间，默认值为120。|
| CUDA_DEVICE_MAX_CONNECTIONS | 定义了任务流能够利用或映射到的硬件队列的数量。|
| [TASK_QUEUE_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/TASK_QUEUE_ENABLE.md) | 用于控制开启task_queue算子下发队列优化的等级：<br>0：关闭<br>1：开启Level 1优化<br>2：开启Level 2优化|
| [COMBINED_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/COMBINED_ENABLE.md) | 设置combined标志。<br>0：表示关闭此功能；<br>1：表示开启此功能，用于优化非连续两个算子组合类场景。|
| [PYTORCH_NPU_ALLOC_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/PYTORCH_NPU_ALLOC_CONF.md) | 内存碎片优化开关，默认是`expandable_segments:False`，使能时配置为`expandable_segments:True`，用于内存管理和碎片回收。|
| [ASCEND_RT_VISIBLE_DEVICES](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0028.html)| 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。通过该环境变量，可实现不修改应用程序即可调整所用Device的功能。|
| NPUS_PER_NODE | 配置一个计算节点上使用的NPU数量。| 
| [HCCL_SOCKET_IFNAME](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0075.html) | 指定HCCL Socket通讯走的网卡配置。|
| GLOO_SOCKET_IFNAME | 指定Gloo Socket通讯走的网卡配置。| 
| [HCCL_LOGIC_SUPERPOD_ID](https://www.hiascend.com/document/detail/zh/canncommercial/900/maintenref/envvar/envref_07_0100.html) | 指定当前设备的逻辑超节点ID，如果走ROCE，不同多机超节点ID不同，0-N。|
| [CPU_AFFINITY_CONF](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/CPU_AFFINITY_CONF.md) | 开启粗/细粒度绑核。该配置能够避免线程间抢占，提高缓存命中，避免跨NUMA节点的内存访问，减少任务调度开销，优化任务执行效率。|
| [NPU_ASD_ENABLE](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/NPU_ASD_ENABLE.md) | 0：关闭检测功能；<br>1：开启特征值检测功能，打印异常日志，不告警；<br>2：开启，并告警；<br>3：开启，告警，并在Device侧info级别日志中记录过程数据。|
| [HCCL_ASYNC_ERROR_HANDLING](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/HCCL_ASYNC_ERROR_HANDLING.md) | 当使用HCCL作为通信后端时，通过此环境变量可控制是否开启异步错误处理，默认值为1 。<br>0：不开启异步错误处理；<br>1：开启异步错误处理。|
