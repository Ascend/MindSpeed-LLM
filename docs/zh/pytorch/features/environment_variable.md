## 模型脚本环境变量介绍：

以上模型列表中脚本的环境变量说明具体如下：

| 环境变量名称                      | 环境变量描述                                                                             | 链接                                                                                                                  |
|-----------------------------|------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| ASCEND_LAUNCH_BLOCKING      | 1：强制算子采用同步模式运行会导致性能下降，会屏蔽task_queue队列优化功能；0：会增加内存消耗，有OOM的风险。                       | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_006.html                          |
| ASCEND_SLOG_PRINT_TO_STDOUT | 0：关闭日志打屏，日志采用默认输出方式，将日志保存在log文件中；1：开启日志打屏，日志将不会保存在log文件中，直接打屏显示。                   | https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0121.html              |
| HCCL_WHITELIST_DISABLE      | HCCL白名单开关,1-关闭/0-开启。                                                               | https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0085.html              |
| HCCL_CONNECT_TIMEOUT        | 设置HCCL超时时间，默认值为120。                                                                | https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0077.html              |
| CUDA_DEVICE_MAX_CONNECTIONS | 定义了任务流能够利用或映射到的硬件队列的数量。                                                            | 无                                                                                                                   |
| TASK_QUEUE_ENABLE           | 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化。                      | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_007.html                          |
| COMBINED_ENABLE             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景。                                | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_005.html                          |
| PYTORCH_NPU_ALLOC_CONF      | 内存碎片优化开关，默认是expandable_segments:False，使能时配置为expandable_segments:True，用于内存管理和碎片回收。 | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_012.html                          |
| ASCEND_RT_VISIBLE_DEVICES   | 指定哪些device对当前进程可见，支持一次指定一个或多个device ID。通过该环境变量，可实现不修改应用程序即可调整所用device的功能。          | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/maintenref/envvar/envref_07_0028.html |
| NPUS_PER_NODE               | 配置一个计算节点上使用的NPU数量。                                                                 | 无                                                                                                                 |  
| HCCL_SOCKET_IFNAME          | 指定hccl socket通讯走的网卡配置。                                                             | https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0075.html              |
| GLOO_SOCKET_IFNAME          | 指定gloo socket通讯走的网卡配置。                                                             | 无                                                                                                                   | 
| HCCL_LOGIC_SUPERPOD_ID      | 指定当前设备的逻辑超节点ID，如果走ROCE，不同多机超节点ID不同，0-N。                                            | https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/maintenref/envvar/envref_07_0100.html              |
| CPU_AFFINITY_CONF           | 开启粗/细粒度绑核。该配置能够避免线程间抢占，提高缓存命中，避免跨NUMA节点的内存访问，减少任务调度开销，优化任务执行效率。                    | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_033.html                          |
| NPU_ASD_ENABLE              | 0：关闭检测功能； 1：开启特征值检测功能，打印异常日志，不告警；2：开启，并告警；3：开启，告警，并在device侧info级别日志中记录过程数据。        | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_029.html                          |
| HCCL_ASYNC_ERROR_HANDLING   | 当使用HCCL用于通信时，0：不开启异步错误处理；1：开启异步错误处理，默认值为1                                          | https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_018.html                          |

