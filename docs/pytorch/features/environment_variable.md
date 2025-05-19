## 模型脚本环境变量介绍：

以上模型列表中脚本的环境变量说明具体如下：

| 环境变量名称                      | 环境变量描述                                                                    |
|-----------------------------|---------------------------------------------------------------------------|
| ASCEND_LAUNCH_BLOCKING      | 将Host日志输出到串口,0-关闭/1-开启。                                                   |
| ASCEND_SLOG_PRINT_TO_STDOUT | 设置默认日志级别,0-debug/1-info/2-warning/3-error。                                |
| HCCL_WHITELIST_DISABLE      | HCCL白名单开关,1-关闭/0-开启。                                                      |
| HCCL_CONNECT_TIMEOUT        | 设置HCCL超时时间，默认值为120。                                                       |
| CUDA_DEVICE_MAX_CONNECTIONS | 定义了任务流能够利用或映射到的硬件队列的数量。                                                   |
| TASK_QUEUE_ENABLE           | 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化。             |
| COMBINED_ENABLE             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景。                       |
| PYTORCH_NPU_ALLOC_CONF      | 内存碎片优化开关，默认是expandable_segments:False，使能时expandable_segments:True。        |
| ASCEND_RT_VISIBLE_DEVICES   | 指定哪些Device对当前进程可见，支持一次指定一个或多个Device ID。通过该环境变量，可实现不修改应用程序即可调整所用Device的功能。 |
| NPUS_PER_NODE               | 配置一个计算节点上使用的NPU数量。                                                        |
| HCCL_SOCKET_IFNAME               | 指定hccl socket通讯走的网卡配置。                                                    |
| GLOO_SOCKET_IFNAME               | 指定gloo socket通讯走的网卡配置。                                                |