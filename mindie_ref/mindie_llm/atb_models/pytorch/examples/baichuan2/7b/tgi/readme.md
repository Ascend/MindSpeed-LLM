# 启动脚本使用指南

```shell
bash start.sh ${device_ids} ${max_memory_gb} ${running_mode}
```

运行

```shell
```shell
bash start.sh 2 35 
```

调试

```shell
bash start.sh 2 35 debug
```

## 推荐参数

| 场景      | max_memory_gb |
|---------|---------------|
| Atalas推理系列产品 单芯 | 35            |
| Atalas推理系列产品 双芯 | 35            |
| Atlas 900 A2 PODc 单卡 | 57            |
| Atlas 900 A2 PODc 双卡 | 57            |