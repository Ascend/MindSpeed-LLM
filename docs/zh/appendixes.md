# 附录

## 常见问题

- **问题1**  
  Q：训练日志显示"Checkpoint path not found"？  
  A：检查`CKPT_LOAD_DIR`是否指向正确的权重转换后路径，确认文件夹内包含`.ckpt`或`.bin`文件，否则路径错误请更正权重路径。 

- **问题2**  
  Q：显示数据集加载out of range？  
  A：微调脚本，没有读取到数据集，请检查脚本中DATA_PATH是否符合上面示例的规范。  

  ![img_3.png](./pytorch/figures/quick_start/img_3.png)

- **问题3**  
  Q：没有生成运行日志文件？  
  A：需要自行创建logs文件夹。  

  ![img_1.png](./pytorch/figures/quick_start/img_1.png)

## 加入昇腾开发者生态

- 🌐 **社区资源**：访问[昇腾开源社区](https://gitcode.com/ascend)获取最新模型支持
- 📈 **性能优化**：参考[MindSpeed Profiling](./pytorch/features/profiling.md)分析瓶颈
- 💡 **定制需求**：通过`model_cfg.json`扩展自定义模型


## 线性度

基于 `GPT3-175B` 稠密大模型，从128颗 NPU 扩展到 7968颗 NPU 进行 MFU 与线性度实验，下图是实验数据：

<p align="center"> <img src="./pytorch/figures/readme/linearity&mfu.png" height="490px" width="715px"> </p>

图中呈现了对应集群规模下的 `MFU` 值与集群整体的 `线性度`情况. 计算公式已经放到社区，点击链接可进行参考：[MFU计算公式](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E6%9C%AF%E8%AF%AD%E5%AE%9A%E4%B9%89%2F%E5%A4%A7%E6%A8%A1%E5%9E%8B%20MFU%20%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.md)，[线性度计算公式](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E6%9C%AF%E8%AF%AD%E5%AE%9A%E4%B9%89%2F%E7%BA%BF%E6%80%A7%E5%BA%A6%E5%85%AC%E5%BC%8F.md).
