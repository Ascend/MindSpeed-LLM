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
