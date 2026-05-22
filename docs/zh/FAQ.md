# MindSpeed LLM FAQ

- **问题1**
  Q：训练日志显示"Checkpoint path not found"？
  A：检查`CKPT_LOAD_DIR`是否指向正确的权重转换后路径，确认文件夹内包含`.ckpt`或`.bin`文件，否则请更正权重路径的设置。

- **问题2**
  Q：显示数据集加载"out of range"？
  A：微调脚本未能读取到数据集，请检查脚本中`DATA_PATH`是否符合示例的规范。

  ![img_3.png](./pytorch/figures/quick_start/img_3.png)

- **问题3**
  Q：没有生成运行日志文件？
  A：需要自行创建logs文件夹。

  ![img_1.png](./pytorch/figures/quick_start/img_1.png)
