# ST测试使用指南

## 概述

本目录包含MindSpeed-LLM的系统测试(ST)用例，用于验证模型训练性能和稳定性。

## 测试结构

- `shell_scripts/`：包含各种模型和配置的测试脚本
- `baseline_results/`：包含测试基线结果，用于比较性能
- `test_st.py`：pytest测试文件，用于动态执行测试

## 使用方法

### 使用pytest执行测试

1. 确保已安装所有依赖：

   ```bash
   pip install -r ../../requirements.txt
   ```

2. 执行所有测试：

   ```bash
   cd ../../
   python -m pytest tests/st/test_st.py -v
   ```

   遇到失败用例则终止测试:

   ```bash
   cd ../../
   python -m pytest tests/st/test_st.py -v -x
   ```

3. 执行特定测试：

   ```bash
   python -m pytest tests/st/test_st.py::test_st_script -k "llama2" -v
   ```

### 测试流程

1. 测试框架会自动发现并执行`shell_scripts/`目录下的所有`.sh`脚本
2. 执行每个脚本并捕获输出
3. 将输出与`baseline_results/`目录下的基线结果进行比较
4. 验证性能指标是否在允许的误差范围内

### 测试指标

测试框架支持比较以下指标：

- 语言模型损失(lm loss)
- 梯度范数(grad norm)
- 吞吐量(throughput)
- 执行时间(time info)
- 内存使用(memo info)

## 注意事项

1. 测试需要在支持的硬件环境中运行
2. 首次运行时会预编译一些算子，可能需要一些时间
3. 测试结果会受到硬件配置和系统负载的影响
4. 如果需要更新基线结果，请修改`baseline_results/`目录下的对应文件
