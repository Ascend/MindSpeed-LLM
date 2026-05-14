<h1 align="center"> <img src="docs/zh/pytorch/figures/readme/logo.png" height="110px" width="500px"> </h1>

<p align="center">
    <a href="https://gitcode.com/ascend/MindSpeed-LLM/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitcode.com/ascend/MindSpeed-LLM">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>

# 简介

---

MindSpeed LLM：基于昇腾生态的大语言模型分布式训练套件，旨在为华为 [昇腾芯片](https://www.hiascend.com/) 生态合作伙伴提供端到端的大语言模型训练方案，包含分布式预训练、分布式指令微调以及对应的开发工具链，如：数据预处理、权重转换、在线推理、基线评估等。

**<small>注意 : 原仓名ModelLink更改为MindSpeed LLM，原包名modellink更改为mindspeed_llm </small>**

# 未来规划

---

未来规划会刷新在[MindSpeed LLM RoadMap](https://gitcode.com/Ascend/MindSpeed-LLM/issues/982)中，欢迎访问LLM最新规划动态。

# 社区会议

---
MindSpeed LLM系列TC及SIG会议安排请查看[Ascend会议中心](https://meeting.ascend.osinfra.cn/)

# 加入我们

---

为了交流开发经验、分享使用心得、及时获取项目更新，我们创建了MindSpeed LLM社区交流群。无论你是正在使用这个项目，还是有奇思妙想，都欢迎加入。

加入方式：

1. 直接扫码加入微信交流群（二维码7天有效，定期更新）
2. 添加昇腾开源小助手，获取群链接，进入MindSpeed LLM社区交流群

<div style="display: flex; justify-content: flex-start; gap: 30px; align-items: flex-start; padding-left: 60px;">
  <div style="text-align: center;">
    <div>MindSpeed LLM社区交流群</div>
    <img src="docs/zh/pytorch/figures/wechat/llm_group.jpg" width="150" alt="MindSpeed LLM 微信群">
  </div>
  <div style="text-align: center;">
    <div>昇腾开源小助手</div>
    <img src="docs/zh/pytorch/figures/wechat/ascend_assistant.jpg" width="150" alt="昇腾小助手 微信">
  </div>
</div>

# 最新消息

---

- [Apr. 25, 2026]: 🚀 [**DeepSeekV4-Flash** 定长数据预训练支持](./examples/mcore/deepseek4_flash/README.md) 【Prototype】

- [Apr. 16, 2026]: 🚀 [**MiniMax_M27** 模型支持](./examples/fsdp2/minimax_m27/) 【Prototype】

- [Mar. 28, 2026]: 🚀 [**Mamba3-block** demo模型支持](./examples/fsdp2/mamba3/) 【Prototype】
- [Mar. 27, 2026]: 🌴 MindSpeed LLM发布[v26.0.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/26.0.0)，支持core_v0.12.1版本
- [Mar. 10, 2026]: 🚀 MindSpeed LLM 模型下架[夕阳计划第二期](https://gitcode.com/Ascend/MindSpeed-LLM/issues/1224) 启动，感谢每一份曾经的贡献
- [Feb. 12, 2026]: 🚀 [**GLM5** 模型支持](./examples/mcore/glm5) 【Prototype】
- [Feb. 11, 2026]: 🚀 [**Step-3.5-Flash** 模型支持](./examples/fsdp2/step35) 【Prototype】
- [Feb. 10, 2026]: 🚀 [FSDP2训练后端上线，支持**Qwen3-Next** 模型](./examples/fsdp2/qwen3_next) 【Prototype】
- [Feb. 04, 2026]: 🚀 [**Qwen3-Coder-Next** 模型支持mcore后端](./examples/mcore/qwen3_coder_next) 【Prototype】
- [Jan. 28, 2026]: 🌴 [社区版镜像配套2.3.0分支上线](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.3.0/docs/pytorch/install_guide.md) 【Prototype】
- [Jan. 23, 2026]: 🌴 [社区版镜像配套2.2.0分支上线](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/docs/pytorch/install_guide.md) 【Prototype】
- [Jan. 16, 2026]: 🌴 MindSpeed LLM发布[v2.3.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.3.0)，支持core_v0.12.1版本
- [Dec. 24, 2025]: 🚀 **GPT-OSS** 模型支持
- [Dec. 11, 2025]: 🚀 **Qwen3-Next** 模型训练支持triton融合加速GDN模块计算 【Prototype】
- [Nov. 25, 2025]: 🚀 [数据/权重在线加载训练](./docs/zh/pytorch/training/pretrain/mcore/train_from_hf.md)
- [Nov. 14, 2025]: 🚀 **magistral** 模型支持 【Prototype】
- [Oct. 30, 2025]: 🚀 MindSpeed LLM 模型下架[夕阳计划](https://gitcode.com/Ascend/MindSpeed-LLM/issues/943) 启动，感谢每一份曾经的贡献
- [Oct. 28, 2025]: 🌴 MindSpeed LLM发布[v2.2.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.2.0)，支持core_v0.12.1版本
- [Oct. 16, 2025]: 🚀 **Qwen3-30B**支持DPO训练
- [Oct. 14, 2025]: 🚀 **DeepSeek-V3**预训练已支持基于 **[MindSpore AI框架](./docs/zh/mindspore/readme.md)** 运行
- [Sep. 16, 2025]: 🚀 **Qwen3-Next** 模型支持
- [Aug. 23, 2025]: 🚀 大参数模型[权重转换v2](./docs/zh/pytorch/tools/checkpoint_convert_hf_mcore_large_params.md)优化版本上线
- [Jul. 28, 2025]: 🚀 **GLM-4.5-Air** 系列模型同步首发支持
- [Jul. 25, 2025]: 🌴 MindSpeed LLM发布[v2.1.0分支](https://gitcode.com/Ascend/MindSpeed-LLM/tree/2.1.0)，支持core_r0.8.0版本
- [Jul. 10, 2025]: 🚀 **[DeepSeek-R1](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_deepseek_671b.md)** 系列功能逐步上线
- [May. 19, 2025]: 🚀 **Qwen3** 系列模型同步首发支持
- [Mar. 27, 2025]: 🚀 **[DeepSeek-R1-ZERO Qwen-7B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_qwen25_7b.md)** **[DeepSeek-R1-ZERO Qwen-32B](https://gitcode.com/ascend/MindSpeed-RL/blob/master/docs/zh/solutions/r1_zero_qwen25_32b.md)**
- [Mar. 26, 2025]: 🚀 **[DeepSeek-V3-671B模型全家桶](./examples/mcore/deepseek3)** 上线

注意：【Prototype】表示特性未经过充分验证，若使用存在问题请至[issue](https://gitcode.com/Ascend/MindSpeed-LLM/issues)反馈。

- [MindSpeed LLM率先支持MiniMax M2.7训练复现，加速模型迭代完成复杂任务](https://mp.weixin.qq.com/s/FWcQLu8InQvLh6YBd5Sq2w)
- [告别繁琐预处理！MindSpeed LLM推出Train_from_HF功能，实现加载即训练](https://mp.weixin.qq.com/s/kMUVWyCYLGKgceHzYXjigg)
- [极速响应！MindSpeed LLM无缝适配Step-3.5-Flash，解锁大规模MoE模型落地新可能](https://mp.weixin.qq.com/s/g7f_mpDgnvsc22P6XGxbmg)
- [MindSpeed LLM全新升级——支持FSDP训练后端，Qwen3-Next-Coder模型天级适配](https://mp.weixin.qq.com/s/Ihfc54P66bcO0r2j_mMX8A)
- [基于昇腾快速上手Qwen3-Coder-Next模型，手把手指南来了！](https://mp.weixin.qq.com/s/yo0RlfU9gIY20NKyYQp4QA)

# 目录结构

---

MindSpeed LLM 项目代码按照模块化设计原则进行组织，详细介绍参见 [项目导读](./docs/zh/project_guide.md)。

``` shell
MindSpeed-LLM/
 ├── ci                        # 门禁看护
 ├── configs                   # 配置文件目录
 ├── docs                      # 项目文档目录
 ├── examples                  # 模型示例脚本
 ├── mindspeed_llm             # 核心代码目录
 ├── tests                     # 测试用例目录
 ├── convert_ckpt.py           # 权重转换工具
 ├── convert_ckpt_v2.py        # 权重转换工具 v2
 ├── preprocess_data.py        # 数据预处理工具
 ├── pretrain_gpt.py           # 预训练流程
 ├── pretrain_mamba.py         # 预训练mamba模型流程
 ├── posttrain_gpt.py          # 后训练流程
 ├── preprocess_prompt.py      # 提示词预处理工具
 ├── rlhf_gpt.py               # RLHF 训练流程
 ├── train_fsdp2.py            # FSDP2 训练流程
 ├── inference.py              # 模型推理工具
 ├── evaluation.py             # 模型评估工具
 ├── setup.py                  # 安装配置文件
 ├── README.md                 # 项目说明文档
```

# 文档导航

---

[文档导读](./docs/zh/docs_guide.md)提供了 MindSpeed LLM 的完整使用指南，包含以下核心内容：

- **环境安装指导**：MindSpeed LLM 的安装配置说明
- **快速入门**：从环境安装到训练拉起的入门指导
- **模型清单**：PyTorch 和 MindSpore 框架支持的模型列表
- **特性清单**：性能优化和显存优化的特性说明
- **训练方案**：预训练、微调、推理、评估等完整方案
- **工具链**：权重转换、数据集处理、性能采集分析、确定性计算等工具使用说明

# 版本说明

---

详见[版本说明](docs/zh/release_notes_llm.md)。

# 安装

---

- 详细的安装步骤和环境配置请参考[MindSpeed LLM安装指导（基于PyTorch）](./docs/zh/pytorch/training/install_guide.md)。
- 详细的安装步骤和环境配置请参考[MindSpeed LLM安装指导（基于MindSpore）](./docs/zh/mindspore/install_guide.md)。

# 快速上手

---

指导开发者快速启动大语言模型的预训练和微调任务，具体的操作请参考：

- [快速入门（基于PyTorch框架）](./docs/zh/pytorch/training/quick_start.md)
- [快速入门（基于MindSpore框架）](./docs/zh/mindspore/quick_start.md)

# 支持模型

---

MindSpeed LLM目前已内置支持百余个业界常用LLM大模型的预训练与微调，支持模型清单可查看：

- [PyTorch框架模型支持列表](./docs/zh/pytorch/models/supported_models.md)
- [MindSpore框架模型支持列表](./docs/zh/mindspore/models/supported_models.md)

# 训练方案与特性

---

MindSpeed LLM包含分布式预训练、分布式微调等训练方案，具体介绍请参考[训练方案与特性说明](./docs/zh/pytorch/features/README.md)。

# 在线推理

---

<table>
  <thead>
    <tr>
      <th>特性</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="docs/zh/pytorch/training/inference/inference.md">流式推理 </a></td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/zh/pytorch/training/inference/chat.md"> Chat对话</a></td>
      <td>✅</td>
      <td>✅</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="docs/zh/pytorch/features/mcore/yarn.md"> yarn上下文扩展 </a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

# 开源数据集评测

---

仓库模型基线见[开源数据集评测基线](docs/zh/pytorch/training/evaluation/models_evaluation.md)
<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>数据集</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><a href="docs/zh/pytorch/training/evaluation/evaluation_guide.md">评测</a></td>
      <td><a href="https://people.eecs.berkeley.edu/~hendrycks/data.tar">MMLU</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/datasets/ceval/ceval-exam/tree/main">CEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google-research-datasets/boolean-questions">BoolQ</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh">BBH</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/ruixiangcui/AGIEval/tree/main">AGIEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
    <tr>
      <td><a href="https://github.com/openai/human-eval/tree/master/data">HumanEval</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【NAIE】</td>
    </tr>
  </tbody>
</table>

# 开发工具链

---

## 权重转换

MindSpeed LLM支持Huggingface、Megatron-core两种格式的权重互转，支持LoRA权重合并。权重转换特性参数和使用说明参考[权重转换](docs/zh/pytorch/tools/checkpoint_convert_hf_mcore.md)。

<table>
  <thead>
    <tr>
      <th>源格式</th>
      <th>目标格式</th>
      <th>切分特性</th>
      <th>LoRA</th>
      <th>贡献方</th>
      <th>Released</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Huggingface</td>
      <td>Megatron-core</td>
      <td>tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>❌</td>
      <td rowspan="3">【Ascend】</td>
      <td rowspan="3">❌</td>
    </tr>
    <tr>
      <td rowspan="2">Megatron-core</td>
      <td>Huggingface</td>
      <td></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Megatron-core</td>
      <td>tp、pp、dpp、vpp、cp、ep、loop layer</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## 数据预处理

MindSpeed LLM支持预训练、指令微调等多种任务的数据预处理。

<table>
  <thead>
    <tr>
      <th>任务场景</th>
      <th>数据集</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>预训练</td>
      <td><a href="docs/zh/pytorch/tools/data_process_pretrain.md">预训练数据处理</a></td>
      <td>✅</td>
      <td>✅</td>
      <td rowspan="3">【Ascend】</td>
    </tr>
    <tr>
      <td rowspan="2">微调</td>
      <td><a href="docs/zh/pytorch/tools/data_process_sft_alpaca_style.md">Alpaca风格</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="docs/zh/pytorch/tools/data_process_sft_sharegpt_style.md">ShareGPT风格</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </tbody>
</table>

## 性能采集

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1">性能采集</td>
      <td><a href="docs/zh/pytorch/tools/profiling.md">基于昇腾芯片采集 profiling 数据</a></td>
      <td>✅</td>
      <td>❌</td>
      <td>【Ascend】</td>
    </tr>
  </tbody>
</table>

## 高可用性

<table>
  <thead>
    <tr>
      <th>场景</th>
      <th>特性</th>
      <th>Mcore</th>
      <th>Released</th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">高可用性</td>
      <td><a href="docs/zh/pytorch/tools/deterministic_computation.md">基于昇腾芯片开启确定性计算</a></td>
      <td>✅</td>
      <td>❌</td>
      <td rowspan="2">【Ascend】</td>
    </tr>
  </tbody>
</table>

# 版本维护策略

---

MindSpeed LLM版本有以下五个维护阶段：

| **状态**            | **时间**  | **说明**                                                     |
| ------------------- | --------- | ------------------------------------------------------------ |
| 计划                | 1—3 个月  | 计划特性                                                     |
| 开发                | 3 个月    | 开发特性                                                     |
| 维护                | 6-12 个月 | 合入所有已解决的问题并发布版本，针对不同的MindSpeed LLM版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月  | 合入所有已解决的问题，无专职维护人员，无版本发布             |
| 生命周期终止（EOL） | N/A       | 分支不再接受任何修改                                         |

MindSpeed LLM已发布版本维护策略：

| **MindSpeed LLM版本** | **对应标签** | **维护策略** | **当前状态** | **发布时间** | **后续状态**              | **EOL日期** |
| --------------------- | ------------| ------------ | ------------ | ------------ | ---------------------- | ----------- |
| 2.3.0                 | v2.3.0       | 常规版本     | 维护         | 2025/12/30   | 预计2026/6/30起无维护   |             |
| 2.2.0                 | v2.2.0       | 常规版本     | 维护         | 2025/9/30    | 预计2026/3/30起无维护   |             |
| 2.1.0                 | v2.1.0       | 常规版本     | EOL         | 2025/6/30    | 生命周期终止  |     2025/12/30        |
| 2.0.0                 | v2.0.0       | 常规版本     | EOL          | 2025/3/30    | 生命周期终止           | 2025/9/30    |
| 1.0.0                 | v1.0.0       | 常规版本     | EOL          | 2024/12/30   | 生命周期终止           | 2025/6/30    |
| 1.0.RC3               | v1.0.RC3.0   | 常规版本     | EOL          | 2024/09/30   | 生命周期终止           | 2025/3/30    |
| 1.0.RC2               | v1.0.RC2.0   | 常规版本     | EOL          | 2024/06/30   | 生命周期终止           | 2024/12/30   |
| 1.0.RC1               | v1.0.RC1.0   | 常规版本     | EOL          | 2024/03/30   | 生命周期终止           | 2024/9/30    |
| bk_origin_23          | \            | Demo        | EOL          | 2023         | 生命周期终止           | 2024/6/30     |

# 安全声明

---

[MindSpeed LLM安全声明](https://gitcode.com/Ascend/MindSpeed-LLM/wiki/%E5%AE%89%E5%85%A8%E7%9B%B8%E5%85%B3%2F%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E.md)

# 免责声明

---

## 致MindSpeed LLM使用者

1. MindSpeed LLM提供的模型仅供您用于非商业目的。
2. MindSpeed LLM功能依赖的Megatron等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题修复依赖相关社区的贡献和反馈。您应理解，MindSpeed LLM仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。
3. 对于各模型，MindSpeed LLM平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
4. 如您在使用MindSpeed LLM模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitcode提交issue，我们将及时审视并解决。

## 致数据集所有者

如果您不希望您的数据集在MindSpeed LLM中的模型被提及，或希望更新MindSpeed LLM中的模型关于您的数据集的描述，请在Gitcode提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed LLM的理解和贡献。

# License声明

- MindSpeed LLM产品的使用许可证，具体请参见[LICENSE](LICENSE)。
- MindSpeed LLM工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/zh/LICENSE)。

# 贡献声明

---

如果您希望向MindSpeed LLM报告问题和贡献代码，具体请参见[贡献指南](./CONTRIBUTING.md)。

# FAQ

---

MindSpeed LLM仓库基本使用过程中常见问题可以参考[MindSpeed LLM FAQ](./docs/zh/FAQ.md)。FAQ中未能涵盖的问题，可以在仓库的[issues列表](https://gitcode.com/Ascend/MindSpeed-LLM/issues)中尝试寻找类似问题，或者提交新的issue。

# 致谢

---

MindSpeed LLM由华为公司的下列部门以及昇腾生态合作伙伴联合贡献 ：

华为公司：

- 计算产品线：Ascend
- 公共开发部：NAIE
- 全球技术服务部：GTS
- 华为云计算：Cloud

生态合作伙伴：

- 移动云（China Mobile Cloud）：大云震泽智算平台
- 工商银行软件开发中心大数据人工智能实验室

感谢来自社区的每一个PR，欢迎贡献 MindSpeed LLM。
