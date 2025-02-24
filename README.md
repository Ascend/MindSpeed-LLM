  <p align="center"> <img src="sources/images/logo.png" height="110px" width="500px"> </p>

<p align="center">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
    <a href="https://gitee.com/ascend/MindSpeed/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://gitee.com/ascend/MindSpeed">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a>
        <img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde">
    </a>
</p>
<p align="center">
        <b>简体中文</b> |
        <b><a href="README_en.md">English</a> </b>
</p>

MindSpeed-LLM，原仓名ModelLink，旨在为华为 [昇腾芯片](https://open.codehub.huawei.com/OpenBaize/Ascend/ascendspeed/files?ref=master&filePath=examples%2Fbaichuan%2Fpretrain_baichuan_zero_7B.sh&isFile=true) 上提供端到端的大语言模型方案, 包含模型，算法，以及下游任务。

---

## MindSpeed-LLM解决方案概览

当前MindSpeed-LLM支撑大模型使用功能:
* [制作预训练数据集](#jump11)/[制作指令微调数据集](#jump12)
* [预训练](#jump13)/[全参微调](#jump14)/[低参微调](#jump15)
* [流式推理/人机对话](#jump16)
* [评估基线数据集](#jump17)
* [加速算法/融合算子/并行策略](#jump18)
* [基于昇腾芯片采集Profiling数据](#jump19)
* [Huggingface与Megatron-LM权重转换](#jump20)
* [基于昇腾芯片的确定性计算功能](#jump21)

强化学习等特性持续研发中....

---

## MindSpeed-LLM版本维护策略

MindSpeed-LLM版本有以下五个维护阶段：

| **状态**            | **时间** | **说明**                                                               |
| ------------------- | -------- |----------------------------------------------------------------------|
| 计划                | 1—3 个月 | 计划特性                                                                 |
| 开发                | 3 个月   | 开发特性                                                                 |
| 维护                | 6-12 个月| 合入所有已解决的问题并发布版本，针对不同的MindSpeed-LLM版本采取不同的维护策略，常规版本和长期支持版本维护周期分别为6个月和12个月 |
| 无维护              | 0—3 个月 | 合入所有已解决的问题，无专职维护人员，无版本发布                                             |
| 生命周期终止（EOL） | N/A      | 分支不再接受任何修改                                                           |


MindSpeed-LLM已发布版本维护策略：

| **MindSpeed-LLM版本** | **维护策略** | **当前状态** | **发布时间**   | **后续状态**         | **EOL日期** |
|-----------------|-----------|--------|------------|------------------|-----------|
| 1.0.RC1         |  常规版本  | 维护   | 2024/03/30 | 预计2024/9/30起无维护  |           |
| 1.0.RC2         |  常规版本  | 维护   | 2024/06/30 | 预计2024/12/30起无维护 |           |

---

## 配套版本与支持模型

【需要注意模型使用时的配套环境版本，参考如下】
<table border="0">
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
    <th>软件安装指南</th>
    <th>推荐硬件形态</th>
  </tr>

  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">Ascend HDK 24.1.RC2</td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a>》</td>
    <td rowspan="8">Atlas 900 A2 PODc</td>
  </tr>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>Toolkit（开发套件）</td>
    <td rowspan="2">CANN 8.0.RC2</td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a>》</td>
  </tr>
  <tr>
    <td>Kernel（算子包）</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td rowspan="3">release v6.0.RC2</td>
    <td rowspan="3">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a>》</td>
  </tr>
  <tr>
    <td>torch_npu插件</td>
  </tr>
  <tr>
    <td>apex</td>
  </tr>
</table>

已安装好以上配套环境的镜像的获取和使用方法见[镜像使用指南](./docs/DOCKER_GUIDE.md)。

【基于现版本我们实测的性能情况统计如下（硬件信息：Atlas 900 A2 PODc）】

下述列表中支持的模型，我们在[examples](./examples/)文件夹中提供了各模型的训练脚本和readme说明，里面有详细的模型训练、推理、评估流程。

以下为开启 mc2 加速特性后的实测性能，该特性只在24RC2以上版本支持，本仓库代码层面默认关闭，若要使用，请参考首页[加速算法与融合算子](#加速算法与融合算子)章节

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>参数</th>
      <th>序列</th>
      <th>集群规模</th>
      <th>精度模式</th>
      <th>性能 </th>
      <th>参考性能 </th>
      <th>贡献方</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="1"><a href="examples/aquila/README.md">Aquila</a></td>
      <td>7B</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2849 </td>
      <td> 2874 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/aquila2/README.md">Aquila2</a></td>
      <td>7B</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 3323 </td>
      <td> 2673 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td>4K</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 854 </td>
      <td> 732 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan/README.md">Baichuan</a></td>
      <td>7B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2685 </td>
      <td> 2036 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1213 </td>
      <td> 862 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/baichuan2/README.md">Baichuan2</a></td>
      <td>7B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 2664 </td>
      <td> 3969 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td> 1x8</td>
      <td> BF16 </td>
      <td> 1668 </td>
      <td> 2062 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/bloom/README.md">Bloom</a></td>
      <td>7B1</td>
      <td>2K</td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 2034 </td>
      <td> 2525 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>176B</td>
      <td>2K</td>
      <td >12x8</td>
      <td> BF16 </td>
      <td> 100 </td>
      <td> 107 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/chatglm3/README.md">ChatGLM3</a></td>
      <td>6B</td>
      <td> 8K </td>
      <td> 1x8</td>
      <td> FP16 </td>
      <td> 4297 </td>
      <td> 4267 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/codellama/README.md">CodeLlama</a></td>
      <td>34B</td>
      <td>4K</td>
      <td> 2x8</td>
      <td> BF16 </td>
      <td> 837 </td>
      <td> 762 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/intern/README.md">InternLM</a></td>
      <td>7B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>BF16</td>
      <td> 2776 </td>
      <td> 2854 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td >65B</td>
      <td>2K</td>
      <td >4x8</td>
      <td> BF16 </td>
      <td> 341 </td>
      <td> 414 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama/README.md">LLaMA</td>
      <td>7B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 3600 </td>
      <td> 3804 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>FP16</td>
      <td> 1895 </td>
      <td> 2012 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
        <td>33B</td>
        <td>2K</td>
        <td>4x8</td>
        <td>FP16</td>
        <td>621</td>
        <td>776</td>
        <td>【昇腾】</td>
    </tr>
    <tr>
      <td>65B</td>
      <td>2K</td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 348 </td>
      <td> 426 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="examples/llama2/README.md">LLaMA2</td>
      <td>7B</td>
      <td>4K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 4200 </td>
      <td> 3850 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>13B</td>
      <td>4K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1990 </td>
      <td> 1920 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>34B</td>
      <td>4K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 749 </td>
      <td> 796 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td>4K</td>
      <td>4x8</td>
      <td>BF16 </td>
      <td> 420 </td>
      <td> 430 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/llama3/README.md">LLaMA3</td>
      <td>8B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2483 </td>
      <td> 2674 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>70B</td>
      <td> 8K </td>
      <td>8x8</td>
      <td>BF16 </td>
      <td> 283 </td>
      <td> 355 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="examples/qwen/README.md">Qwen</a></td>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2499 </td>
      <td> 2867 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>14B</td>
      <td>2K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 1560 </td>
      <td> 1578 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>72B</td>
      <td> 8K </td>
      <td>16x8</td>
      <td>BF16 </td>
      <td> 285 </td>
      <td> 345 </td>
      <td>【昇腾】</td>
    </tr>
    </tr>
       <tr>
      <td rowspan="7"><a href="examples/qwen15/README.md">Qwen1.5</a></td>
      <td> 0.5B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 22834 </td>
      <td> 25306 </td>
      <td>【社区】</td>
      <tr>
      <td> 1.8B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 13029 </td>
      <td> 12181 </td>
      <td>【社区】</td>
      <tr>
      <td> 4B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 5033 </td>
      <td> 5328 </td>
      <td>【社区】</td>
      <tr>
      <td> 7B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 2862 </td>
      <td> 2621 </td>
      <td>【社区】</td>
      <tr>
      <td> 14B </td>
      <td> 8K </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td> 1717 </td>
      <td> 1702 </td>
      <td>【社区】</td>
      <tr>
      <td> 32B </td>
      <td> 8K </td>
      <td> 4x8 </td>
      <td> BF16 </td>
      <td> 751 </td>
      <td> 708 </td>
      <td>【社区】</td>
      <tr>
      <td> 72B </td>
      <td> 8K </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td> 301 </td>
      <td> 317 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/yi/README.md">Yi</a></td>
      <td>34B</td>
      <td> 4K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 809 </td>
      <td> 730 </td>
      <td>【社区】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mixtral/README.md">Mixtral</a></td>
      <td>8x7B</td>
      <td> 32K</td>
      <td>2x8</td>
      <td>BF16 </td>
      <td> 487 </td>
      <td> 610 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/mistral/README.md">Mistral</a></td>
      <td>7B</td>
      <td> 32K</td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2806 </td>
      <td> 2734 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="examples/gemma/README.md">Gemma</a></td>
      <td>2B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 6821 </td>
      <td> 7602 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td>7B</td>
      <td> 8K </td>
      <td>1x8</td>
      <td>BF16 </td>
      <td> 2938 </td>
      <td> 2607 </td>
      <td>【昇腾】</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="examples/gpt3/README.md">GPT3</a></td>
      <td>175B</td>
      <td> 2K </td>
      <td> 16x8 </td>
      <td> FP16 </td>
      <td> 153 </td>
      <td> -- </td>
      <td>【社区】</td>
    </tr>
  </tbody>
</table>

---

## 加速算法与融合算子
MindSpeed-LLM支持张量并行、流水线并行、序列并行、重计算、分布式优化器等多种加速算法和融合算子，下表为各种加速特性对应的使能开关：

|                 加速特性                 |              使能参数              |
|:------------------------------------:|:------------------------------:|
|                 张量并行                 |  --tensor-model-parallel-size  |
|                流水线并行                 | --pipeline-model-parallel-size |
|               流水并行动态划分               |        --num-layer-list        |
|                 序列并行                 |      --sequence-parallel       |
|                 重计算                  |    --recompute-granularity     |
|                分布式优化器                |  --use-distributed-optimizer   |
|           DDP allreduce 掩盖           |     --overlap-grad-reduce      |
|           Flash attention            |        --use-flash-attn        |
|            Fused rmsnorm             |      --use-fused-rmsnorm       |
|             Fused swiglu             |       --use-fused-swiglu       |
|                 mc2                  |           --use-mc2            |
| Fused rotary <br/>position embedding |   --use-fused-rotary-pos-emb   |
|       Sliding window attention       |        --sliding-window        |


```bash
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layer-list 1,2,2,2,1 \
    --sequence-parallel \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 72 \
    --use-distributed-optimizer \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-fused-rotary-pos-emb \
    --use-mc2 \
    --sliding-window 4096 \
    ... \
    ...
```

```bash
注意：
如果需要开启 mc2，需保证:
1. 配套环境版本如本仓首页所述;
2. 将 modellink\arguments.py 中 validate_args_decorator 函数中的第283行进行注释
   #args.use_mc2 = False
```

---

## 基于昇腾芯片采集Profiling数据
MindSpeed-LLM支持基于昇腾芯片采集profiling数据，以提供对模型运行情况的分析，主要API如下：


```bash
--profile                        # 打开profiling采集数据开关
--profile-step-start  5          # 指定开启采集数据的步骤
--profile-step-end 6             # 指定结束采集数据的步骤，实际采集步数为 end-start，不包含end
--profile-ranks 0 1 2 3 4        # 指定采集数据的卡号，默认为0，可以设置为 0 1 2 3 4 5 6 7 8 9 列表指定全局卡号
--profile-level level2           # 数据采集水平，level0, 1, 2, 级别越高采集信息越多，默认为level0
--profile-with-cpu               # 是否采集CPU数据，加入参数采集
--profile-with-stack             # 采集指令运行堆栈，加入参数采集
--profile-with-memory            # 是否采集内存，加入参数采集
--profile-record-shapes          # 是否采集计算shape，加入参数采集
--profile-save-path ./profile_dir    # profiling数据采集保存路径
```

---

## 基于昇腾芯片的确定性计算功能
昇腾芯片默认采用了不确定计算加速模型训练，有时为了重复实验与对比实验需要确定性的计算结果，MindSpeed-LLM使能确定性计算的开关如下：

- 启动命令中加入开关
```shell
--use-deter-comp
```
- 环境变量中加入开关
```shell
export HCCL_DETERMINISTIC=True
```


---

## 致谢

MindSpeed-LLM由华为公司的下列部门联合贡献 ：
- 昇腾计算产品部
- 计算算法部
- 计算研究部
- 开源计算工具部
- 公共开发部
- 全球技术服务部

感谢来自社区的每一个PR，欢迎贡献 MindSpeed-LLM

---

## 安全声明

[MindSpeed-LLM安全声明](https://gitee.com/ascend/MindSpeed-LLM/wikis/%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)

# 免责声明

## 致MindSpeed-LLM使用者
1. MindSpeed-LLM提供的模型仅供您用于非商业目的。
2. 对于各模型，MindSpeed-LLM平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用MindSpeed-LLM模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在Gitee提交issue，我们将及时审视并解决。

## 致数据集所有者
如果您不希望您的数据集在MindSpeed-LLM中的模型被提及，或希望更新MindSpeed-LLM中的模型关于您的数据集的描述，请在Gitee提交issue，我们将根据您的issue要求删除或更新您的数据集描述。衷心感谢您对MindSpeed-LLM的理解和贡献。

## License声明
Ascend MindSpeed-LLM提供的模型，如模型目录下存在License的，以该License为准。如模型目录下不存在License的，以Apache 2.0许可证许可，对应许可证文本可查阅Ascend MindSpeed-LLM根目录。
