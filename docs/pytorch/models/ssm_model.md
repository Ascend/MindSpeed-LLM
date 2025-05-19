## MindSpeed-LLM 预置ssm大模型

`认证`【Pass】表示经过昇腾官方版本测试的模型，【Test】表示待测试模型
<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>下载链接</th>
      <th>脚本位置</th>
      <th>序列</th>
      <th>实现</th>
      <th>集群</th>
      <th>贡献方</th>
      <th>认证</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/">Mamba2</a></td>
      <td><a href="https://huggingface.co/state-spaces/mamba2-2.7b/tree/main">2.7B</a></td>
      <td rowspan="2"><a href="../../examples/mcore/mamba2">mamba2</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 1x8</td>
      <td>【Ascend】</td>
      <td>【test】</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/nvidia/mamba2-8b-3t-4k/tree/main">8B</a></td>
      <td>4K</td>
      <th>Mcore</th>
      <td> 1x8</td>
      <td>【Ascend】</td>
      <td>【test】</td>
    </tr>      
    <tr>
      <td rowspan="2"><a href="https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k">Mamba2Hybrid</a></td>
      <td><a href="https://huggingface.co/nvidia/mamba2-hybrid-8b-3t-4k/tree/main">8B</a></td>
       <td><a href="../../examples/mcore/mamba2">mamba2</a></td>
      <td> 4K</td>
      <th>Mcore</th>
      <td>1x8</td>
      <td>【Ascend】</td>
      <td>【test】</td>
    </tr>   
  </tbody>
</table>

## 以上模型脚本环境变量声明：
HCCL_CONNECT_TIMEOUT：设置HCCL超时时间，默认值为120<br>
CUDA_DEVICE_MAX_CONNECTIONS：定义了任务流能够利用或映射到的硬件队列的数量<br>
PYTORCH_NPU_ALLOC_CONF：内存碎片优化开关，默认是expandable_segments:False，使能时expandable_segments:True<br>
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量<br>