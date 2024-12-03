# 安全声明
## 系统安全加固
1. 建议用户在系统中配置开启ASLR（级别2），又称**全随机地址空间布局随机化**，可参考以下方式进行配置：
    ```
    echo 2 > /proc/sys/kernel/randomize_va_space
    ```

## 运行用户建议
出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用MindSpeed-LLM。

## 文件权限控制
1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控。涉及场景如MindSpeed-LLM安装目录权限管控、多用户使用共享数据集权限管控，管控权限可参考表1进行设置。
3. MindSpeed-LLM在数据预处理中会生成训练数据，在训练过程会生成权重文件，文件权限默认640，用户可根据实际需求对生成文件权限进行进阶管控。

**表1 文件（夹）各场景权限管控推荐最大值**
| 类型          | linux权限参考最大值 |
| --------------- | --------------------|
| 用户主目录                          |    750（rwxr-x---）                |
| 程序文件（含脚本文件、库文件等）      |    550（r-xr-x---）                |
| 程序文件目录                        |    550（r-xr-x---）                |
| 配置文件                            |    640（rw-r-----）                |
| 配置文件目录                        |    750（rwxr-x---）                |
| 日志文件（记录完毕或者已经归档）      |    440（r--r-----）                |
| 日志文件（正在记录）                 |    640（rw-r-----）                |
| 日志文件记录                        |    750（rwxr-x---）                |
| Debug文件                          |    640（rw-r-----）                |
| Debug文件目录                      |    750 (rwxr-x---)                 |
| 临时文件目录                       |     750（rwxr-x---）                |
| 维护升级文件目录                    |    770（rwxrwx---）                |
| 业务数据文件                       |     640（rw-r-----）                |
| 业务数据文件目录                   |     750（rwxr-x---）                |
| 密钥组件、私钥、证书、密文文件目录   |     700（rwx------）                |
| 密钥组件、私钥、证书、加密密文      |     600（rw-------）                |
| 加解密接口、加解密脚本             |     500（r-x------）                |



## 数据安全声明

1. MindSpeed-LLM会在megatron中的checkpointing模块中保存模型文件，其中部分模型文件使用了风险模块pickle，可能存在数据风险。


## 运行安全声明

1. 建议用户结合运行资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
2. MindSpeed-LLM内部用到了pytorch,可能会因为版本不匹配导致运行错误，具体可参考pytorch[安全声明](https://gitee.com/ascend/pytorch#%E5%AE%89%E5%85%A8%E5%A3%B0%E6%98%8E)。


## 公网地址声明

| 类型     | 开源代码地址                                                                                                         | 文件名                                                                | 公网IP地址/公网URL地址/域名/邮箱地址                                                                                                                     | 用途说明      |
|--------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| 自研 | 不涉及                                                                                                            | modellink/model/language_model.py:85                                                        | https://github.com/kingoflolz/mesh-transformer-jax/                                                                          | 详情地址      |
| 自研 | 涉及                                                                                                            | tests/pipeline/common.py:6                                                          | https://github.com/microsoft/DeepSpeed/blob/master/tests/unit/common.py                                                                        | 源代码地址      |
| 自研 | 涉及                                                                                                            | tests/pipeline/conftest.py:6                                                        | https://github.com/microsoft/DeepSpeed/blob/master/tests/conftest.py                                                                          | 源代码地址      |
| 自研 | 不涉及                                                                                                            | examples/mcore/gemma/data_convert_gemma_pretrain.sh:5                               | https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered/resolve/main/wikipedia-cn-20230720-filtered.json?download=true  | 数据下载地址      |
| 自研 | 涉及                                                                                                            | modellink/tasks/rl/dpo.py:178                                                       | https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py                                                                          | 源代码地址      |
| 自研 | 不涉及                                                                                                            | modellink/core/transformer/moe/moe_utils.py:135                                     | https://arxiv.org/abs/2101.03961               | 论文地址      |

## 公开接口声明
MindSpeed-LLM 暂时未发布wheel包，无正式对外公开接口，所有功能均通过shell脚本调用。5个入口脚本分别为[pretrain_gpt.py](https://gitee.com/ascend/MindSpeed-LLM/blob/master/pretrain_gpt.py), [inference.py](https://gitee.com/ascend/MindSpeed-LLM/blob/master/inference.py), [evaluation.py](https://gitee.com/ascend/MindSpeed-LLM/blob/master/evaluation.py), [preprocess_data.py](https://gitee.com/ascend/MindSpeed-LLM/blob/master/preprocess_data.py) 和 [convert_ckpt.py](https://gitee.com/ascend/MindSpeed-LLM/blob/master/convert_ckpt.py)。


## 通信安全加固

[通信安全加固说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA
)

## 通信矩阵

[通信矩阵说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5%E4%BF%A1%E6%81%AF)
