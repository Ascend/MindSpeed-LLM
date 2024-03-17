# AscendSpeed 安全声明

## 系统安全加固

- 用户可在运行系统配置时开启 ASLR（级别2）以提高系统安全性，保护系统随机化开启。  
可参考以下方式进行配置：

  ```
  echo 2 > /proc/sys/kernel/randomize_va_space
  ```

## 运行用户建议

- 基于安全性考虑，建议您在执行任何命令时，不建议使用root等管理员类型账户执行，遵循权限最小化原则。

## 文件权限控制

- 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
- 建议用户对训练所需文件、训练过程中保存的文件、用户个人的隐私数据、商业资产等敏感文件做好权限控制等安全措施，例如多用户共享数据集场景下的数据集文件写权限控制等，设定的权限建议参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)进行设置。
- AscendSpeed 中各类融合算子通过调用 PyTorch 中的 cpp_extension 特性进行编译，编译结果会默认缓存到 `~/.cache/torch_extensions` 目录下，建议用户根据自身需要，参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)对生成文件做好权限控制。
- 原生 Megatron-LM 以及 PyTorch 框架运行中所生成的文件权限依赖系统设定，如 Megatron-LM 生成的数据集索引文件、torch.save 接口保存的文件等。建议当前执行脚本的用户根据自身需要，对生成文件做好权限控制，设定的权限可参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)进行设置。
- 运行时 CANN 可能会缓存算子编译文件，存储在运行目录下的`kernel_meta_*`文件夹内，加快后续训练的运行速度，用户可根据需要自行对生成后的相关文件进行权限控制。
- 用户安装和使用过程需要做好权限控制，建议参考[附录A 文件（夹）各场景权限管控推荐最大值](#A-文件（夹）各场景权限管控推荐最大值)文件权限参考进行设置。如需要保存安装/卸载日志，可在安装/卸载命令后面加上参数 `--log <FILE>`， 注意对`<FILE>`文件及目录做好权限管控。

## 数据安全声明

- AscendSpeed 依赖 CANN 的基础能力实现 AOE 性能调优、算子 dump、日志记录等功能，用户需要关注上述功能生成文件的权限控制。

## 运行安全声明

- 建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
- AscendSpeed 在运行异常时会退出进程并打印报错信息，建议根据报错提示定位具体错误原因，包括设定算子同步执行、查看 CANN 日志、解析生成的 Core Dump 文件等方式。

## 公网地址声明
- AscendSpeed代码中包含公网地址声明如下表所示：

|      类型      |                                     开源代码地址                                      |                         文件名                         |             公网IP地址/公网URL地址/域名/邮箱地址             |          用途说明           |
| :------------: |:-------------------------------------------------------------------------------:|:---------------------------------------------------:| :----------------------------------------------------------: | :-------------------------: |
|  开源引入  | https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py |   AscendSpeed/ascendspeed/components/moe/gate.py    |          https://arxiv.org/pdf/2006.16668.pdf       | 开源引入TopKGate类实现 |
|  开源引入  | https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py |   AscendSpeed/ascendspeed/components/moe/gate.py    |          https://arxiv.org/pdf/2202.08906.pdf       | 开源引入apply_z_loss实现 |
|  开源引入  | https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py | AscendSpeed/ascendspeed/components/moe/moe_layer.py |          https://arxiv.org/pdf/2006.16668.pdf       | 开源引入MOELayer类实现 |



## 公开接口声明

- AscendSpeed采用python的装饰器，对原生megatron中的部分接口进行patch，使能原生meagtron在昇腾设备使用，AscendSpeed不暴露任何公开接口，用户实际使用时调用原生megatron接口。

## 通信安全加固

[通信安全加固说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA
)

## 通信矩阵

[通信矩阵说明](https://gitee.com/ascend/pytorch/blob/master/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E7%9F%A9%E9%98%B5%E4%BF%A1%E6%81%AF)

## 附录

### A-文件（夹）各场景权限管控推荐最大值

| 类型           | linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）            |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）            |
| 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             | 
| 日志文件(正在记录)                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）            |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |
