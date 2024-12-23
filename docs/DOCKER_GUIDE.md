
## 1.镜像下载（待补充昇腾社区的镜像下载地址）
通过uname -a确认自身系统是ubuntu_x86 或者 openeuler
根据需要下载对应的镜像,如下为下载链接：
https://www.hiascend.com/developer/ascendhub/detail/e26da9266559438b93354792f25b2f4a

## 2.镜像加载
```bash
# 挂载镜像,确认挂载是否成功                          
docker image list
```

## 3.创建镜像容器
注意当前默认配置驱动和固件安装在/usr/local/Ascend，如有差异请修改指令路径。
当前容器默认初始化npu驱动和CANN环境信息，如需要安装新的，请自行替换或手动source，详见容器的bashrc
```bash
# 挂载镜像
docker run -dit --ipc=host --network host --name 'llm_test' --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware  -v /usr/local/sbin/:/usr/local/sbin/ -v /home/:/home/ mindspeed-llm:tag
```

## 4.登录镜像并确认环境状态
```bash
# 登录容器
docker exec -it llm_test /bin/bash                           
# 确认npu是否可以正常使用，否则返回3.检查配置
npu-smi info
```

## 5.拉取配套版本
当前镜像推荐配套版本,用户可根据自己所需的版本配套，进行MindSpeed-LLM和MindSpeed的更新使用。
rc+序号为对应配套版本，镜像与分支名是配套的。例如：
1. 2024.rc2-arm/2024.rc2-x86 镜像版本匹配 [MindSpeed-LLM的1.0.RC2分支](https://gitee.com/ascend/MindSpeed-LLM/tree/1.0.RC2/)
2. 2024.rc3-arm/2024.rc3-x86 镜像版本匹配 [MindSpeed-LLM的1.0.RC3分支](https://gitee.com/ascend/MindSpeed-LLM/tree/1.0.RC3/)
3. ...

**注意：master为研发分支，无支持镜像。**

下面以MindSpeed-LLM的1.0.RC3分支进行配套说明。
镜像根据系统区分选择2024.rc3-arm/2024.rc3-x86。
```bash
# 从Gitee克隆MindSpeed-LLM仓库 (git checkout 1.0.RC3)
git clone https://gitee.com/ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 1.0.RC3
# 从Gitee克隆MindSpeed仓库(git checkout 4ea42a23)
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 4ea42a23
pip install -e .
cd ..
# 拉取megatron并切换对应版本放到MindSpeed-LLM下
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.6.0
cp -r megatron ../
```

## 6.单机以及多机模型的预训练任务运行
基于拉取的镜像和仓库代码，执行单机和多机的预训练任务，具体可参考[MindSpeed-LLM 使用指南](../examples/README.md)