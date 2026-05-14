# 贡献指南

**1. 报告问题**

- 如果您发现任何问题，请先查看仓库的[issues列表](https://gitcode.com/Ascend/MindSpeed-LLM/issues)，尝试寻找类似问题或解决方案。

- 如果现有[issues列表](https://gitcode.com/Ascend/MindSpeed-LLM/issues)中没有您遇到的问题，可以[提交一个新的issue](https://gitcode.com/Ascend/MindSpeed-LLM/issues/create/choose)，并尽量提供清晰的问题描述、复现步骤与环境信息。

**2. 性能优化与新功能**

- 有关性能优化的提案，请在提交issue时使用`Performance`标签，并描述性能优化特性和使用场景。

- 有关新功能的建议或讨论，请在提交issue时使用`Feature`标签，并描述背景、预期和方案。

**3. 贡献代码流程**

若您希望提交代码改动，请遵循以下简要步骤：

- 在您的个人分支上开发并提交，然后向本项目仓库发起Pull Request（PR），常见场景PR类型：`feat`（功能/脚本）/ `fix`（Bug修复）/ `docs`（文档修改）；

- 发起PR后，请同步创建issue并关联该PR，issue标签请与PR类型对应：`Feature`（功能讨论）/ `Bug`（Bug反馈）/ `Doc`（文档问题）。我们将在issue中与您讨论方案是否采纳，提出意见或修改要求，您也可以通过该issue跟进后续流程进展；

- 根据讨论进行修改，并更新PR；

- 在评论区输入`compile`以触发门禁流水线（CI）；

- 当PR的CI通过且获得足够的标签后，仓库Committer将进行最终审核，并合入在研分支。

感谢您的参与与贡献！我们期待与您共同推动项目发展。
