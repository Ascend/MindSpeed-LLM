---
name: mindspeed-llm-dev-guide
description: "MindSpeed-LLM 开发规范。在仓库中新增 Python 源文件时使用，按开源目录布局添加版权和许可证声明。"
---

# MindSpeed-LLM 开发规范

在新增 `.py` 文件前，读取仓库根目录的 `.gitcode/opensource-repo-layout.yaml`。凡是位于 `directories` 中 `type: source` 的 `path` 或其子目录下的新建 `.py` 文件，都必须在文件头添加以下完整声明。当前源代码路径为 `mindspeed_llm/`，包括该路径下新建的 `__init__.py`。

```python
# Copyright 2024 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

文件有 shebang 或 `# coding=utf-8` 等编码声明时，将其保留在 Python 要求的位置，再添加版权和许可声明。保留原有的非华为版权、许可证及归属说明。提交前，按目录布局逐个确认本次新增的 Python 源文件都包含完整声明。
