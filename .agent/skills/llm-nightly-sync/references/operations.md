# llm-nightly-sync 操作手册

> 各阶段完整 bash 命令。环境组装参照仓库内 `docs/zh/pytorch/training/install_guide.md`（方式三：Submodule 统一安装最贴近巡检 workspace 场景）。

## 0. 首次部署（推荐：一键部署）

```bash
# 交互式（会提示填 GLM API Key 等）：
bash scripts/deploy.sh

# 非交互（CI/重部署/容器重建后修复）：
GLM_API_KEY=<key> bash scripts/deploy.sh                # 部署但不接 cron
GLM_API_KEY=<key> INSTALL_CRON=1 bash scripts/deploy.sh # 部署并接 cron(0 6 * * *)
```

deploy.sh 为**交互向导**（①容器名/②conda env/③conda 根/④部署根/⑤PIPELINE 日志目录/
⑥历史经验导入/⑦GLM key/⑧gitcode 凭据/⑨cron 时间/⑩是否接 cron）：逐项显示默认值，
**直接回车即采用**；预 export 同名环境变量则跳过对应询问（CI 形态）。它会完成：
环境校验(容器/conda/PIPELINE 目录)→导入历史经验(可选，EXPERIENCE_DIR=旧 state
目录或交互输入；案例库+last_good，jsonl 去重合并，仅存本地不上仓)→容器内落实约定根
`/home/llm_nightly`（实际目录自动软链适配）→workspace 三仓 clone→`.cc_env` 生成（⑧填
token 时同步生成 .push_config.json，600）→容器内安装
claude code→冒烟验证→(可选)接入 `/etc/cron.d/llm-nightly-sync`。**幂等**，
容器重建后重跑即完成修复。可覆盖参数：DEPLOY_ROOT/DOCKER/CONDA_ENV/CONDA_HOME/
PIPELINE_LOG_BASE/CRON_SCHEDULE/INSTALL_CRON/EXPERIENCE_DIR/GLM_API_KEY/SKIP_PUSHCONF。

以下分步说明仅作排障/特殊环境参考：

```bash
# 1) 部署根目录：本 skill 统一以 /home/llm_nightly 为部署根（下同）；该根需对
#    llm_ci_q4 容器可见（宿主机 -v 挂载）；实际挂载点不同时，用环境变量
#    （REVERIFY_WS/PIPELINE_LOG_BASE 等）或等价路径替换，不影响 skill 自身
#    skill 随仓就位：/home/llm_nightly/MindSpeed-LLM/.agent/skills/llm-nightly-sync/
#    换机部署：git clone <你的fork>/MindSpeed-LLM.git /home/llm_nightly/MindSpeed-LLM

# 2) 组装重验 workspace（fetch_mrs delta 扫描 + 9 PATCH 重验；须在容器可见的部署根内）
mkdir -p /home/llm_nightly/workspace && cd /home/llm_nightly/workspace
git clone https://gitcode.com/ascend/MindSpeed.git
git clone https://gitcode.com/Ascend/FSDPTurbo.git
git clone https://gitcode.com/ascend/MindSpeed-LLM.git
# megatron_lm 为 pinned（不参与日常 delta），重验需要时再 clone 并 checkout core_v0.12.1

# 3) 安装（与 pipeline 环境一致的 conda env）
pip3 install -r MindSpeed/requirements.txt && pip3 install -e MindSpeed
pip3 install -e FSDPTurbo
ln -s Megatron-LM/megatron MindSpeed-LLM/megatron
pip3 install -r MindSpeed-LLM/requirements.txt

# 4) agent 运行环境（Step2 诊断用）
#    首次安装 Claude Code（全链路国内镜像）：bash setup-cc-ds.sh
#    （其 DeepSeek 变量段与巡检无关，key 可随意输入；巡检 API 端点以 .cc_env 为准）
#    ⚠️ cron 为非交互 shell 不加载 ~/.bashrc，必须另生成 cron 安全的 env 文件
#    （orchestrator 启动时显式 source，PATH 一并补全；默认 GLM Coding Plan）：
bash scripts/gen_cc_env.sh     # 交互输入智谱 GLM Coding Plan API Key，生成 scripts/.cc_env（600）
#    非交互（批量部署/CI）：GLM_API_KEY=<key> bash scripts/gen_cc_env.sh
#    （key 仍需人工在智谱平台一次性创建；本步仅免去交互输入）
#    模型：glm-5.3[1m]（主）/ glm-5.3-flash（子代理）；切换回 DeepSeek 见 .cc_env.example 注释

# 5) 接入 cron（宿主机 root crontab；每日 06:00 起轮询 PIPELINE 完成信号，完成即巡检）
#    推荐由 deploy.sh（⑩接入 cron）生成 /etc/cron.d/llm-nightly-sync，容器名与 conda
#    配置自动经 -e 传导；手动示例（容器名默认 llm_ci_q4，以 deploy ①项为准）：
# 0 6 * * * docker exec llm_ci_q4 bash /home/llm_nightly/MindSpeed-LLM/.agent/skills/llm-nightly-sync/scripts/llm_nightly_orchestrator.sh >> /home/llm_nightly/cron.log 2>&1

#### 执行环境：PIPELINE 同容器同 conda env（默认 llm_ci_q4 + llm_test，可配）

巡检直接跑在 PIPELINE 同容器同 conda env：`/home/pipeline_log`、`/data/ci` 均在该容器
内；**执行环境为容器内 conda env（默认 `/root/miniconda3/envs/llm_test`，torch/
torch_npu 所在，重验跑 test_run.sh 必需）**；skill 与 `.cc_env` 经部署根挂载
（`-v /home/llm_nightly:/home/llm_nightly` 或等价方式）直接共用，容器内已装 node v22.14.0 + claude code 2.1.247（华为云镜像 + npmmirror）。
⚠️ cron 非交互不读 .bashrc：conda init 与 CANN set_env 均不自动生效，orchestrator
已内置显式加载（conda activate $CONDA_ENV + source CANN/NNAL set_env.sh）。
容器名/conda 环境名/conda 根经 deploy 向导①②③项配置（环境变量
DOCKER/CONDA_ENV/CONDA_HOME，部署期经 cron 行 docker exec -e 传导至运行期，
旧名 CONDA_ENV_NAME 仍兼容）；Step2 同形态冒烟已通过（source .cc_env + IS_SANDBOX=1）。

```bash
# 宿主机 cron（推荐，deploy.sh ⑩自动生成；下例容器名/conda 变量以 deploy ①②③项为准）：
0 6 * * * docker exec -e CONDA_ENV=<conda环境名> -e CONDA_HOME=<conda根> <容器名> bash /home/llm_nightly/MindSpeed-LLM/.agent/skills/llm-nightly-sync/scripts/llm_nightly_orchestrator.sh >> /home/llm_nightly/cron.log 2>&1

# 手动补跑某日（容器内；<容器名> 默认 llm_ci_q4，与 deploy ①项一致）：
docker exec <容器名> bash -c 'bash /home/llm_nightly/MindSpeed-LLM/.agent/skills/llm-nightly-sync/scripts/llm_nightly_orchestrator.sh <YYYY-MM-DD>'
```

⚠️ 容器重建后 `/usr/local` 下 node/claude 不持久（部署根挂载区之外），
重建后重跑容器内安装段（华为云 node tar.xz + npmmirror 装 claude-code，见本仓部署记录），
或 docker commit 固化成镜像。

## 1. 手动触发 / 重跑历史日期

```bash
# 等待当晚 PIPELINE 完成后全自动（cron 路径）
bash scripts/llm_nightly_orchestrator.sh

# 重跑某日后处理（不等待，产物已在）
python3 scripts/nightly_postprocess.py --date 2026-08-24 --no-wait

# 指定非默认路径
python3 scripts/nightly_postprocess.py --date 2026-08-24 \
    --pipeline-log-base /home/pipeline_log --state-dir ./state --no-wait
```

退出码：`0` 全绿（last_good 已前移）/ `1` 需诊断（含 baseline 漂移、数据缺失类，见 attribution_report.json 的 overall 字段细分）/ `2` 环境 SKIP / `3` PIPELINE 未完成。

## 2. 重验失败用例（9 PATCH 循环内）

```bash
cd /home/llm_nightly/workspace/MindSpeed-LLM
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source "$CONDA_HOME/etc/profile.d/conda.sh" && conda activate "$CONDA_ENV"   # conda 根/环境名默认 /root/miniconda3/llm_test（torch/torch_npu 所在），以 deploy ②③项为准

# 只重跑失败的 ST 用例（test_run.sh 原生 -k/-f 过滤）
bash tests/test_run.sh --pipeline-st -k pretrain_qwen3_8b_general_cp_ring

# 只重跑失败的 UT 用例
bash tests/test_run.sh --pipeline-ut -f test_checkpoint.py
```

注意：重验前确认 workspace 各仓 checkout 与当日快照一致（`state/env_report.json` 的 commits）；修复提交用 `git commit --amend` 保持单 commit，验证通过后才推送。

**重验训练产物清理规则**：重验会在各仓根（含嵌套 clone）落 `output/checkpoints/*.distcp` 与 HF 导出 `*.safetensors`，单轮可达数十 GB 且因目录嵌套常为双份存放。orchestrator Step 0 每晚自动清理 **mtime > 5 天**的 `output/` 目录（只删仓根/嵌套仓根一层的 output，不碰 .git/源码/state/evidence；`DRY_RUN=1` 只报不删）；手动重验后也可直接 `rm -rf <repo>/output/`，验证结论以日志与 baseline 比对为准、不依赖 checkpoint 文件。

**⚠️ CLA 身份硬性要求**（commit 的 author 与 committer 必须同用 CLA 签署身份，否则 PR CLA 检查失败。签署身份 = GitCode 账号真实绑定邮箱；author_email 填自造 noreply 邮箱(如 user@noreply.gitcode.com)时平台无法关联账号(commit 显示 gitcode_user=None)、CLA 必失败）：

```bash
# 从 .push_config.json 读 repos.<fix_repo>.author_name/author_email 提交
A_NAME=<CLA-signed-name>; A_MAIL=<CLA-signed-email>
git -c user.name="$A_NAME" -c user.email="$A_MAIL" commit --amend --no-edit
# push 由 create_issue_pr.py 的 ensure_pushed 自动完成，agent 不手动 push
```

## 3. 手动跑 delta 扫描（7 FETCH_MRS）

```bash
python3 scripts/fetch_mrs.py --state-dir ./state --workspace /home/llm_nightly/workspace
# 产物 diff_report.json：每仓 commit 列表 + 变更符号 + 与 LLM import 面的交叉命中（severity high/medium）
```

## 4. 刷新 baseline（BASELINE_DRIFT 处理流程）

```bash
# 1) 用当日日志重提取指标（tests/test_tools/acquire_json.py）
cd /home/llm_nightly/workspace/MindSpeed-LLM
python3 -c "from tests.test_tools.acquire_json import transfer_logs_as_json; \
transfer_logs_as_json('/home/pipeline_log/<date>/[PIPELINE_ST]<name>.log', '/tmp/<name>.json')"

# 2) 人工确认漂移合理性（性能趋势/精度 margin），替换 baseline json
cp /tmp/<name>.json tests/pipeline/st/baseline/<name>.json   # 或 tests/st/baseline_results/

# 3) 连同说明提 PR（标注 margin 与漂移原因，勿夹带代码改动）
```

## 5. 诊断陷阱速查（DIAG 前必读）

1. **先看归因先验再读 diff**——attribution_report.json 的 suspects 已按用例特征排序，不要从头全量扫描
2. **穿透 ChildFailedError**——日志尾部常见 torchrun 包装错误，真实异常在其上方的 traceback 中
3. **零 delta 失败优先怀疑环境/数据/baseline**，不要假设「隐性破坏」
4. **修复落归因仓**：MindSpeed/FSDPTurbo 回归修上游并提 PR；不可控时 LLM 侧兼容 + 上游 Issue；LLM 自身回归修本仓且**特性须与看护用例一同上仓**（tests/README.md 门禁要求）
5. **margin 语义**：loss rel 0.02 / grad_norm 0.1 / throughput 0.05 / time 0.05 / memory 0.1（tests/test_tools/test_ci_st.py），小幅越线且无崩溃 → baseline 流程，不是代码 bug
6. **复发防护与短路**：同类失败先查 `state/nightly_case_library.jsonl`（postprocess 已自动匹配 similar_case）、历史 `evidence_*_ORIGINAL.log` 签名与 git reflog（昨日修复是否被覆盖）。**同时满足**：a) 当日失败签名与历史逐签名一致；b) 归因结论不变（同一 breaking commit 或同一既有缺陷，当日 delta 无新增可疑改动）；c) CI fork 存在覆盖该根因的 open PR —— 判定**老问题复发**，走短路：不重建修复分支、不重验、不建新 PR，diag_report.json 写 `recurrence_short_circuit: true` 与 `reuse_pr`，仅在既有 Issue 追加评论「`<date>` 复发，签名与历史一致，修复已在 PR !`<N>`，等待合入即可闭环」；DIAGNOSIS_AND_FIX.md 照常记录复发事实并指向老 PR。任一条不满足（新签名 / PR 已合并或关闭但用例仍失败=修复无效 / 无 open PR）→ 走完整修复流程

## 6. Issue/PR 与报告推送

```bash
# Step 2.5 幂等建 Issue/PR（orchestrator 已自动调；手动重跑：）
#   前置：state/diag_report.json（agent 诊断产出，含 fix_repo/root_cause）
#         state/fixed_tests.json —— 确认修复的真问题用例名列表，如 ["pretrain_qwen3_8b_general_cp_ring"]
python3 scripts/create_issue_pr.py --date 2026-08-27 --state-dir ./state --dry-run   # 先预检
python3 scripts/create_issue_pr.py --date 2026-08-27 --state-dir ./state             # 正式跑
# Issue 粒度 =（仓, 问题级）：同根因跨天只建一个，其余追加每日评论；PR 按 fork 分支查重复用
# 老问题复发短路：diag_report.json 含 recurrence_short_circuit=true 时，脚本只追加既有 Issue
#   评论（评论体含 reuse_pr），不推分支、不建新 PR —— 避免同一未合入修复夜夜重复出 PR

# 11 报告推送（orchestrator 已自动调；手动重跑：）
python3 scripts/push_sync_ci_results.py --date 2026-08-27 --state-dir ./state
# 默认 ci_repo.enabled=false 仅本地 SYNC_SUMMARY_<date>.md；置 true 后推 sync_build_results/YYYY-MM/DD/
```

修复完成后必须写 `DIAGNOSIS_AND_FIX.md`（中文正文，代码/日志摘录保留原文）落 `state/`，随 11 一并归档推送。
