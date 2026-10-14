#!/bin/bash
# 交互式生成 scripts/.cc_env —— cron 安全的 agent 环境文件（默认 GLM Coding Plan，可切 DeepSeek）。
# 背景：cron 为非交互 shell，不加载 ~/.bashrc（setup-cc-ds.sh 写入的 ANTHROPIC_* 不生效），
# 且 cron 默认 PATH 不含 /usr/local/bin（claude 软链位置）。orchestrator 启动时显式
# source 本脚本生成的 .cc_env。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.cc_env"

# 非交互：环境变量 GLM_API_KEY 已设置则直接使用（批量部署/CI 场景）；否则交互读入
if [ -n "${GLM_API_KEY:-}" ]; then
    API_KEY="$GLM_API_KEY"
    echo "[INFO] 使用环境变量 GLM_API_KEY 非交互生成"
else
    # read 遇 EOF/无终端返回非零，给出明确报错而非被 set -e 静默杀掉
    if ! read -s -p "请输入你的智谱 GLM Coding Plan API Key（输入不会显示）: " API_KEY; then
        echo; echo "[ERROR] 读入失败（EOF/无终端）：非交互场景请用 GLM_API_KEY=<key> 环境变量注入" >&2
        exit 1
    fi
    echo
fi
if [ -z "$API_KEY" ]; then
    echo "[ERROR] API Key 不能为空" >&2
    exit 1
fi

umask 177   # 生成即 600，含 token 不外泄
cat > "$ENV_FILE" <<EOF
# 由 gen_cc_env.sh 生成；含 token，已被 .gitignore 排除，严禁入仓/外发。
# cron 非交互 shell 不加载 ~/.bashrc，orchestrator 启动时显式 source 本文件。
# Provider 切换（GLM 默认 / DeepSeek 备选）参考 .cc_env.example 注释。
export PATH=/usr/local/bin:\$PATH

# ---- GLM Coding Plan（智谱）----
export ANTHROPIC_BASE_URL='https://open.bigmodel.cn/api/anthropic'
export ANTHROPIC_AUTH_TOKEN='$API_KEY'
export ANTHROPIC_MODEL='glm-5.3[1m]'
export ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.3[1m]'
export ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.3[1m]'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='glm-5.3-flash'
export CLAUDE_CODE_SUBAGENT_MODEL='glm-5.3-flash'
export CLAUDE_CODE_AUTO_COMPACT_WINDOW='1000000'
export CLAUDE_CODE_EFFORT_LEVEL='max'
export API_TIMEOUT_MS='3000000'
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
EOF
chmod 600 "$ENV_FILE"

echo "[OK] 已生成 $ENV_FILE（权限 600）。orchestrator 将在启动时自动 source。"
echo "     模型：glm-5.3[1m]（主）/ glm-5.3-flash（子代理）；若 [1m] 后缀不被识别，"
echo "     升级 claude code 后重试，或改用 glm-5.3 / 官方默认 GLM-5.3-Flash。"
