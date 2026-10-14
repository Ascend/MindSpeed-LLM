#!/bin/bash
# ============================================
# llm-nightly-sync 一键部署脚本（交互向导版）
# ============================================
# 约定默认值：部署根 /home/llm_nightly、容器 llm_ci_q4、conda llm_test、
#             PIPELINE 日志 /home/pipeline_log、cron "0 6 * * *"（06:00 起轮询完成信号）。
# 交互式：逐项询问，每项显示 [默认值]，直接回车即采用默认。
# 非交互：预先 export 同名环境变量即跳过询问（cron/CI 推荐）。
#
# 用法：
#   bash scripts/deploy.sh                                  # 全交互向导
#   GLM_API_KEY=<key> INSTALL_CRON=1 bash scripts/deploy.sh # 全默认+装+接 cron
# 环境变量（设置即跳过对应询问）：DEPLOY_ROOT / DOCKER / CONDA_ENV / CONDA_HOME /
#   PIPELINE_LOG_BASE / CRON_SCHEDULE / INSTALL_CRON / EXPERIENCE_DIR / GLM_API_KEY /
#   SKIP_PUSHCONF=1（跳过 gitcode 凭据询问）
# 幂等：任意步骤已就绪则跳过；容器重建后重跑本脚本即完成修复。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLM_REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd)"   # skill 所在 MindSpeed-LLM 仓根（scripts→skill→skills→.agent→仓根）
SKIP_PUSHCONF="${SKIP_PUSHCONF:-0}"   # 预定义默认（set -u）

info(){ echo -e "\033[32m[INFO]\033[0m $*"; }
warn(){ echo -e "\033[33m[WARN]\033[0m $*"; }
fail(){ echo -e "\033[31m[FATAL]\033[0m $*"; exit 1; }

TTY=0; [ -t 0 ] && TTY=1

# ask 变量 提示语 默认值 —— 环境变量已设置则直接用（不问）；tty 下询问，回车=默认；非 tty 用默认
ask(){ local v=$1 prompt=$2 def=$3 in=""
    if [ -n "${!v:-}" ]; then return; fi
    if [ "$TTY" = 1 ]; then read -rp "$prompt [默认: $def]（回车采用）: " in; fi
    [ -n "$in" ] || in="$def"
    printf -v "$v" '%s' "$in"; }
ask_yn(){ local v=$1 prompt=$2 def=$3 in=""
    if [ -n "${!v:-}" ]; then return; fi
    if [ "$TTY" = 1 ]; then read -rp "$prompt [默认: $def]（回车采用）: " in; fi
    case "$in" in y|Y|yes|1) printf -v "$v" '1';; *) printf -v "$v" '0';; esac; }

# ---------------- 交互向导：逐项收集配置 ----------------
echo "======== llm-nightly-sync 部署向导（每项直接回车采用默认值）========"
DOCKER_DEF=llm_ci_q4; CONDA_ENV_DEF=llm_test; CONDA_HOME_DEF=/root/miniconda3
DEPLOY_ROOT_DEF="/home/llm_nightly"   # 约定部署根（真实目录；容器 /home 全挂载时容器内同路径直达）
PIPELINE_LOG_BASE_DEF=/home/pipeline_log
CRON_SCHEDULE_DEF="0 6 * * *"

ask DOCKER        "① 执行容器名（PIPELINE 所在 docker）" "$DOCKER_DEF"
ask CONDA_ENV     "② 容器内 conda 环境名（torch/torch_npu 所在）" "$CONDA_ENV_DEF"
ask CONDA_HOME    "③ 容器内 conda 安装根" "$CONDA_HOME_DEF"
ask DEPLOY_ROOT   "④ 实际部署根（skill/workspace 所在；≠/home/llm_nightly 时容器内自动软链）" "$DEPLOY_ROOT_DEF"
ask PIPELINE_LOG_BASE "⑤ PIPELINE 产物目录（容器内）" "$PIPELINE_LOG_BASE_DEF"

# 飞轮经验导入（本地运行数据，不上仓）
STATE_DIR_LOCAL="$SCRIPT_DIR/../state"; mkdir -p "$STATE_DIR_LOCAL"
if [ -z "${EXPERIENCE_DIR:-}" ] && [ "$TTY" = 1 ]; then
    read -rp "⑥ 是否导入过去部署的巡检经验（旧 state 目录路径，回车跳过）: " EXPERIENCE_DIR
fi
import_experience(){
    local src="$1" imported=0
    [ -d "$src" ] || { warn "经验目录不存在：$src，跳过导入"; return; }
    for f in nightly_case_library.jsonl last_good.json; do
        [ -f "$src/$f" ] || continue
        if [ -f "$STATE_DIR_LOCAL/$f" ]; then
            if [[ "$f" == *.jsonl ]]; then
                cat "$STATE_DIR_LOCAL/$f" "$src/$f" | awk '!seen[$0]++' > "$STATE_DIR_LOCAL/$f.tmp" \
                    && mv "$STATE_DIR_LOCAL/$f.tmp" "$STATE_DIR_LOCAL/$f"
                info "经验合并（行级去重）：$f"
            else
                info "$f 本地已存在，保留本地（如需覆盖请手动处理）"
            fi
        else
            cp "$src/$f" "$STATE_DIR_LOCAL/$f"; info "经验导入：$f"; imported=1
        fi
    done
    [ $imported -eq 0 ] && [ ! -f "$src/nightly_case_library.jsonl" ] && [ ! -f "$src/last_good.json" ] \
        && warn "$src 中未找到 nightly_case_library.jsonl / last_good.json，无可导入经验"
    return 0; }
[ -n "${EXPERIENCE_DIR:-}" ] && { info "导入历史巡检经验：$EXPERIENCE_DIR"; import_experience "$EXPERIENCE_DIR"; }

# ⑦ GLM API Key：.cc_env 已存在沿用；否则 gen_cc_env.sh 交互/环境变量注入
if [ ! -f "$SCRIPT_DIR/.cc_env" ] && [ -z "${GLM_API_KEY:-}" ] && [ "$TTY" != 1 ]; then
    fail "非交互环境且无 .cc_env：请设置 GLM_API_KEY=<key> 后重跑"
fi

# ⑧ gitcode 凭据（用户名+令牌）与 CI fork 闭环仓；可跳过
if [ "$SKIP_PUSHCONF" != "1" ] && [ ! -f "$SCRIPT_DIR/.push_config.json" ]; then
    GC_TOKEN=""
    if [ "$TTY" = 1 ]; then
        read -rp "⑧ gitcode 访问令牌（用户名+令牌方式；回车跳过→降级 NEEDS_HUMAN）: " GC_TOKEN
    fi
    if [ -n "$GC_TOKEN" ]; then
        ask GC_USER  "   gitcode 用户名（与令牌配对，也是凭据注入用）" ""
        ask GC_CI_FORK "   CI fork 仓地址（自动修复 Issue/PR 提到此 fork，每次先同步上游 master）" \
            "https://gitcode.com/${GC_USER}/MindSpeed-LLM_ci.git"
        ask GC_NAME  "   CLA 签署姓名（三仓共用，commit 身份）" ""
        ask GC_MAIL  "   CLA 签署邮箱" ""
        sed -e "s|<your-gitcode-access-token>|$GC_TOKEN|g" \
            -e "s|<your-gitcode-username>|$GC_USER|g" \
            -e "s|https://gitcode.com/Rostellaria/MindSpeed-LLM_ci.git|$GC_CI_FORK|" \
            -e "s|<CLA-signed-name>|$GC_NAME|g" -e "s|<CLA-signed-email>|$GC_MAIL|g" \
            "$SCRIPT_DIR/.push_config.example.json" > "$SCRIPT_DIR/.push_config.json"
        chmod 600 "$SCRIPT_DIR/.push_config.json"
        info "已生成 .push_config.json（600，gitignore 排除；CI fork=$GC_CI_FORK）"
    else
        warn "gitcode 凭据未配置：Issue/PR 闭环降级（本地报告+NEEDS_HUMAN）；之后可重跑 deploy 或手动放置"
    fi
fi

ask CRON_SCHEDULE "⑨ 巡检 cron 时间（默认 06:00 起轮询 PIPELINE 完成信号）" "$CRON_SCHEDULE_DEF"
ask_yn INSTALL_CRON "⑩ 是否接入 cron（写入 /etc/cron.d/llm-nightly-sync）" "0"
CANONICAL=/home/llm_nightly   # 约定部署根（容器内必须可用）
echo "================================================================"

# ---------------- 1. 环境校验 ----------------
info "部署参数：DEPLOY_ROOT=$DEPLOY_ROOT DOCKER=$DOCKER CONDA_ENV=$CONDA_ENV CRON=${INSTALL_CRON}@[$CRON_SCHEDULE]"
command -v docker >/dev/null 2>&1 || fail "宿主机无 docker"
docker inspect "$DOCKER" >/dev/null 2>&1 || fail "容器 $DOCKER 不存在（docker ps -a 检查；或用 DOCKER=<名> 重跑）"
[ -d "$LLM_REPO/.git" ] || fail "未定位到 MindSpeed-LLM 仓根：$LLM_REPO"

dex(){ docker exec "$DOCKER" bash -c "$1"; }
dex "[ -d $CONDA_HOME/envs/$CONDA_ENV ]" >/dev/null 2>&1 \
    && info "conda env OK：$CONDA_HOME/envs/$CONDA_ENV" \
    || warn "容器内未找到 conda env $CONDA_ENV（重验将失败，请检查②③项）"
dex "[ -d $PIPELINE_LOG_BASE ]" >/dev/null 2>&1 \
    && info "PIPELINE 日志目录 OK：$PIPELINE_LOG_BASE" \
    || warn "容器内无 $PIPELINE_LOG_BASE（确认⑤项，或 PIPELINE 尚未在此机运行）"

# ---------------- 2. 容器内落实约定部署根 ----------------
if [ "$DEPLOY_ROOT" = "$CANONICAL" ]; then
    dex "mkdir -p $CANONICAL" || fail "容器内创建 $CANONICAL 失败"
    info "部署根即约定路径 $CANONICAL"
else
    [ -d "$DEPLOY_ROOT" ] || fail "DEPLOY_ROOT 不存在：$DEPLOY_ROOT（检查④项）"
    dex "mkdir -p \$(dirname $CANONICAL) && ln -sfn $DEPLOY_ROOT $CANONICAL" \
        || fail "容器内软链 $CANONICAL -> $DEPLOY_ROOT 失败"
    info "容器内已建约定路径：$CANONICAL -> $DEPLOY_ROOT"
fi
dex "[ -f $CANONICAL/$(basename $LLM_REPO)/.agent/skills/llm-nightly-sync/SKILL.md ]" >/dev/null 2>&1 \
    && info "skill 经 $CANONICAL 容器内可见" \
    || fail "skill 在容器内不可见（检查④项：部署根需包含本仓）"

# ---------------- 3. workspace 组装（幂等） ----------------
WS="$DEPLOY_ROOT/workspace"; mkdir -p "$WS"
for repo in "https://gitcode.com/ascend/MindSpeed.git" \
            "https://gitcode.com/Ascend/FSDPTurbo.git" \
            "https://gitcode.com/ascend/MindSpeed-LLM.git"; do
    name=$(basename "$repo" .git)
    if [ -d "$WS/$name/.git" ]; then
        info "workspace/$name 已存在，跳过 clone（更新：git -C $WS/$name pull）"
    else
        info "clone $name ..."
        git clone -q "$repo" "$WS/$name" || warn "clone $name 失败（网络？），delta 扫描将报 clone 缺失"
    fi
done
# megatron_lm 为 pinned（不参与日常 delta），重验需要时按 operations.md 手动 clone

# ---------------- 4. agent 环境 .cc_env ----------------
if [ -z "${GLM_API_KEY:-}" ]; then
    [ -f "$SCRIPT_DIR/.cc_env" ] && info ".cc_env 已存在（沿用；如需更换 key 删后重跑）" \
        || { info "⑦ 生成 agent 环境（GLM Coding Plan）："; bash "$SCRIPT_DIR/gen_cc_env.sh" || fail "gen_cc_env 失败"; }
else
    info "使用 GLM_API_KEY 环境变量生成 .cc_env"
    GLM_API_KEY="$GLM_API_KEY" bash "$SCRIPT_DIR/gen_cc_env.sh" >/dev/null || fail "gen_cc_env 失败"
fi
[ -f "$SCRIPT_DIR/.cc_env" ] || fail ".cc_env 缺失"

# ---------------- 5. 容器内安装 Claude Code（幂等） ----------------
if dex "command -v claude >/dev/null" >/dev/null 2>&1; then
    info "容器内 claude 已装：$(dex 'claude --version 2>/dev/null | head -1')"
else
    info "容器内安装 node + claude code（华为云镜像 + npmmirror）..."
    dex 'set -e; command -v xz >/dev/null || yum install -y xz >/dev/null 2>&1 || true
         V=v22.14.0; cd /tmp
         curl -fSL --max-time 300 "https://mirrors.huaweicloud.com/nodejs/$V/node-$V-linux-arm64.tar.xz" -o node.tar.xz
         tar -xJf node.tar.xz -C /usr/local; rm -f node.tar.xz
         NB=/usr/local/node-$V-linux-arm64/bin
         ln -sf $NB/node $NB/npm $NB/npx /usr/local/bin/
         npm config set registry https://registry.npmmirror.com
         npm install -g @anthropic-ai/claude-code >/dev/null 2>&1
         ln -sf $NB/claude /usr/local/bin/claude' \
        || fail "容器内 claude 安装失败（网络/镜像？手动按 operations.md 第 0 节排查）"
    info "容器内 claude：$(dex 'claude --version 2>/dev/null | head -1')"
fi

# ---------------- 6. 冒烟验证（容器内，与 orchestrator 同形态） ----------------
info "冒烟：容器内 source .cc_env + claude -p（GLM）"
SMOKE=$(dex "cd /tmp && . $CANONICAL/$(basename $LLM_REPO)/.agent/skills/llm-nightly-sync/scripts/.cc_env \
             && IS_SANDBOX=1 timeout 120 claude -p --dangerously-skip-permissions '只回复两个字:就绪' 2>&1 | tail -1")
echo "    agent 应答：$SMOKE"
echo "$SMOKE" | grep -q "就绪" || fail "冒烟未通过（key/网络/模型名？检查 .cc_env）"

# ---------------- 7. 接入 cron ----------------
if [ "$INSTALL_CRON" = "1" ]; then
    CRON_FILE=/etc/cron.d/llm-nightly-sync
    # 部署期配置经 docker exec -e 传导到运行期（orchestrator 同名变量接收），
    # 避免只改部署配置而 cron 仍回落默认值导致的静默分叉；CANN 不传导（有意跟随 pipeline）
    cat > "$CRON_FILE" <<EOF
# llm-nightly-sync 夜间巡检（deploy.sh 生成；每日 06:00 起轮询 PIPELINE 完成信号 tar.gz，完成即巡检）
$CRON_SCHEDULE root docker exec -e CONDA_ENV=$CONDA_ENV -e CONDA_HOME=$CONDA_HOME -e PIPELINE_LOG_BASE=$PIPELINE_LOG_BASE $DOCKER bash $CANONICAL/$(basename $LLM_REPO)/.agent/skills/llm-nightly-sync/scripts/llm_nightly_orchestrator.sh >> $CANONICAL/cron.log 2>&1
EOF
    chmod 644 "$CRON_FILE"
    info "已接入 cron：$CRON_FILE（$CRON_SCHEDULE）"
else
    info "未接 cron（INSTALL_CRON=1 重跑即可接入）。手动首跑："
    echo "    docker exec -e CONDA_ENV=$CONDA_ENV -e CONDA_HOME=$CONDA_HOME $DOCKER bash $CANONICAL/$(basename $LLM_REPO)/.agent/skills/llm-nightly-sync/scripts/llm_nightly_orchestrator.sh"
fi

# ---------------- 部署摘要 ----------------
cat <<EOF

================ 部署完成 ================
部署根(实际)   ：$DEPLOY_ROOT
容器内约定路径 ：$CANONICAL（$([ "$DEPLOY_ROOT" = "$CANONICAL" ] && echo 原生 || echo "软链 -> $DEPLOY_ROOT")）
容器 / conda   ：$DOCKER / $CONDA_ENV
skill          ：$CANONICAL/$(basename $LLM_REPO)/.agent/skills/llm-nightly-sync/
workspace      ：$CANONICAL/workspace（三上游仓）
PIPELINE 日志  ：$PIPELINE_LOG_BASE（容器内）
gitcode 凭据   ：$([ -f "$SCRIPT_DIR/.push_config.json" ] && echo 已配置 || echo 未配置（降级 NEEDS_HUMAN）)
cron           ：$([ "$INSTALL_CRON" = "1" ] && echo "$CRON_FILE @ $CRON_SCHEDULE" || echo 未接入)
容器重建后     ：重跑 bash scripts/deploy.sh 即完成修复（幂等）
==========================================
EOF
