#!/bin/bash
# ============================================
# llm-nightly-sync Cron 编排入口（Step1/2/3）
# ============================================
# 用法：bash llm_nightly_orchestrator.sh [YYYY-MM-DD]
# crontab 示例（宿主机，每日 06:00 启动即轮询 tar.gz 完成信号，PIPELINE 完成即巡检；
# 轮询兼兜底 PIPELINE 晚归，超时 6h。<容器名>/<conda配置> 由 deploy.sh ⑩生成 cron 行时
# 自动写入，默认 llm_ci_q4 + CONDA_ENV/CONDA_HOME -e 传导；也可手动替换）：
#   0 6 * * * docker exec -e CONDA_ENV=<conda环境名> -e CONDA_HOME=<conda根> <容器名> bash /home/llm_nightly/MindSpeed-LLM/.agent/skills/llm-nightly-sync/scripts/llm_nightly_orchestrator.sh >> /home/llm_nightly/cron.log 2>&1
#   （部署根默认 /home/llm_nightly；实际挂载点不同时加 REVERIFY_WS=<路径> 覆盖）
#
# Step1  nightly_postprocess.py  确定性后处理（等完成信号/快照/归因/报告）
# Step2  LLM 诊断修复            仅 exit=1 时触发（agent 读 state/diag_prompt.txt）
# Step3  报告推送 + 飞轮

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/../state}"
DATE="${1:-$(date '+%Y-%m-%d')}"
LOG_BASE="${PIPELINE_LOG_BASE:-/home/pipeline_log}"
AGENT_BIN="${AGENT_BIN:-claude}"   # agent 命令（qoder/claude 等），按部署环境配置
# 重验 workspace（fetch_mrs delta 扫描 + 9 PATCH 重验的四仓 clone）
# 必须放在部署根（容器可见的挂载区）内
REVERIFY_WS="${REVERIFY_WS:-/home/llm_nightly/workspace}"

# ---------- flock 防重 ----------
# cron 超时未归 + 次日触发并发会路坏 state；已有实例在跑则本次直接退出
LOCK=/tmp/llm-nightly-unattended.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[$(date '+%F %T')] 已有实例在运行，本次触发跳过"; exit 0; }

# NPU 空闲等待参数（仅 Step2 重验烧卡前生效；Step1 消费日志不需要卡）
AICORE_THRESHOLD=${AICORE_THRESHOLD:-10}   # 单卡 AICore% ≥ 此值视为忙
WAIT_INTERVAL=${WAIT_INTERVAL:-300}        # 轮询间隔（秒）
MAX_WAIT_MINUTES=${MAX_WAIT_MINUTES:-240}  # 最长等待，超时 NEEDS_HUMAN 保留现场
DRY_RUN=${DRY_RUN:-0}

# agent 运行环境：cron 为非交互 shell，不加载 ~/.bashrc（交互守卫提前 return，
# setup-cc-ds.sh 写入的 ANTHROPIC_* 不生效），且 cron 默认 PATH 不含 /usr/local/bin
# （claude 软链所在）。必须从专用 env 文件显式加载；生成方式：bash gen_cc_env.sh
CC_ENV="${CC_ENV:-$SCRIPT_DIR/.cc_env}"
if [ -f "$CC_ENV" ]; then
    # shellcheck disable=SC1090
    . "$CC_ENV"
else
    echo "[WARN] 缺少 $CC_ENV：Step2 的 agent 将无 token。生成：bash $SCRIPT_DIR/gen_cc_env.sh"
fi

# 巡检执行环境：执行容器内 CONDA_ENV conda env + CANN（默认 llm_test，随 deploy 配置）。
# cron 非交互不读 .bashrc（conda init 不生效，默认 python 为系统旧版）；重验跑
# test_run.sh 需 torch/torch_npu（在 CONDA_ENV）+ CANN 环境变量，故全部显式加载。
# ⚠️ set -u 与 conda/CANN 官方脚本不兼容（其内部引用 LD_LIBRARY_PATH/PYTHONPATH/
# ZSH_VERSION 等未定义变量，cron 环境尤甚），source 期间临时关闭
# （set_env.sh 内部引用未定义变量，不关闭会被 set -u 直接中断）。
set +u
# 变量名与 deploy.sh 统一为 CONDA_ENV（部署期配置经 cron 行 docker exec -e 传导至此）；
# CONDA_ENV_NAME 为旧名兼容别名，优先级：CONDA_ENV > CONDA_ENV_NAME > 默认 llm_test。
# CANN 有意不配置化：跟随 pipeline（硬编码 /usr/local/Ascend/ascend-toolkit，即系统
# latest 链接当前指向的版本），保证重验与当夜 PIPELINE 同一运行时，避免对照变量污染。
CONDA_ENV="${CONDA_ENV:-${CONDA_ENV_NAME:-llm_test}}"
CONDA_HOME="${CONDA_HOME:-/root/miniconda3}"
if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
    . "$CONDA_HOME/etc/profile.d/conda.sh"
    if conda activate "$CONDA_ENV" 2>/dev/null; then
        echo "[INFO] conda env: $CONDA_ENV ($(python3 -V 2>&1))"
    else
        echo "[WARN] conda activate $CONDA_ENV 失败，Step1 退用系统 python3（重验将失败，需人工检查 env）"
    fi
else
    echo "[WARN] 未找到 conda（$CONDA_HOME），Step1 退用系统 python3"
fi
for env_sh in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh; do
    [ -f "$env_sh" ] && . "$env_sh"   # CANN/NNAL，torch_npu 与 NPU 访问必需
done
# npu-smi 依赖 driver 多个子目录库（容器内缺路径→空输出→NPU 误判忙死等待超时）
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
set -u   # 外部环境脚本加载完毕，恢复 nounset

mkdir -p "$STATE_DIR"
echo "[$(date '+%F %T')] llm-nightly-sync start date=$DATE"

# ---------------- Step 0: 重验训练产物清理（mtime > 5 天的 output/） ----------------
# 重验 15-step 用例会在各仓根（含嵌套 clone：workspace/<repo>/<repo>/）落
# output/checkpoints/*.distcp 与 HF 导出 *.safetensors，单轮可达数十 GB
# （且因目录嵌套常为双份存放）。只清训练产物，不碰
# 源码 / state / evidence；DRY_RUN=1 时只报不删。
# 深度限制在 find 层实现：-mindepth 2 -maxdepth 3 恰好覆盖 ws/<repo>/output
# 与嵌套 ws/<repo>/<repo>/output（bash case 的 * 会匹配 /，无法限深，勿用）。
# 口径：以 output/ 目录自身 mtime 判 5 天（向既有子目录写新文件不刷新父目录
# mtime，极端情况会删到含更新文件的目录；验证结论不依赖 checkpoint，可接受）。
cleanup_stale_outputs() {
    [ -d "$REVERIFY_WS" ] || return 0
    find "$REVERIFY_WS" -mindepth 2 -maxdepth 3 -type d -name output -mtime +5 \
        -not -path "*/.git/*" 2>/dev/null | while read -r d; do
        size=$(du -sm "$d" 2>/dev/null | cut -f1)
        if [ "$DRY_RUN" = "1" ]; then
            echo "[CLEAN][DRY] would remove $d (${size}MB, mtime>5d)"
        else
            rm -rf --one-file-system "$d"
            echo "[CLEAN] removed $d (${size}MB freed, mtime>5d)"
        fi
    done
}
cleanup_stale_outputs

# ---------------- Step 1: 确定性后处理 ----------------
python3 "$SCRIPT_DIR/nightly_postprocess.py" --date "$DATE" --pipeline-log-base "$LOG_BASE" --state-dir "$STATE_DIR"
STEP1=$?

if [ "$STEP1" -eq 0 ]; then
    echo "[$(date '+%F %T')] all green, last_good advanced."
elif [ "$STEP1" -eq 2 ]; then
    echo "[$(date '+%F %T')] env skip (NPU occupied / env issue). No diagnosis, no Issue/PR."
elif [ "$STEP1" -eq 3 ]; then
    echo "[$(date '+%F %T')] FATAL: pipeline not finished or artifacts missing. Alert owner."
    exit 3
fi

# ---------------- Step 2: LLM 诊断修复（仅 needs_llm_diag） ----------------
if [ "$STEP1" -eq 1 ] && [ -f "$STATE_DIR/diag_prompt.txt" ]; then
    # 等待 NPU 空闲：Step2 内含重验烧卡，NPU 被占时重验只会产出假失败。
    # npu-smi 异常视为忙，保守不烧卡。
    npu_free() {
        local cores
        cores=$(npu-smi info 2>/dev/null | awk -F'|' '{
            chip=$2; bus=$3
            gsub(/^ +| +$/, "", chip); gsub(/^ +| +$/, "", bus)
            if (chip ~ /^[0-9]+$/ && bus ~ /^0000:/ && match($4, /[0-9]+/)) {
                print substr($4, RSTART, RLENGTH)
            }
        }')
        [ -z "$cores" ] && return 1
        for c in $cores; do
            [ "$c" -ge "$AICORE_THRESHOLD" ] && return 1
        done
        return 0
    }
    if [ "$DRY_RUN" != "1" ]; then
        start_epoch=$(date +%s)
        echo "[$(date '+%F %T')] 等待 NPU 空闲（阈值 ${AICORE_THRESHOLD}%，${WAIT_INTERVAL}s 轮询）..."
        until npu_free; do
            elapsed_min=$(( ($(date +%s) - start_epoch) / 60 ))
            if [ "$elapsed_min" -ge "$MAX_WAIT_MINUTES" ]; then
                echo "[NEEDS_HUMAN] NPU 忙，等待 ${elapsed_min}min 超时。diag_prompt.txt 保留在 $STATE_DIR，空闲后手动重跑"
                exit 1
            fi
            sleep "$WAIT_INTERVAL"
        done
        echo "[$(date '+%F %T')] NPU 空闲"
    fi
    TOKEN_STATE="缺失"
    [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] && TOKEN_STATE="已设置"
    if command -v "$AGENT_BIN" >/dev/null 2>&1 && [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
        echo "[$(date '+%F %T')] launching LLM diagnosis agent... (token=$TOKEN_STATE)"
        # root/容器环境下 claude -p --dangerously-skip-permissions 需 IS_SANDBOX=1 放行
        # （Claude Code 安全策略拒绝 root+skip-permissions；容器/CI 场景官方允许此变量放行）
        export IS_SANDBOX="${IS_SANDBOX:-1}"
        # 诊断前先做 delta 扫描压缩搜索空间（步骤 7，产物 diff_report.json 供 agent 读）
        python3 "$SCRIPT_DIR/fetch_mrs.py" --state-dir "$STATE_DIR" --workspace "$REVERIFY_WS" \
            || echo "[WARN] fetch_mrs.py 失败，agent 将无 diff_report.json 先验，继续"
        # agent 需可读：SKILL.md / references/operations.md / state/*.json / $LOG_BASE/<date>/
        if "$AGENT_BIN" -p --dangerously-skip-permissions "$(cat "$STATE_DIR/diag_prompt.txt")"; then
            rm -f "$STATE_DIR/diag_prompt.txt"   # 成功才消费，失败保留供手动重试
        else
            echo "[WARN] agent exited non-zero; diag_prompt.txt kept in $STATE_DIR for manual retry"
        fi
    else
        echo "[NEEDS_HUMAN] agent 未就绪（binary=$AGENT_BIN, token=$TOKEN_STATE）。手动执行："
        echo "  . $CC_ENV && $AGENT_BIN -p \"\$(cat $STATE_DIR/diag_prompt.txt)\""
    fi
fi

# ---------------- Step 2.5: 幂等 Issue/PR（agent 修复后） ----------------
# 前置：state/diag_report.json（agent 诊断产出，含 fix_repo/root_cause）+ state/fixed_tests.json
# 当日性守卫：diag_report.json 必须是今天写的（-newermt $DATE 00:00），防止全绿/环境跳过日
# 拿陈旧报告重复追加 Issue 评论（diag_report 不自动清理，历史文件会一直存在）
if [ -f "$SCRIPT_DIR/.push_config.json" ] && [ -f "$STATE_DIR/diag_report.json" ] \
    && [ -f "$SCRIPT_DIR/create_issue_pr.py" ] \
    && [ -n "$(find "$STATE_DIR/diag_report.json" -newermt "$DATE 00:00" 2>/dev/null)" ]; then
    python3 "$SCRIPT_DIR/create_issue_pr.py" --date "$DATE" --state-dir "$STATE_DIR" --workspace "$REVERIFY_WS" --dry-run \
        && python3 "$SCRIPT_DIR/create_issue_pr.py" --date "$DATE" --state-dir "$STATE_DIR" --workspace "$REVERIFY_WS" \
        || echo "[WARN] create_issue_pr 未完成（dry-run 预检或正式执行失败），可手动重跑"
fi

# ---------------- Step 2.6: 巡检结论追加到当日 exec_error.log ----------------
# exec_error.log 是人工看 PIPELINE 失败的第一入口；把巡检归因/修复 PR 追加进去，
# 早晨只看这一个文件即可闭环。追加不覆盖，保留 PIPELINE 原始记录。
EXEC_LOG="$LOG_BASE/$DATE/exec_error.log"
if [ -f "$EXEC_LOG" ]; then
    python3 - "$STATE_DIR/final_report.json" "$STATE_DIR" "$DATE" <<'PYEOF' > /tmp/nightly_exec_summary.txt
import glob, json, os, sys
report_path, state_dir, date = sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.load(open(report_path, encoding="utf-8"))
overall = rep.get("overall", "unknown")
print(f"==================== nightly-sync 巡检结论 ({date}) ====================")
if overall == "pass":
    print("[巡检] 今日全绿，last_good 前移。")
    sys.exit(0)
if overall == "skip_env":
    print("[巡检] 环境类跳过（NPU 占用/环境故障），未做诊断，无 Issue/PR。")
    sys.exit(0)
# 今日最新一条非 dry-run 的 Issue/PR 记录（可能分布在多个 issue_registry_<repo>.jsonl）
reg = None
for p in glob.glob(os.path.join(state_dir, "issue_registry_*.jsonl")):
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
            if r.get("date") == date and not r.get("dry_run"):
                reg = r
        except Exception:
            pass
for name, info in rep.get("failures", {}).items():
    print(f"  - {name} [{info.get('category')}]")
print(f"[巡检] overall={overall}，失败 {len(rep.get('failures', {}))} 项（明细见上方日志名）")
if reg:
    issue, pr = reg.get("issue"), reg.get("pr")
    if reg.get("short_circuit"):
        print(f"[巡检] 归因：既有缺陷复发（签名与历史一致），走短路——复用 Issue #{issue} / PR !{reg.get('reuse_pr')}，未重建分支/未重验，等待上游合入")
    elif issue:
        print(f"[巡检] 归因+修复完成：Issue #{issue} / PR !{pr}（CI fork，等待合入上游）")
else:
    print("[巡检] 今日无 Issue/PR 记录（诊断未闭环或 NEEDS_HUMAN），见 state/DIAGNOSIS_AND_FIX.md")
print(f"[巡检] 详情：state/DIAGNOSIS_AND_FIX.md、SYNC_SUMMARY_{date}.md")
print("=" * 64)
PYEOF
    if [ -s /tmp/nightly_exec_summary.txt ]; then
        cat /tmp/nightly_exec_summary.txt >> "$EXEC_LOG"
        echo "[INFO] 巡检结论已追加 $EXEC_LOG"
    fi
fi

# ---------------- Step 3: 报告推送（凭据就绪时） ----------------
if [ -f "$SCRIPT_DIR/.push_config.json" ] && [ -f "$SCRIPT_DIR/push_sync_ci_results.py" ]; then
    python3 "$SCRIPT_DIR/push_sync_ci_results.py" --date "$DATE" --state-dir "$STATE_DIR" \
        || echo "[WARN] push failed, report kept locally"
else
    echo "[INFO] .push_config.json / push_sync_ci_results.py 未配置，报告保留本地 $STATE_DIR"
fi

echo "[$(date '+%F %T')] llm-nightly-sync done (step1=$STEP1)"
exit "$STEP1"
