#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm-nightly-sync 核心脚本：消费夜间 PIPELINE 产物，做快照/归因/报告。

职责（对应 SKILL.md 的 5 SNAPSHOT → 6 ATTRIBUTION → 10 REPORT）：
  1. 轮询等待 PIPELINE 完成信号（当日目录出现 *.tar.gz）
  2. 解析 exec_error.log 提取失败用例，逐个读日志尾部做错误模式分类
  3. git ls-remote 后置快照三滚动仓 commit（若 PIPELINE 记录了 commits.txt 则优先读取）
  4. 与 state/last_good.json 比对各仓 delta
  5. 确定性归因树（DATA/ENV/BASELINE_DRIFT/FLAKY/回归+嫌疑仓）
  6. 产出 env_report.json / attribution_report.json / final_report.json / diag_prompt.txt
  7. 全绿则前移 last_good

退出码：0=全绿；1=需 LLM 诊断；2=环境类 SKIP；3=PIPELINE 未完成/产物缺失
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE_DIR = os.path.join(SCRIPT_DIR, "..", "state")
GIT_BIN = shutil.which("git") or "git"  # 全路径，规避 bandit B607 部分可执行路径

# exec_error.log 中的失败行格式（tests/test_run.sh 写入）
RE_SCRIPT_FAIL = re.compile(r"^\[(PIPELINE_ST|PIPELINE_UT|ST|UT)\](\S+?) (?:\.sh )?Script has failed", re.M)
RE_COMPARE_FAIL = re.compile(r"^\[(PIPELINE_ST|ST)\](\S+?)\.sh compare to baseline has failed", re.M)
RE_PY_FAIL = re.compile(r"^\[(PIPELINE_UT|UT)\](\S+?) has failed", re.M)

# 确定性错误模式（按优先级，命中即归类，不再往下判）
ENV_PATTERNS = [
    r"hcclCommInitRootInfoConfig",
    r"EI0020",
    r"EE1001",
    r"Device is occupied",
    r"device\s+\d+\s+is\s+not\s+available",
    r"Address already in use",
]
DATA_PATTERNS = [
    r"FileNotFoundError",
    r"No such file or directory",
]
CRASH_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"RuntimeError",
    r"AssertionError",
    r"ChildFailedError",
    r"torch\.npu\.OutOfMemoryError",
]
PIPELINE_DATA_PREFIX = "/data/ci"  # pipeline 共享数据资源根（tests/resource_record.md 登记约束）


def sh(cmd, timeout=60):
    """执行命令（参数列表，不经 shell），返回 (rc, stdout+stderr)。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def load_json(path, default=None):
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return default


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def wait_for_pipeline(log_base, date, timeout_s, poll_s=300):
    """轮询等待完成信号：当日目录内 tar.gz 或父级 <date>.tar.gz（test_run.sh 打包两种
    落点均认；若产物只落在父级而仅检查 day_dir，会空轮直到超时）。
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if glob.glob(os.path.join(log_base, date, "*.tar.gz")) or os.path.exists(
            os.path.join(log_base, f"{date}.tar.gz")
        ):
            return True
        time.sleep(poll_s)
    return False


def snapshot_commits(matrix, day_dir):
    """后置快照。若 PIPELINE 记录了 commits.txt 则优先使用（精确）。"""
    recorded = {}
    commits_txt = os.path.join(day_dir, "commits.txt")
    if os.path.isfile(commits_txt):
        with open(commits_txt, encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    recorded[k.strip()] = v.strip()[:12]
    snap = {}
    for repo, meta in matrix["repos"].items():
        if repo in recorded:
            snap[repo] = recorded[repo]
            continue
        ref = meta["ref"]
        rc, out = sh([GIT_BIN, "ls-remote", meta["url"], f"refs/heads/{ref}"])
        sha = out.split()[0][:12] if rc == 0 and out.split() else "unknown"
        snap[repo] = sha
    return snap


def parse_failures(day_dir):
    """解析 exec_error.log + 各日志，返回失败用例列表及分类。"""
    err_log = os.path.join(day_dir, "exec_error.log")
    failures = {}
    if os.path.isfile(err_log):
        with open(err_log, encoding="utf-8", errors="replace") as f:
            content = f.read()
        # 用例名统一去 .sh 后缀，与日志文件名（[PIPELINE_ST]<name>.log）对齐
        for m in RE_COMPARE_FAIL.finditer(content):
            failures.setdefault(m.group(2), {"kinds": set()})["kinds"].add("baseline_compare")
        for m in RE_SCRIPT_FAIL.finditer(content):
            name = m.group(2)
            name = name[:-3] if name.endswith(".sh") else name  # 兼容 py3.8（无 removesuffix）
            failures.setdefault(name, {"kinds": set()})["kinds"].add("script_fail")
        for m in RE_PY_FAIL.finditer(content):
            failures.setdefault(m.group(2), {"kinds": set()})["kinds"].add("pytest_fail")

    for name, info in failures.items():
        log = find_case_log(day_dir, name)
        info["log"] = log
        info["category"], info["evidence"] = classify_log(log, info["kinds"])
        info["is_fsdp2"] = "fsdp2" in name.lower()
    return failures


def find_case_log(day_dir, name):
    for prefix in ("[PIPELINE_ST]", "[ST]", "[PIPELINE_UT]", "[UT]"):
        cand = os.path.join(day_dir, f"{prefix}{name}.log")
        if os.path.isfile(cand):
            return cand
    # UT 用例日志名是路径展平格式（MindSpeed-LLM_tests_pipeline_ut_xxx.log）
    hits = glob.glob(os.path.join(day_dir, f"*{name}*.log"))
    return hits[0] if hits else None


def tail(path, n=300):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def classify_log(log_path, kinds):
    """单用例确定性分类：ENV > DATA > BASELINE_DRIFT > CRASH(regression)。"""
    text = tail(log_path)
    for pat in ENV_PATTERNS:
        if re.search(pat, text, re.I):
            return "ENV", pat
    for pat in DATA_PATTERNS:
        m = re.search(rf"([^\s']*{re.escape(PIPELINE_DATA_PREFIX)}[^\s']*)", text)
        if m and re.search(pat, text):
            return "DATA", f"{pat}: {m.group(1)}"
    if kinds == {"baseline_compare"} and not any(re.search(p, text) for p in CRASH_PATTERNS):
        # 脚本退出码 0、无崩溃特征，仅指标越 margin → 精度/性能漂移
        return "BASELINE_DRIFT", "baseline_compare_only"
    for pat in CRASH_PATTERNS:
        if re.search(pat, text):
            return "REGRESSION", pat
    return "UNKNOWN", "no_pattern_matched"


def compute_deltas(matrix, snap, last_good):
    deltas = {}
    for repo, meta in matrix["repos"].items():
        if meta.get("track") != "remote":
            deltas[repo] = {"changed": False, "last_good": meta["ref"], "target": meta.get("ref")}
            continue
        lg = last_good.get(repo, {}).get("commit")
        tg = snap.get(repo, "unknown")
        deltas[repo] = {
            "changed": bool(lg) and lg != tg and tg != "unknown",
            "last_good": lg,
            "target": tg,
        }
    return deltas


def attribute(failures, deltas):
    """归因决策树（SKILL.md 步骤 6）。返回 (overall, needs_llm_diag, suspects)。"""
    cats = {info["category"] for info in failures.values()}
    if not failures:
        return "pass", False, {}
    if cats <= {"ENV"}:
        return "skip_env", False, {}
    reg = {n: i for n, i in failures.items() if i["category"] in ("REGRESSION", "UNKNOWN")}
    drift = {n: i for n, i in failures.items() if i["category"] == "BASELINE_DRIFT"}
    non_reg = {n: i for n, i in failures.items() if i["category"] in ("DATA",)}

    changed = [r for r, d in deltas.items() if d["changed"]]
    suspects = {}
    if reg:
        # 嫌疑仓先验：fsdp2 用例失败→fsdpturbo；大面积失败→mindspeed；单用例→llm 自身
        if len(reg) > 5:
            order = ["mindspeed", "mindspeed_llm", "fsdpturbo"]
        elif all(i["is_fsdp2"] for i in reg.values()):
            order = ["fsdpturbo", "mindspeed_llm", "mindspeed"]
        else:
            order = ["mindspeed_llm", "mindspeed", "fsdpturbo"]
        suspects = {r: deltas[r] for r in order if r in changed} or {"mindspeed_llm": deltas.get("mindspeed_llm")}
        if not changed:
            suspects = {"(zero-delta)": "环境抖动或数据问题，重跑一次定 FLAKY"}
        overall = "fail_needs_diag" if suspects else "fail_unknown"
        return overall, bool(reg), suspects
    if drift:
        # 零真回归、仅漂移 → 基线刷新流程（NEEDS_HUMAN），不烧 LLM
        return "fail_baseline_drift", False, {"baseline": "需用 acquire_json 重录 baseline"}
    if non_reg:
        return "fail_data", False, {"shared_data": "NEEDS_HUMAN: /data/ci 资源缺失"}
    return "fail_unknown", False, {}


def build_diag_prompt(day_dir, failures, deltas, suspects, report):
    lines = [
        "你是 MindSpeed-LLM 夜间巡检诊断 agent。昨天全绿，今天 PIPELINE 有失败。",
        f"日期：{report['date']}",
        "",
        "## 归因先验（确定性归因树产出）",
        json.dumps(suspects, ensure_ascii=False, indent=2),
        "",
        "## 各仓 delta（last_good → target）",
        json.dumps({r: f"{d['last_good']} → {d['target']}" for r, d in deltas.items()}, ensure_ascii=False, indent=2),
        "",
        "## 失败用例（按嫌疑仓获取 git log last_good..target 的 MR 列表与 diff，交叉比对）",
    ]
    for name, info in failures.items():
        if info["category"] not in ("REGRESSION", "UNKNOWN"):
            continue
        lines.append(f"### {name}  [{info['category']}] 证据: {info['evidence']}")
        lines.append(f"日志: {info['log']}")
        lines.append("```")
        lines.append(tail(info["log"], 40))
        lines.append("```")
    lines += [
        "",
        "## 任务",
        "1. 对每个嫌疑仓执行 git log/diff 扫描，关联「commit 变更符号 × 失败用例特征」",
        "1.1 缺包类专用（证据为 ModuleNotFoundError/ImportError）：commit delta 可能看不见——",
        "    依赖要求变更藏在安装期。对依赖仓做「自失败前一天 00:00 起」的宽窗口扫描：",
        "    git log --since=<失败前一天> -S <缺失包名或其依赖> -- pyproject.toml setup.py",
        "    （实例：fsdp_turbo 缺包真因是上游于失败前一天将 triton_ascend 提为 >=3.2.2",
        "    而 PyPI 最高仅 3.2.0，pip install 必败；仓 delta 为零，因该 commit 恰在 last_good 内。",
        "    另核实 pip 源可提供的版本区间），",
        "2. 输出 diag_report.json: breaking_mrs / root_cause / fix_repo / fix_summary（若命中 3.5 短路，另加 recurrence_short_circuit: true 与 reuse_pr: <PR号>，两者必須成对出现）",
        "3. 修复归属：归因仓优先（mindspeed/fsdpturbo 回归修上游仓；LLM 自身回归修本仓并连同看护用例提 PR；baseline 过期走重录流程）",
        "3.5 既有缺陷复发短路（先判再修，避免夜夜重复劳动）：以下三条同时满足 → 判定老问题复发，",
        "    跳过修复/重验/新分支/新 PR：",
        "    a) 当日失败签名与历史 evidence_*_ORIGINAL.log（或案例库）逐签名一致；",
        "    b) 归因结论不变（同一 breaking commit 或同一既有缺陷，当日 delta 无新增可疑改动）；",
        "    c) CI fork 仓存在覆盖该根因的 open PR（含修复 commit，未合并未关闭）。",
        "    短路动作：diag_report.json 写 \"recurrence_short_circuit\": true 与 \"reuse_pr\": <PR号>；",
        "    不重建分支不重验；create_issue_pr.py 会只在既有 Issue 追加评论",
        "    「<date> 复发，签名与历史一致，修复已在 PR !<N>，等待合入即可闭环」；",
        "    DIAGNOSIS_AND_FIX.md 照常写（记录复发事实并指向老 PR，不提新 PR）。",
        "    任一条不满足（新签名 / PR 已合并或关闭但用例仍失败=修复无效 / 无 open PR）→ 走完整 4-5 流程。",
        "4. （未触发 3.5 短路时）生成修复 → py_compile/AST 校验 → 在重验 workspace 只重验失败用例（bash tests/test_run.sh --pipeline-st -k <name>）",
        "5. （未触发 3.5 短路时）修复验证通过后按 operations.md 幂等建 Issue/PR，并写 DIAGNOSIS_AND_FIX.md（中文）",
    ]
    path = os.path.join(report["state_dir"], "diag_prompt.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def find_similar_case(state_dir, failure_names, suspects):
    """复发防护：查案例库匹配同 suspect/同证据模式的历史案例（SKILL 步骤 8 飞轮回流）。"""
    lib = os.path.join(state_dir, "nightly_case_library.jsonl")
    if not os.path.isfile(lib):
        return None
    suspect_keys = set(suspects.keys()) if isinstance(suspects, dict) else set()
    best = None
    with open(lib, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                continue
            overlap = suspect_keys & set(case.get("suspects", {}).keys())
            if overlap and (best is None or case.get("date", "") > best.get("date", "")):
                best = {
                    "date": case.get("date"),
                    "overall": case.get("overall"),
                    "suspects": sorted(overlap),
                    "failures": list(case.get("failures", {}).keys())[:3],
                }
    return best


def append_case_library(state_dir, report):
    """失败案例落库（全绿不落，delta 起点已由 last_good.json 承担）。"""
    if report["overall"] == "pass":
        return
    entry = {
        "date": report["date"],
        "overall": report["overall"],
        "failures": {n: {"category": i["category"], "evidence": i["evidence"]} for n, i in report["failures"].items()},
        "suspects": report["suspects"]
        if isinstance(report["suspects"], dict)
        else {str(k): str(v) for k, v in report["suspects"].items()},
        "deltas": {
            r: {"last_good": d.get("last_good"), "target": d.get("target")}
            for r, d in report["deltas"].items()
            if d.get("changed")
        },
    }
    with open(os.path.join(state_dir, "nightly_case_library.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--pipeline-log-base", default="/home/pipeline_log")
    ap.add_argument("--matrix", default=os.path.join(SCRIPT_DIR, "repo_matrix.json"))
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    ap.add_argument("--no-wait", action="store_true", help="不等待完成信号（手动重跑历史日期）")
    ap.add_argument("--wait-timeout", type=int, default=6 * 3600)
    args = ap.parse_args()

    matrix = load_json(args.matrix)
    if not matrix:
        print("[FATAL] repo_matrix.json 不可读", file=sys.stderr)
        return 3
    day_dir = os.path.join(args.pipeline_log_base, args.date)
    state_dir = os.path.abspath(args.state_dir)
    os.makedirs(state_dir, exist_ok=True)

    t0 = time.time()
    if not args.no_wait and not wait_for_pipeline(args.pipeline_log_base, args.date, args.wait_timeout):
        print(
            f"[FATAL] 等待 PIPELINE 完成信号超时（{args.wait_timeout}s），无 {args.pipeline_log_base}/{args.date}*.tar.gz",
            file=sys.stderr,
        )
        return 3
    if not os.path.isdir(day_dir):
        print(f"[FATAL] 目录不存在: {day_dir}", file=sys.stderr)
        return 3

    last_good_path = os.path.join(state_dir, "last_good.json")
    last_good = load_json(last_good_path, default={})

    # 5 SNAPSHOT
    snap = snapshot_commits(matrix, day_dir)
    dump_json(
        os.path.join(state_dir, "env_report.json"),
        {
            "date": args.date,
            "commits": snap,
            "snapshot_mode": "recorded" if os.path.isfile(os.path.join(day_dir, "commits.txt")) else "ls-remote",
        },
    )

    # 6 解析 + 归因
    failures = parse_failures(day_dir)
    deltas = compute_deltas(matrix, snap, last_good)
    overall, needs_llm, suspects = attribute(failures, deltas)

    report = {
        "date": args.date,
        "pipeline_log": day_dir,
        "commits": snap,
        "deltas": deltas,
        "failures": {
            n: {"category": i["category"], "kinds": sorted(i["kinds"]), "evidence": i["evidence"], "log": i["log"]}
            for n, i in failures.items()
        },
        "suspects": suspects,
        "overall": overall,
        "needs_llm_diag": needs_llm,
        "attribution_latency_s": round(time.time() - t0, 1),
        "state_dir": state_dir,
    }
    dump_json(os.path.join(state_dir, "attribution_report.json"), report)

    # 10 REPORT + 飞轮（全绿前移 last_good；失败落案例库 + 复发匹配）
    if overall == "pass":
        # ls-remote 失败的 "unknown" 占位不得前移，否则次日 delta 判定会拿 unknown 去 git log，归因链路断裂
        for repo, sha in snap.items():
            if sha != "unknown":
                last_good[repo] = {"commit": sha, "date": args.date}
        dump_json(last_good_path, last_good)
    else:
        report["similar_case"] = find_similar_case(state_dir, failures, suspects)
        append_case_library(state_dir, report)
    dump_json(os.path.join(state_dir, "final_report.json"), report)

    if needs_llm:
        prompt = build_diag_prompt(day_dir, failures, deltas, suspects, report)
        print(f"[DIAG] needs_llm_diag=true → {prompt}")

    if overall == "pass":
        code = 0
    elif overall == "skip_env":
        code = 2
    else:
        code = 1
    print(f"[DONE] overall={overall} failures={len(failures)} exit={code}")
    return code


if __name__ == "__main__":
    sys.exit(main())
