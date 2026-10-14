#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 11 CI_REPORT — 合成 Markdown 汇总并推送巡检报告仓。

读取 state/*.json（final/attribution/diff/diag）合成 SYNC_SUMMARY_<date>.md：
  各仓 commit 速查 + delta + 阶段状态 + 失败/归因/修复 + 案例库复发提示；
归档 DIAGNOSIS_AND_FIX.md（如 agent 已写）与 nightly_case_library.jsonl 备份。

推送：.push_config.json 的 ci_repo.enabled=true 时 clone/更新报告仓并推送
sync_build_results/YYYY-MM/MM-DD/；默认 false 仅本地生成。
退出码：0 报告已生成（本地或已推送）；1 输入缺失
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_BIN = shutil.which("git") or "git"  # 全路径，规避 bandit B607 部分可执行路径


def _mask(msg):
    """脱敏 URL 内嵌凭据（oauth2:<token>@），防 token 进 cron 日志"""
    return re.sub(r"https://[^@/]+@", "https://***@", msg)


def load_json(path, default=None):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return default


def build_summary(state_dir, date):
    final = load_json(os.path.join(state_dir, "final_report.json"), {})
    diff = load_json(os.path.join(state_dir, "diff_report.json"), {})
    diag = load_json(os.path.join(state_dir, "diag_report.json"), {})

    lines = [
        f"# llm-nightly-sync 巡检报告 {date}",
        "",
        f"- overall：**{final.get('overall', 'N/A')}**　"
        f"needs_llm_diag：{final.get('needs_llm_diag')}　"
        f"归因耗时：{final.get('attribution_latency_s', 'N/A')}s",
        "",
    ]

    lines += ["## 各仓 commit 快照与 delta", "", "| 仓 | target | delta |", "|---|---|---|"]
    for repo, d in (final.get("deltas") or {}).items():
        arrow = f'{d.get("last_good", "?")[:8]} → {str(d.get("target", "?"))[:8]}' if d.get("changed") else "无"
        lines.append(f"| {repo} | {str(d.get('target', '?'))[:12]} | {arrow} |")

    failures = final.get("failures") or {}
    if failures:
        lines += ["", "## 失败用例", "", "| 用例 | 分类 | 证据 |", "|---|---|---|"]
        for name, info in failures.items():
            lines.append(f"| {name} | {info.get('category')} | {str(info.get('evidence'))[:60]} |")
        sus = final.get("suspects")
        if sus:
            lines += ["", f"归因先验：`{json.dumps(sus, ensure_ascii=False)[:300]}`"]
        sim = final.get("similar_case")
        if sim:
            lines += ["", f"⚠️ 复发提示：与 {sim.get('date')} 案例疑似同源（suspects={sim.get('suspects')}）"]

    repos_delta = diff.get("repos") or {}
    if repos_delta:
        lines += ["", "## delta 扫描（步骤 7）", ""]
        for repo, d in repos_delta.items():
            if "error" in d:
                lines.append(f"- {repo}: ERROR {d['error']}")
                continue
            hits = d.get("hits") or []
            hit_str = ", ".join(f"`{h['symbol']}`({'/'.join(h['llm_callers'][:2])})" for h in hits[:5]) or "无符号命中"
            lines.append(
                f"- **{repo}**：{d.get('commit_count', 0)} commits，severity={d.get('severity')}，命中：{hit_str}"
            )

    if diag:
        lines += [
            "",
            "## 诊断结论（步骤 8）",
            "",
            f"- fix_repo：**{diag.get('fix_repo')}**",
            f"- root_cause：{diag.get('root_cause', '')}",
            f"- fix_summary：{diag.get('fix_summary', '')}",
        ]

    lines += ["", f"*生成时间：{datetime.now().strftime('%F %T')}，数据源：state/*.json（可复现）*", ""]
    return "\n".join(lines)


def push_to_ci_repo(cfg, state_dir, date, summary_path):
    ci = cfg.get("ci_repo", {})
    if not ci.get("enabled"):
        print(f"[INFO] ci_repo.enabled=false，报告仅本地：{summary_path}")
        return False
    repo_dir = ci.get("repo_dir", "/home/llm_nightly/ci_report_clone")
    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        rc = subprocess.run([GIT_BIN, "clone", ci["url"], repo_dir], capture_output=True, text=True, check=False)
        if rc.returncode != 0:
            print(f"[WARN] clone 报告仓失败: {_mask(rc.stderr.strip())[:150]}，报告保留本地")
            return False
    else:
        subprocess.run([GIT_BIN, "-C", repo_dir, "pull", "--rebase"], capture_output=True, check=False)

    dest = os.path.join(repo_dir, "sync_build_results", date[:7], date[8:])
    os.makedirs(dest, exist_ok=True)
    shutil.copy(summary_path, os.path.join(dest, os.path.basename(summary_path)))
    for extra, name in (
        (os.path.join(state_dir, "DIAGNOSIS_AND_FIX.md"), "DIAGNOSIS_AND_FIX.md"),
        (os.path.join(state_dir, "nightly_case_library.jsonl"), "nightly_case_library.jsonl"),
    ):
        if os.path.isfile(extra):
            shutil.copy(extra, os.path.join(dest, name))
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME=ci.get("author_name", "llm-nightly"),
        GIT_AUTHOR_EMAIL=ci.get("author_email", "llm-nightly@localhost"),
        GIT_COMMITTER_NAME=ci.get("author_name", "llm-nightly"),
        GIT_COMMITTER_EMAIL=ci.get("author_email", "llm-nightly@localhost"),
    )
    for cmd in ([GIT_BIN, "add", "-A"], [GIT_BIN, "commit", "-m", f"nightly report {date}"]):
        subprocess.run(cmd, cwd=repo_dir, capture_output=True, env=env, check=False)
    url = ci["url"].replace("https://", f"https://oauth2:{cfg['gitcode_token']}@")
    rc = subprocess.run([GIT_BIN, "-C", repo_dir, "push", url, "HEAD"], capture_output=True, text=True, check=False)
    if rc.returncode != 0:
        print(f"[WARN] push 报告仓失败: {_mask(rc.stderr.strip())[:150]}")
        return False
    print(f"[OK] 已推送报告仓 sync_build_results/{date[:7]}/{date[8:]}/")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--state-dir", default=os.path.join(SCRIPT_DIR, "..", "state"))
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(args.state_dir, "final_report.json")):
        print("[FATAL] state/final_report.json 缺失（先跑 nightly_postprocess.py）", file=sys.stderr)
        return 1

    summary = build_summary(args.state_dir, args.date)
    summary_path = os.path.join(args.state_dir, f"SYNC_SUMMARY_{args.date}.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[OK] summary → {summary_path}")

    cfg = load_json(os.path.join(SCRIPT_DIR, ".push_config.json"), {})
    if cfg:
        push_to_ci_repo(cfg, args.state_dir, args.date, summary_path)
    else:
        print("[INFO] .push_config.json 未配置，报告仅本地")
    return 0


if __name__ == "__main__":
    sys.exit(main())
