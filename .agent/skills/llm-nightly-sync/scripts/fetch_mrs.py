#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7 FETCH_MRS — 按仓 delta 扫描 + LLM 调用点交叉比对。

对 attribution_report.json 中每个有 delta 的仓：git log 取 commit 列表、git diff 提取
变更符号（def/class 名 + 涉及文件）；再扫描 mindspeed_llm 源码（AST）中对上游仓模块的
import 调用点，与变更符号交叉比对——命中即「嫌疑 commit × 被波及的 LLM 调用点」，
直接压缩 8 DIAG 的搜索空间。

无 vendor drift 检查（LLM 不 vendor 上游文件）；交叉比对基准为 LLM 自身的上游 import 面。

输入：state/attribution_report.json（deltas：last_good/target）+ repo_matrix.json + workspace 各仓 clone
输出：state/diff_report.json
退出码：0 正常产出（含 0 delta）；1 输入缺失
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# LLM 源码中要扫描 import 调用点的目录（相对 mindspeed_llm clone 根）
LLM_SCAN_DIRS = ["mindspeed_llm"]
# 上游仓 → 其 import 前缀（用于判定 LLM 是否调用该仓 API）
UPSTREAM_IMPORT_PREFIXES = {
    "mindspeed": ("mindspeed",),
    "fsdpturbo": ("fsdpturbo",),
    "megatron_lm": ("megatron",),
}
# workspace 下各仓 clone 的目录名（与 operations.md 第 0 节组装一致）
WS_DIRNAME = {
    "mindspeed_llm": "MindSpeed-LLM",
    "mindspeed": "MindSpeed",
    "fsdpturbo": "FSDPTurbo",
    "megatron_lm": "Megatron-LM",
}

RE_DEF = re.compile(r"^\+\s*(?:async\s+)?def\s+(\w+)|^\+\s*class\s+(\w+)", re.M)


def sh(cmd, cwd=None, timeout=60):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
        return p.returncode, p.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[WARN] {' '.join(cmd[:3])}... failed: {e}", file=sys.stderr)
        return 1, ""


def load_json(path, default=None):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return default


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_changed_symbols(diff_text):
    """从 unified diff 的新增行提取 def/class 名。"""
    return sorted({m.group(1) or m.group(2) for m in RE_DEF.finditer(diff_text) if (m.group(1) or m.group(2))})


def scan_repo_delta(repo_path, last_good, target):
    """单仓 delta：commit 列表 + 涉及文件 + 变更符号。"""
    # workspace clone 仅部署时创建，target（当日分支头）通常不在本地；先 fetch 再扫，
    # 失败仅告警降级（本地已有对象时仍可扫通），保持既有 error 路径
    rc, _ = sh(["git", "fetch", "--quiet", "origin"], cwd=repo_path, timeout=120)
    if rc != 0:
        print(f"[WARN] {os.path.basename(repo_path)}: git fetch origin 失败，delta 扫描基于本地旧对象", file=sys.stderr)
    rc, log_out = sh(["git", "log", "--format=%H %s", f"{last_good}..{target}"], cwd=repo_path)
    if rc != 0:
        return {"error": f"git log failed: {last_good}..{target}"}
    commits = []
    for line in log_out.strip().splitlines():
        if not line:
            continue
        sha, _, subject = line.partition(" ")
        commits.append({"sha": sha[:12], "subject": subject})
    rc, diff_out = sh(["git", "diff", "--name-only", last_good, target], cwd=repo_path)
    files = [ln for ln in diff_out.strip().splitlines() if ln]
    rc, diff_full = sh(["git", "diff", last_good, target], cwd=repo_path)
    symbols = extract_changed_symbols(diff_full)
    return {"commit_count": len(commits), "commits": commits, "files": files, "changed_symbols": symbols}


def collect_llm_upstream_calls(llm_repo_path):
    """AST 扫描 LLM 源码，收集对上游仓模块的 import 调用点。

    返回 {repo_key: {symbol: [file:line, ...]}}，symbol 取 import 的短名
    （from mindspeed.x import foo → foo；import mindspeed.x.y → y）。
    """
    calls = {}
    for scan_dir in LLM_SCAN_DIRS:
        base = os.path.join(llm_repo_path, scan_dir)
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        tree = ast.parse(f.read())
                except SyntaxError:
                    continue
                rel = os.path.relpath(path, llm_repo_path)
                for node in ast.walk(tree):
                    imported = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        imported = [(a.asname or a.name, node.module) for a in node.names]
                    elif isinstance(node, ast.Import):
                        imported = [((a.asname or a.name).split(".")[-1], a.asname or a.name) for a in node.names]
                    for short, full in imported:
                        for repo_key, prefixes in UPSTREAM_IMPORT_PREFIXES.items():
                            top = full.split(".")[0]
                            if top in prefixes:
                                calls.setdefault(repo_key, {}).setdefault(short, []).append(f"{rel}:{node.lineno}")
    return calls


def cross_reference(repo_key, delta, llm_calls):
    """上游变更符号 × LLM 调用点交叉比对。返回 hits 列表。"""
    calls = llm_calls.get(repo_key, {})
    if not calls or "changed_symbols" not in delta:
        return []
    hits = []
    # 每 commit 关联：diff 归属 commit 粒度较粗，这里按符号级命中 + 全 commit 候选
    for sym in delta.get("changed_symbols", []):
        if sym in calls:
            hits.append(
                {
                    "symbol": sym,
                    "llm_callers": calls[sym][:5],
                    "candidate_commits": [c["sha"] for c in delta.get("commits", [])][:10],
                }
            )
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.path.join(SCRIPT_DIR, "..", "state"))
    ap.add_argument("--workspace", default="/home/llm_nightly/workspace", help="重验 workspace 根（各仓 clone 所在）")
    args = ap.parse_args()

    attr = load_json(os.path.join(args.state_dir, "attribution_report.json"))
    if not attr or "deltas" not in attr:
        print("[FATAL] attribution_report.json 缺失或无 deltas（先跑 nightly_postprocess.py）", file=sys.stderr)
        return 1

    llm_repo = os.path.join(args.workspace, WS_DIRNAME["mindspeed_llm"])
    llm_calls = collect_llm_upstream_calls(llm_repo) if os.path.isdir(llm_repo) else {}

    report = {"date": attr.get("date"), "repos": {}, "scan_note": None}
    for repo_key, d in attr["deltas"].items():
        if not d.get("changed"):
            continue
        repo_path = os.path.join(args.workspace, WS_DIRNAME.get(repo_key, repo_key))
        if not os.path.isdir(repo_path):
            report["repos"][repo_key] = {"error": f"clone 不存在: {repo_path}（按 operations.md 组装 workspace）"}
            continue
        delta = scan_repo_delta(repo_path, d["last_good"], d["target"])
        delta["hits"] = cross_reference(repo_key, delta, llm_calls)
        delta["severity"] = "high" if delta.get("hits") else ("medium" if delta.get("commit_count") else "none")
        report["repos"][repo_key] = delta

    if not llm_calls:
        report["scan_note"] = f"LLM clone 未找到（{llm_repo}），调用点交叉比对跳过"

    out = os.path.join(args.state_dir, "diff_report.json")
    dump_json(out, report)
    total = sum(r.get("commit_count", 0) for r in report["repos"].values())
    hits = sum(len(r.get("hits", [])) for r in report["repos"].values())
    print(f"[DONE] repos_with_delta={len(report['repos'])} commits={total} symbol_hits={hits} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
