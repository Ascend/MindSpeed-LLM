#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2.5 幂等建 Issue/PR。

读取 agent 诊断产物 state/diag_report.json（fix_repo/root_cause/breaking_mrs）与
state/fixed_tests.json（确认修复的真问题用例），按 fix_repo 路由到对应仓，两种模式：

A) CI fork 闭环（repos.<key>.ci_fork_url 非空，推荐）：
  1. 同步：workspace clone fetch 上游 base_branch → push 到 ci fork 同名分支
     （fork 落后则 --force 使其与上游对齐）
  2. 分支：ci_repair_<YYMMDD>（基于同步后的 base，agent 9 PATCH 已在其上 commit）
  3. push：分支推到 ci fork（https://username:token@gitcode.com/...）
  4. Issue/PR：均提 ci fork（owner/repo 从 ci_fork_url 解析；同仓 PR head=分支名）

B) fork→上游主仓（ci_fork_url 为空，旧模式）：分支 daily_fix_<date> 推 fork_user 名下
   fork，从 fork 提 PR 到上游主仓 base_branch。

Issue：粒度=（仓, 问题级）。标题 `[daily-ci] <root_cause 摘要>`，open+label 查重，
       已存在 → 追加每日评论；不存在 → 创建。
PR：  open PR 按 head 查重复用（force-push 自动更新 diff）。

⚠️ CLA 身份：commit 的 author/committer 必须用 .push_config.json 中
repos.<fix_repo>.author_name/author_email（agent 在 9 PATCH 阶段已按此约束提交），
否则 PR 的 CLA 检查失败。

--dry-run：校验输入与配置、打印将执行的动作，不调 API 不 push（orchestrator 先预检再正式跑）。
退出码：0 成功或幂等跳过；1 输入/配置缺失；2 API/git 失败。
       主路径与短路分支对称：issue/PR 创建失败或 push 真实失败均 return 2，
       仅「本地无 clone/无修复分支」这类有意跳过才 return 0，避免假闭环。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import urllib.parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_BIN = shutil.which("git") or "git"  # 全路径，规避 bandit B607 部分可执行路径
LABEL = "daily-ci"


def load_json(path, default=None):
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return default


def api(cfg, method, path, body=None):
    """gitcode API 调用，返回 (status, json)。⚠️ gitcode v5 仅接受 form 编码 body（json 报 400 parsing error）。"""
    url = f"{cfg['gitcode_api_base']}{path}"
    data, headers = None, {"Authorization": f"Bearer {cfg['gitcode_token']}"}
    if body is not None:
        flat = {k: (",".join(v) if isinstance(v, list) else v) for k, v in body.items()}
        data = urllib.parse.urlencode(flat).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return 0, {"error": str(e)}


def fork_url_parts(url):
    """https://gitcode.com/<owner>/<repo>.git → (owner, repo)。"""
    m = re.search(r"gitcode\.com/([^/]+)/([^/]+?)(?:[.]git)?$", url or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def cred_url(cfg, url):
    """凭据注入：https://<username>:<token>@gitcode.com/...（username:token 方式可 push）。"""
    user = cfg.get("gitcode_username") or cfg.get("fork_user") or "oauth2"
    return "https://" + user + ":" + cfg["gitcode_token"] + "@" + url.split("https://", 1)[-1]


def sync_upstream_to_fork(cfg, repo_key, ws_root, dry):
    """CI fork 模式前置：workspace clone fetch 上游 base → push 到 ci fork（fork 落后则 force 对齐）。"""
    repo_cfg = cfg["repos"][repo_key]
    local_dir = os.path.join(ws_root, repo_cfg.get("ws_dirname", repo_key))
    base = repo_cfg.get("base_branch", "master")
    if not os.path.isdir(os.path.join(local_dir, ".git")):
        print(f"[SKIP] 本地 clone 不存在: {local_dir}，无法同步（NEEDS_HUMAN）")
        return False
    if dry:
        print(f"[DRY] sync upstream {base} → {repo_cfg['ci_fork_url']}（fetch origin + push，落后则 --force 对齐）")
        return True
    fork_push = cred_url(cfg, repo_cfg["ci_fork_url"])

    def g(*a):
        return subprocess.run([GIT_BIN, "-C", local_dir] + list(a), capture_output=True, text=True, check=False)

    if g("fetch", "origin", base).returncode != 0:
        print(f"[WARN] fetch origin {base} 失败，fork 保持现状")
        return False
    r1 = g("push", fork_push, f"FETCH_HEAD:refs/heads/{base}")
    if r1.returncode != 0:  # fork 领先/分岔 → 与上游对齐
        r2 = g("push", "--force", fork_push, f"FETCH_HEAD:refs/heads/{base}")
        if r2.returncode != 0:
            # 脱敏：cred_url 把 user:token 注入 URL，git 失败信息会回显完整凭据
            msg = re.sub(r"https://[^@/]+@", "https://***@", (r1.stderr or r2.stderr).strip())
            print(f"[WARN] fork base 同步失败: {msg[:200]}")
            return False
    print(f"[OK] fork base 已同步上游 {base}")
    return True


def ensure_pushed(cfg, repo_key, ws_root, branch, dry):
    """把本地修复分支推到目标仓：CI fork 模式推 ci_fork，否则推 fork_user 名下 fork。

    返回三态：branch（push 成功）/ "skip"（有意跳过：无 clone 或本地无修复分支，
    不视为失败）/ None（fork 同步或 push 真实失败，主路径据此 return 2）。
    """
    repo_cfg = cfg["repos"][repo_key]
    local_dir = os.path.join(ws_root, repo_cfg.get("ws_dirname", repo_key))
    if not os.path.isdir(os.path.join(local_dir, ".git")):
        print(f"[SKIP] 本地 clone 不存在: {local_dir}，无法 push（NEEDS_HUMAN 手动处理）")
        return "skip"
    rc = subprocess.run(
        [GIT_BIN, "-C", local_dir, "rev-parse", "--verify", branch], capture_output=True, check=False
    ).returncode
    if rc != 0:
        print(f"[SKIP] 本地无分支 {branch}（agent 未产生修复 commit），不建 PR")
        return "skip"
    ci_fork = repo_cfg.get("ci_fork_url")
    if ci_fork:
        if not sync_upstream_to_fork(cfg, repo_key, ws_root, dry):
            print(f"[FATAL] fork base 同步失败，无法 push {branch}（详见上方日志）", file=sys.stderr)
            return None
        target_desc = ci_fork
        remote_url = cred_url(cfg, ci_fork)
    else:
        target_desc = f"{cfg['fork_user']}/{repo_cfg['repo']}"
        remote_url = f"https://{cfg.get('gitcode_username') or 'oauth2'}:{cfg['gitcode_token']}@gitcode.com/{cfg['fork_user']}/{repo_cfg['repo']}.git"
    if dry:
        print(f"[DRY] git push {target_desc} {branch}（force-with-lease）@ {local_dir}")
        return branch
    # 新建分支直接 push；已存在则 force-with-lease 更新（幂等）
    r1 = subprocess.run(
        [GIT_BIN, "-C", local_dir, "push", remote_url, branch], capture_output=True, text=True, check=False
    )
    if r1.returncode != 0:
        r2 = subprocess.run(
            [GIT_BIN, "-C", local_dir, "push", "--force-with-lease", remote_url, branch],
            capture_output=True,
            text=True,
            check=False,
        )
        if r2.returncode != 0:
            # 脱敏：cred_url 把 user:token 注入 URL，git 失败信息会回显完整凭据
            msg = re.sub(r"https://[^@/]+@", "https://***@", (r1.stderr or r2.stderr).strip())
            print(f"[FATAL] push 失败: {msg[:200]}", file=sys.stderr)
            return None
    print(f"[OK] pushed → {target_desc}:{branch}")
    return branch


def issue_title(root_cause):
    summary = re.sub(r"\s+", " ", root_cause or "unknown root cause").strip()
    return f"[daily-ci] {summary[:80]}"


def target_owner_repo(cfg, repo_key):
    """Issue/PR 目标仓：CI fork 模式 → ci_fork_url 解析；否则上游主仓（issue_owner/repo）。"""
    ci_fork = cfg["repos"][repo_key].get("ci_fork_url")
    if ci_fork:
        return fork_url_parts(ci_fork)
    return cfg["issue_owner"], cfg["repos"][repo_key]["repo"]


def ensure_issue(cfg, repo_key, title, comment_body, dry):
    owner, repo = target_owner_repo(cfg, repo_key)
    if dry:
        print(f"[DRY] ensure issue {owner}/{repo}: {title}")
        return {"number": "dry-run", "html_url": "(dry-run)"}
    st, issues = api(cfg, "GET", f"/repos/{owner}/{repo}/issues?state=open&labels={LABEL}&per_page=50")
    for it in issues if isinstance(issues, list) else []:
        if it.get("title") == title:
            st2, _ = api(cfg, "POST", f"/repos/{owner}/{repo}/issues/{it['number']}/comments", {"body": comment_body})
            print(f"[OK] issue #{it['number']} 已存在，追加评论（{st2}）: {it.get('html_url')}")
            return it
    st2, it = api(
        cfg, "POST", f"/repos/{owner}/{repo}/issues", {"title": title, "body": comment_body, "labels": [LABEL]}
    )
    if st2 in (200, 201):
        print(f"[OK] issue created #{it.get('number')}: {it.get('html_url')}")
        return it
    print(f"[WARN] issue create failed ({st2})")
    return None


def ensure_pr(cfg, repo_key, branch, date, dry):
    owner, repo = target_owner_repo(cfg, repo_key)
    ci_fork = cfg["repos"][repo_key].get("ci_fork_url")
    head = branch if ci_fork else f"{cfg['fork_user']}:{branch}"  # 同仓 PR head=分支名
    base = cfg["repos"][repo_key].get("base_branch", "master")
    title = f"[daily-ci] fix for {date} regression ({branch})"
    if dry:
        print(f"[DRY] ensure PR {owner}/{repo} {head} -> {base}")
        return {"number": "dry-run", "html_url": "(dry-run)"}
    st, prs = api(cfg, "GET", f"/repos/{owner}/{repo}/pulls?state=open&head={head}&per_page=10")
    for pr in prs if isinstance(prs, list) else []:
        print(f"[OK] PR #{pr.get('number')} 已存在，复用（force-push 已更新 diff）: {pr.get('html_url')}")
        return pr
    st2, pr = api(
        cfg,
        "POST",
        f"/repos/{owner}/{repo}/pulls",
        {
            "title": title,
            "head": head,
            "base": base,
            "body": f"daily-ci automated fix, see Issue with label `{LABEL}`. branch `{branch}`.",
        },
    )
    if st2 in (200, 201):
        url = (
            pr.get("html_url")
            or pr.get("web_url")
            or f"https://gitcode.com/{owner}/{repo}/merge_requests/{pr.get('number')}"
        )
        print(f"[OK] PR created #{pr.get('number')}: {url}")
        return pr
    print(f"[WARN] PR create failed ({st2}): {pr.get('error', '')}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--state-dir", default=os.path.join(SCRIPT_DIR, "..", "state"))
    ap.add_argument("--workspace", default="/home/llm_nightly/workspace")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_json(os.path.join(SCRIPT_DIR, ".push_config.json"))
    diag = load_json(os.path.join(args.state_dir, "diag_report.json"))
    if not cfg or not diag:
        print("[FATAL] 缺 .push_config.json 或 state/diag_report.json（agent 诊断后才有）", file=sys.stderr)
        return 1
    fix_repo = diag.get("fix_repo")
    if fix_repo not in cfg.get("repos", {}):
        print(f"[FATAL] fix_repo='{fix_repo}' 不在 .push_config.json repos 中", file=sys.stderr)
        return 1
    a_email = (cfg["repos"][fix_repo].get("author_email") or "").strip()
    if not a_email or "noreply" in a_email.lower():
        print(
            f"[FATAL] author_email='{a_email}' 不是 GitCode 账号真实绑定邮箱。CLA 按 author 邮箱关联账号,"
            f"自造 noreply 假邮箱会使 commit 显示 gitcode_user=None 且 CLA 检查失败;"
            f"请改 .push_config.json repos.{fix_repo}.author_email 为账号设置中绑定的邮箱",
            file=sys.stderr,
        )
        return 1
    fixed = load_json(os.path.join(args.state_dir, "fixed_tests.json"), default=[])
    root_cause = diag.get("root_cause", "")
    author = f'{cfg["repos"][fix_repo].get("author_name")} <{cfg["repos"][fix_repo].get("author_email")}>'
    print(f"[INFO] fix_repo={fix_repo} author={author} fixed_tests={fixed}")

    comment = json.dumps(
        {
            "date": args.date,
            "root_cause": root_cause,
            "breaking_mrs": diag.get("breaking_mrs", []),
            "fixed_tests": fixed,
            "fix_summary": diag.get("fix_summary", ""),
        },
        ensure_ascii=False,
        indent=2,
    )

    # 老问题复发短路：签名与历史一致且已有 open PR 覆盖该根因时，agent 置
    # recurrence_short_circuit=true。此时不重建分支、不重验、不建新 PR，
    # 只在既有 Issue 追加「等待合入」评论（reuse_pr 指向老 PR）。
    if diag.get("recurrence_short_circuit"):
        reuse_pr = diag.get("reuse_pr")
        if not reuse_pr:
            # diag_report 由 LLM 生成，字段遗漏完全可能；缺 reuse_pr 时拒绝短路，
            # 避免公开发布「修复已在 PR !None」（与 fix_repo 的 FATAL 校验对称）
            print("[FATAL] recurrence_short_circuit=true 但缺 reuse_pr，请补齐后重跑或走完整流程", file=sys.stderr)
            return 1
        comment = json.dumps(
            {
                "date": args.date,
                "recurrence_short_circuit": True,
                "reuse_pr": reuse_pr,
                "root_cause": root_cause,
                "note": (
                    f"老问题复发：当日失败签名与历史一致，修复已在 PR !{reuse_pr}，"
                    f"等待合入即可闭环；本次不重建分支/不重验/不建新 PR。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        issue = ensure_issue(cfg, fix_repo, issue_title(root_cause), comment, args.dry_run)
        if not issue and not args.dry_run:
            print("[FATAL] 短路评论失败（ensure_issue 未命中且创建失败），需人工重试", file=sys.stderr)
            return 2
        reg = os.path.join(args.state_dir, f"issue_registry_{fix_repo}.jsonl")
        with open(reg, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "date": args.date,
                        "issue": issue and issue.get("number"),
                        "pr": None,
                        "short_circuit": True,
                        "reuse_pr": reuse_pr,
                        "dry_run": args.dry_run,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        print(
            f"[DONE][SHORT-CIRCUIT] issue={issue and issue.get('number')} reuse_pr={reuse_pr} (no new branch/push/PR)"
        )
        return 0

    # 分支名：CI fork 模式 ci_repair_<YYMMDD>（如 ci_repair_260828）；旧模式 daily_fix_<date>
    ymd = args.date.replace("-", "")[2:]
    branch = f"ci_repair_{ymd}" if cfg["repos"][fix_repo].get("ci_fork_url") else f"daily_fix_{args.date}"
    pushed = ensure_pushed(cfg, fix_repo, args.workspace, branch, args.dry_run)
    issue = ensure_issue(cfg, fix_repo, issue_title(root_cause), comment, args.dry_run)
    push_ok = pushed not in (None, "skip")  # 有意跳过与真实失败都不建 PR
    pr = ensure_pr(cfg, fix_repo, branch, args.date, args.dry_run) if push_ok else None

    # per-repo 注册表（幂等审计）
    reg = os.path.join(args.state_dir, f"issue_registry_{fix_repo}.jsonl")
    with open(reg, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "date": args.date,
                    "issue": issue and issue.get("number"),
                    "pr": pr and pr.get("number"),
                    "dry_run": args.dry_run,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    # 退出码与短路分支对称：issue/PR 创建失败或 push 真实失败 → 2；仅有意跳过 → 0
    push_failed = pushed is None
    if not args.dry_run and (issue is None or push_failed or (push_ok and pr is None)):
        print(
            f"[FATAL] 闭环未完成: pushed={pushed!r} "
            f"issue={'创建失败' if issue is None else issue.get('number')} "
            f"pr={'创建失败' if pr is None else pr.get('number')}（注册表已记录，修复后可幂等重跑）",
            file=sys.stderr,
        )
        return 2
    print(f"[DONE] issue={issue and issue.get('number')} pr={pr and pr.get('number')} registry={reg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
