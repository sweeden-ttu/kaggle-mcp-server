"""Extract feature vectors from all MLSysEng automation branches.

Scans each branch's diff against main and produces a JSON manifest
of what each branch contributed. This becomes the training data for
the GBDT expert router.
"""

import json
import subprocess
import sys
from pathlib import Path


def run(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return result.stdout.strip()


def extract_branch_features(branch: str) -> dict:
    ref = f"origin/{branch}"

    stat_raw = run(f"git diff main..{ref} --shortstat 2>/dev/null")
    files_changed = 0
    insertions = 0
    deletions = 0
    for part in stat_raw.split(","):
        part = part.strip()
        if "file" in part:
            files_changed = int(part.split()[0])
        elif "insertion" in part:
            insertions = int(part.split()[0])
        elif "deletion" in part:
            deletions = int(part.split()[0])

    tree_raw = run(f"git ls-tree -r --name-only {ref} 2>/dev/null")
    files = tree_raw.splitlines() if tree_raw else []

    diff_files_raw = run(f"git diff main..{ref} --name-only 2>/dev/null")
    diff_files = diff_files_raw.splitlines() if diff_files_raw else []

    commit_count = len(run(f"git log main..{ref} --oneline 2>/dev/null").splitlines())
    last_msg = run(f"git log {ref} -1 --format='%s' 2>/dev/null")
    last_date = run(f"git log {ref} -1 --format='%aI' 2>/dev/null")

    def has_file(pattern: str) -> bool:
        return any(pattern in f for f in files)

    def count_files(pattern: str) -> int:
        return sum(1 for f in files if pattern in f)

    py_diff = [f for f in diff_files if f.endswith(".py")]
    rs_diff = [f for f in diff_files if f.endswith(".rs")]
    rb_diff = [f for f in diff_files if f.endswith(".rb")]

    modules_present = []
    for mod in ["server", "database", "embeddings", "expert_registry",
                 "loop_controller", "docling_worker", "skill_generator"]:
        if has_file(mod):
            modules_present.append(mod)

    return {
        "branch": branch,
        "files_changed": files_changed,
        "insertions": insertions,
        "deletions": deletions,
        "net_lines": insertions - deletions,
        "commit_count": commit_count,
        "last_commit_msg": last_msg,
        "last_commit_date": last_date,
        "python_files_changed": len(py_diff),
        "rust_files_changed": len(rs_diff),
        "ruby_files_changed": len(rb_diff),
        "has_server": has_file("server.py"),
        "has_database": has_file("database.py"),
        "has_embeddings": has_file("embeddings.py"),
        "has_expert_registry": has_file("expert_registry"),
        "has_loop_controller": has_file("loop_controller"),
        "has_docling_worker": has_file("docling_worker"),
        "has_tests": count_files("test_") > 0,
        "test_file_count": count_files("test_"),
        "has_rust": count_files(".rs") > 0,
        "rust_file_count": count_files(".rs"),
        "has_ruby": count_files(".rb") > 0,
        "has_skills_yaml": has_file("skills.yaml"),
        "has_skill_generator": has_file("skill_generator"),
        "has_kmap": has_file("kmap"),
        "has_ac_matcher": has_file("ac_matcher"),
        "modules_present": modules_present,
        "module_count": len(modules_present),
        "diff_files": diff_files[:20],
        "maturity_score": (
            insertions / 100.0
            + commit_count * 2.0
            + len(modules_present) * 5.0
            + (10.0 if has_file("test_") else 0.0)
            + (8.0 if has_file("skills.yaml") else 0.0)
            + (5.0 if count_files(".rs") > 0 else 0.0)
        ),
    }


def main():
    branches_raw = run("git branch -a | grep -E 'competition-mlsyseng|sweeden-ttu/mlsyseng' | sed 's|remotes/origin/||' | sed 's|^[[:space:]]*/||' | sort -u")
    branches = [b.strip() for b in branches_raw.splitlines() if b.strip()]
    print(f"Scanning {len(branches)} branches...", file=sys.stderr)

    results = []
    for i, branch in enumerate(branches):
        if i % 10 == 0:
            print(f"  [{i}/{len(branches)}] {branch}", file=sys.stderr)
        try:
            features = extract_branch_features(branch)
            results.append(features)
        except Exception as e:
            print(f"  SKIP {branch}: {e}", file=sys.stderr)

    results.sort(key=lambda r: r["maturity_score"], reverse=True)

    output = {
        "total_branches": len(results),
        "top_branches": [r["branch"] for r in results[:10]],
        "summary": {
            "total_insertions": sum(r["insertions"] for r in results),
            "total_commits": sum(r["commit_count"] for r in results),
            "branches_with_tests": sum(1 for r in results if r["has_tests"]),
            "branches_with_rust": sum(1 for r in results if r["has_rust"]),
            "branches_with_full_stack": sum(
                1 for r in results
                if r["has_server"] and r["has_database"] and r["has_embeddings"]
                and r["has_expert_registry"] and r["has_loop_controller"]
            ),
            "avg_maturity_score": sum(r["maturity_score"] for r in results) / max(len(results), 1),
            "max_maturity_score": results[0]["maturity_score"] if results else 0,
        },
        "branches": results,
    }

    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
