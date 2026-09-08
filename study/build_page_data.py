"""Turn scored predictions into the page's data.js. Every number on the page
comes from here; the page never types a number.

Run in study/ after score.py:  python build_page_data.py > ../site/data.js
"""
import glob
import json
import sys
from collections import defaultdict

MODELS = ["qwen3b", "llama3b", "gemma4b", "llama1b", "mimir"]
MODEL_LABELS = {"qwen3b": "Qwen 2.5 3B", "llama3b": "Llama 3.2 3B",
                "gemma4b": "Gemma 3 4B", "llama1b": "Llama 3.2 1B",
                "mimir": "DFM Mimir 1B"}
TYPES = ["leaf_count", "closer", "clade", "same_topology", "p2v"]
TYPE_LABELS = {"leaf_count": "count the taxa", "closer": "which is closer",
               "clade": "is it in the group", "same_topology": "same tree?",
               "p2v": "expand a phylo2vec vector"}
CONDS = ["unaided", "probe", "agent", "agent2"]


def rows_of(model, cond):
    fn = f"pred_{model}_{cond}.scored.jsonl"
    try:
        return [json.loads(l) for l in open(fn)]
    except FileNotFoundError:
        return []


def trim(s, n):
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def main():
    results = json.load(open("results.json"))
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open("tasks.jsonl")}

    table = {}
    for m in MODELS:
        for c in CONDS:
            per = results["table"].get(f"{m}|{c}")
            if per:
                table[f"{m}|{c}"] = per

    browser = []
    per_task = defaultdict(dict)
    for m in MODELS:
        for c in CONDS:
            for r in rows_of(m, c):
                e = {"ok": bool(r.get("ok")), "pred": trim(r.get("pred", ""), 160)}
                if c == "probe":
                    e["claim"] = bool(r.get("claims_know"))
                if c == "agent":
                    e["calls"] = r.get("tool_calls", 0)
                    tr = []
                    for step in (r.get("transcript") or [])[:8]:
                        if "model" in step:
                            tr.append({"who": "model", "text": trim(step["model"], 300)})
                        else:
                            tr.append({"who": "tool", "text": trim(
                                f"{step.get('tool')} -> {step.get('result')}", 300)})
                    e["tr"] = tr
                per_task[r["id"]][f"{m}|{c}"] = e

    for tid, t in tasks.items():
        browser.append({"id": tid, "type": t["type"], "q": t["question"],
                        "truth": t["truth"], "runs": per_task.get(tid, {})})

    data = {"models": MODELS, "model_labels": MODEL_LABELS, "types": TYPES,
            "type_labels": TYPE_LABELS, "conds": CONDS, "table": table,
            "honesty": results["honesty"], "agent": results["agent"],
            "n_tasks": len(tasks), "tasks": browser}
    sys.stdout.write("const DATA = " + json.dumps(data) + ";\n")


if __name__ == "__main__":
    main()
