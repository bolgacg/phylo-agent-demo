"""Run the task set against an OpenAI-compatible chat endpoint (llama-server).

Conditions:
  unaided  question only, model answers from reading the Newick
  probe    first "can you do this reliably without tools, yes/no", then unaided
  agent    tools from tree_kit offered through a plain text protocol, max 5 rounds

Every row logs prompt, raw output, tool transcript. Usage:
  python runner.py <condition> --model NAME --url http://localhost:8080/v1 \
      --tasks tasks.jsonl --out pred.jsonl [--limit N] [--parallel K]
"""
import argparse
import concurrent.futures as cf
import json
import re
import sys
import time

import requests

import tree_kit

SYSTEM = ("You are answering questions about phylogenetic trees. "
          "Be precise and answer only in the requested format.")

ANSWER_RE = re.compile(r"ANSWER:\s*(.+)", re.IGNORECASE)
TOOL_RE = re.compile(r"TOOL:\s*([a-z_]+)\s*\|(.*)", re.IGNORECASE)


def chat(url, messages, max_tokens=160, temperature=0.0, timeout=240):
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    r = requests.post(f"{url}/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": "Bearer none"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def extract_answer(text):
    m = None
    for m in ANSWER_RE.finditer(text):
        pass
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def run_unaided(url, task):
    prompt = task["question"] + "\n\nEnd your reply with one line of the form\nANSWER: <your answer>"
    out = chat(url, [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}])
    return {"raw": out, "pred": extract_answer(out)}


def run_probe(url, task):
    ask = (task["question"] +
           "\n\nDo NOT answer the question yet. First: can you determine the answer "
           "reliably by yourself, without any software tools? Reply with only the word yes or no.")
    dec = chat(url, [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": ask}], max_tokens=8)
    rec = run_unaided(url, task)
    rec["claim_raw"] = dec
    rec["claims_know"] = bool(re.match(r"\W*yes\b", dec.strip().lower()))
    return rec


USAGE = {
    "count_leaves": "TOOL: count_leaves | <newick>   (1 field after the tool name)",
    "path_length": "TOOL: path_length | <newick> | <leaf1> | <leaf2>   (3 fields)",
    "clade_contains": "TOOL: clade_contains | <newick> | <a> | <b> | <c>   (4 fields: the tree, "
                      "the two taxa that define the group, then the taxon to test)",
    "same_topology": "TOOL: same_topology | <newick1> | <newick2>   (2 fields)",
    "vector_to_newick": "TOOL: vector_to_newick | 0,0,2,1   (1 field; a valid phylo2vec vector "
                        "always starts with 0)",
    "newick_to_vector": "TOOL: newick_to_vector | <newick>   (1 field)",
}


def helpful_error(name, args, exc):
    u = USAGE.get(name, "unknown tool; available: " + ", ".join(USAGE))
    return (f"error: that call to {name or 'the tool'} could not run ({str(exc)[:80]}). "
            f"You gave {len(args)} field(s). Correct usage: {u}")


def run_agent(url, task, max_rounds=5, helpful=False):
    preamble = (task["question"] + "\n\n" + tree_kit.TOOL_DOC)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": preamble}]
    transcript, tool_calls, malformed = [], 0, 0
    for _ in range(max_rounds):
        out = chat(url, messages, max_tokens=200)
        transcript.append({"model": out})
        m = TOOL_RE.search(out)
        if not m or ANSWER_RE.search(out):
            return {"raw": out, "pred": extract_answer(out), "transcript": transcript,
                    "tool_calls": tool_calls, "malformed": malformed}
        name, argstr = m.group(1), m.group(2)
        args = [a.strip() for a in argstr.split("|")]
        try:
            result = tree_kit.run_tool(name, args)
            tool_calls += 1
        except BaseException as e:  # phylo2vec panics on malformed input are BaseException
            result = helpful_error(name, args, e) if helpful else f"error: {str(e)[:120]}"
            malformed += 1
        transcript.append({"tool": f"{name}", "args": args, "result": result})
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content": f"TOOL RESULT: {result}\n\n"
                         "Call another tool, or finish with one line: ANSWER: <your answer>"})
    return {"raw": "", "pred": "", "transcript": transcript, "timeout_rounds": True,
            "tool_calls": tool_calls, "malformed": malformed}


def run_agent2(url, task):
    """The remedy arm: identical agent, but tool errors name the missing field
    and show correct usage instead of a bare Python message."""
    return run_agent(url, task, helpful=True)


RUNNERS = {"unaided": run_unaided, "probe": run_probe, "agent": run_agent,
           "agent2": run_agent2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", choices=list(RUNNERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://localhost:8080/v1")
    ap.add_argument("--tasks", default="tasks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(a.tasks)]
    if a.limit:
        tasks = tasks[:a.limit]
    fn = RUNNERS[a.condition]
    t0, done = time.time(), 0
    rows = [None] * len(tasks)

    def work(i):
        task = tasks[i]
        try:
            rec = fn(a.url, task)
        except Exception as e:
            rec = {"error": str(e)[:200], "pred": ""}
        rec.update({"id": task["id"], "type": task["type"], "truth": task["truth"],
                    "model": a.model, "condition": a.condition})
        return i, rec

    with cf.ThreadPoolExecutor(a.parallel) as ex:
        for i, rec in ex.map(work, range(len(tasks))):
            rows[i] = rec
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} {time.time()-t0:.0f}s", file=sys.stderr)

    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{a.out}: {len(rows)} rows in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
