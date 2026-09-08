"""Generate the task set. Seeded; every truth is computed, never typed.

Five task types, letter-labelled trees except p2v (integer-labelled):
  leaf_count     how many leaf taxa (n varies 5..12)
  closer         which of two taxa is nearer a target, by branches on the path
  clade          does the smallest group holding a and b also hold c
  same_topology  two Newick strings, same branching structure or not
  p2v            expand a phylo2vec vector to its Newick tree

Writes tasks.jsonl. Run inside the study venv.
"""
import json
import random

import numpy as np
import phylo2vec as p2v

from tree_kit import (LETTERS, parse, serialize, leaves, path_length,
                      clade_contains, vector_to_newick, to_int_newick)

rng = random.Random(42)
np_rng = np.random.default_rng(42)


def letter_tree(n):
    """Random topology on n leaves, letter labels, no internal labels."""
    v = p2v.sample_vector(n)
    nw = p2v.to_newick(v)
    t = parse(nw)

    def relabel(t):
        if isinstance(t, str):
            return LETTERS[int(t)]
        return tuple(relabel(c) for c in t)

    return serialize(relabel(t))


tasks = []


def add(ttype, question, truth, **meta):
    tasks.append({"id": f"{ttype}-{sum(1 for t in tasks if t['type'] == ttype) + 1:02d}",
                  "type": ttype, "question": question, "truth": str(truth), **meta})


PREFACE = ("The tree below is written in Newick format: nested parentheses group taxa, "
           "so ((A,B),C); means A and B are sisters and C sits outside that pair.\n\n")

# 1. leaf_count, n 5..12
for i in range(20):
    n = 5 + (i % 8)
    nw = letter_tree(n)
    add("leaf_count",
        PREFACE + f"Tree: {nw}\n\nHow many leaf taxa does this tree contain? "
        "Reply with the number only.",
        n, newick=nw)

# 2. closer, n = 8
made = 0
while made < 25:
    nw = letter_tree(8)
    labs = leaves(parse(nw))
    a, b, c = rng.sample(labs, 3)
    db, dc = path_length(nw, a, b), path_length(nw, a, c)
    if db == dc:
        continue
    truth = b if db < dc else c
    add("closer",
        PREFACE + f"Tree: {nw}\n\nWhich taxon is closer to {a} in this tree, {b} or {c}? "
        "Closer means fewer branches on the path between them. Reply with the single letter only.",
        truth, newick=nw, target=a, cand=[b, c], dists=[db, dc])
    made += 1

# 3. clade, n = 8, yes/no balanced
made, want_yes = 0, True
while made < 25:
    nw = letter_tree(8)
    labs = leaves(parse(nw))
    a, b, c = rng.sample(labs, 3)
    truth = clade_contains(nw, a, b, c)
    if truth != want_yes:
        continue
    add("clade",
        PREFACE + f"Tree: {nw}\n\nConsider the smallest group (clade) that contains both {a} and {b}. "
        f"Does that group also contain {c}? Reply yes or no only.",
        "yes" if truth else "no", newick=nw, abc=[a, b, c])
    made += 1
    want_yes = not want_yes

# 4. same_topology, n = 7, yes/no balanced
made, want_same = 0, True
while made < 25:
    n = 7
    v1 = p2v.sample_vector(n)
    if want_same:
        nw_int = p2v.to_newick(v1)
        t = parse(nw_int)

        def relabel(t):
            if isinstance(t, str):
                return LETTERS[int(t)]
            return tuple(relabel(c) for c in t)

        t = relabel(t)
        nw1 = serialize(t)
        nw2 = serialize(t, rng=rng)   # shuffled children: same topology, different string
        if nw1 == nw2:
            continue
        truth = True
    else:
        v2 = p2v.sample_vector(n)
        if p2v.stats.robinson_foulds(v1, v2) == 0.0:
            continue

        def relab(nw):
            t = parse(nw)

            def r(t):
                if isinstance(t, str):
                    return LETTERS[int(t)]
                return tuple(r(c) for c in t)

            return serialize(r(t))

        nw1, nw2 = relab(p2v.to_newick(v1)), relab(p2v.to_newick(v2))
        truth = False
    add("same_topology",
        "Two trees in Newick format follow. Nested parentheses group taxa; the order of taxa "
        "inside a parenthesis does not matter, only the grouping does.\n\n"
        f"Tree 1: {nw1}\nTree 2: {nw2}\n\n"
        "Do these two trees show the same branching structure (the same topology)? Reply yes or no only.",
        "yes" if truth else "no", newick=nw1, newick2=nw2)
    made += 1
    want_same = not want_same

# 5. p2v, n 5..7, integer labels
for i in range(25):
    n = 5 + (i % 3)
    v = p2v.sample_vector(n)
    truth_nw = vector_to_newick(v)
    add("p2v",
        "phylo2vec is a published scheme that encodes a phylogenetic tree as an integer vector. "
        f"What is the Newick string of the tree encoded by the phylo2vec vector [{','.join(str(int(x)) for x in v)}]? "
        "Use integer leaf labels and no branch lengths. Reply with the Newick string only.",
        truth_nw, vector=[int(x) for x in v])

with open("tasks.jsonl", "w") as f:
    for t in tasks:
        f.write(json.dumps(t) + "\n")

from collections import Counter
print(len(tasks), "tasks:", dict(Counter(t["type"] for t in tasks)))
