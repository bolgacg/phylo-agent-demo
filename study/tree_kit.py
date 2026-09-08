"""Tree operations for the phylo agent study.

Ground truth AND the agent's tools come from here. Conversions and topology
comparison go through phylo2vec (Scheidwasser et al.); path lengths and clades
go through an independent Newick parser in this file, so the two sides
cross-check each other. Trees are rooted binary, topology only.
"""
import re

import numpy as np
import phylo2vec as p2v

LETTERS = "ABCDEFGHIJKLMNOP"


# ---------- parsing (own implementation, independent of phylo2vec) ----------

def parse(newick):
    """Newick -> nested tuples; leaves are label strings. Internal labels ignored."""
    s = newick.strip().rstrip(";").strip()
    pos = 0

    def node():
        nonlocal pos
        if s[pos] == "(":
            pos += 1
            children = [node()]
            while s[pos] == ",":
                pos += 1
                children.append(node())
            if s[pos] != ")":
                raise ValueError(f"expected ) at {pos} in {s!r}")
            pos += 1
            # swallow an internal-node label if present
            m = re.match(r"[^,();]*", s[pos:])
            pos += m.end()
            return tuple(children)
        m = re.match(r"[^,();]+", s[pos:])
        if not m:
            raise ValueError(f"expected label at {pos} in {s!r}")
        pos += m.end()
        return m.group().strip()

    t = node()
    if pos != len(s):
        raise ValueError(f"trailing text at {pos} in {s!r}")
    return t


def leaves(tree):
    if isinstance(tree, str):
        return [tree]
    out = []
    for c in tree:
        out.extend(leaves(c))
    return out


def serialize(tree, rng=None):
    """Nested tuples -> Newick. With rng, children are shuffled: same topology,
    different string."""
    def rec(t):
        if isinstance(t, str):
            return t
        cs = list(t)
        if rng is not None:
            rng.shuffle(cs)
        return "(" + ",".join(rec(c) for c in cs) + ")"
    return rec(tree) + ";"


def _path_to_root(tree, label):
    chain = []

    def dfs(t, acc):
        if isinstance(t, str):
            if t == label:
                chain.extend(acc + [t])
                return True
            return False
        return any(dfs(c, acc + [t]) for c in t)

    dfs(tree, [])
    if not chain:
        raise ValueError(f"leaf {label!r} not in tree")
    return chain  # [root, ..., leaf]


def path_length(newick, a, b):
    """Number of branches on the path between leaves a and b."""
    tree = parse(newick)
    pa, pb = _path_to_root(tree, a), _path_to_root(tree, b)
    i = 0
    while i < min(len(pa), len(pb)) and pa[i] is pb[i]:
        i += 1
    return (len(pa) - i) + (len(pb) - i)


def clade_contains(newick, a, b, c):
    """Does the smallest clade containing leaves a and b also contain c?"""
    tree = parse(newick)
    pa, pb = _path_to_root(tree, a), _path_to_root(tree, b)
    i = 0
    while i < min(len(pa), len(pb)) and pa[i] is pb[i]:
        i += 1
    mrca = pa[i - 1]
    return c in leaves(mrca)


def count_leaves(newick):
    return len(leaves(parse(newick)))


# ---------- phylo2vec bridge ----------

def _label_map(newick):
    labs = sorted(set(leaves(parse(newick))), key=lambda x: (len(x), x))
    return {lab: i for i, lab in enumerate(labs)}


def to_int_newick(newick):
    """Letter-labelled Newick -> integer-labelled (alphabetical map), for phylo2vec."""
    m = _label_map(newick)
    def rec(t):
        if isinstance(t, str):
            return str(m[t])
        return "(" + ",".join(rec(c) for c in t) + ")"
    return rec(parse(newick)) + ";"


def newick_to_vector(newick):
    nw = newick.strip()
    if re.search(r"[A-Za-z]", nw.replace(";", "")):
        nw = to_int_newick(nw)
    else:
        nw = serialize(parse(nw))  # strip internal labels, normalise
    return p2v.from_newick(nw)


def vector_to_newick(vec):
    v = np.asarray(list(vec), dtype=np.int64)
    nw = p2v.to_newick(v)
    return serialize(parse(nw))  # strip phylo2vec's internal node labels


def same_topology(n1, n2):
    v1, v2 = newick_to_vector(n1), newick_to_vector(n2)
    if len(v1) != len(v2):
        return False
    l1 = sorted(leaves(parse(n1)))
    l2 = sorted(leaves(parse(n2)))
    if l1 != l2:
        return False
    return p2v.stats.robinson_foulds(v1, v2) == 0.0


def _is_binary(tree):
    if isinstance(tree, str):
        return True
    return len(tree) == 2 and all(_is_binary(c) for c in tree)


def newick_equal(n1, n2):
    """Semantic equality of two Newick strings (same leaves, same topology).
    n1 may be model output: validated first, and phylo2vec's Rust panics
    (BaseException, not Exception) are caught."""
    try:
        t1, t2 = parse(n1), parse(n2)
        if sorted(leaves(t1)) != sorted(leaves(t2)):
            return False
        if not (_is_binary(t1) and _is_binary(t2)):
            return False
        return same_topology(n1, n2)
    except BaseException:
        return False


# ---------- agent tool dispatch ----------

TOOL_DOC = """You may use these tools. To call one, reply with EXACTLY one line and nothing else:
TOOL: count_leaves | <newick>
TOOL: path_length | <newick> | <leaf1> | <leaf2>        (branches on the path between two leaves)
TOOL: clade_contains | <newick> | <a> | <b> | <c>       (does the smallest group holding a and b also hold c?)
TOOL: same_topology | <newick1> | <newick2>             (same branching structure? true/false)
TOOL: vector_to_newick | <comma-separated integers>     (expand a phylo2vec vector to a Newick tree)
TOOL: newick_to_vector | <newick>
After each TOOL RESULT you may call another tool, or finish with one line:
ANSWER: <your answer>"""


def run_tool(name, args):
    name = name.strip().lower()
    args = [a.strip() for a in args]
    if name == "count_leaves":
        return str(count_leaves(args[0]))
    if name == "path_length":
        return str(path_length(args[0], args[1], args[2]))
    if name == "clade_contains":
        return "yes" if clade_contains(args[0], args[1], args[2], args[3]) else "no"
    if name == "same_topology":
        return "true" if same_topology(args[0], args[1]) else "false"
    if name == "vector_to_newick":
        vec = [int(x) for x in re.split(r"[,\s]+", args[0].strip("[] ")) if x != ""]
        return vector_to_newick(vec)
    if name == "newick_to_vector":
        return ",".join(str(int(x)) for x in newick_to_vector(args[0]))
    raise ValueError(f"unknown tool {name!r}")
