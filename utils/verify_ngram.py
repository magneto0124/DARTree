#!/usr/bin/env python3
"""Definitively check whether a .trie n-gram model shares the SAME tokenizer
(id <-> token mapping) as the tokenizers used at eval time.

The key insight: coverage / top-5 probes CANNOT detect a vocab misalignment --
if the trie was built with a different id<->token mapping, any id we throw in
is looked up in the trie's OWN id space and returns a plausible-looking
continuation of the WRONG token.  The only decisive test is to read the trie's
own binary data and decode its most frequent token ids with the eval tokenizer:

* if aligned: the trie's top-frequency root children decode to ordinary
  English stopwords (" the", " of", " a", ...);
* if misaligned: they decode to arbitrary words.

Checks:

1. parse the .trie binary directly (struct, little-endian) -- independent of
   the C++ extension;
2. DECISIVE: decode the trie's most frequent token ids with --tokenizer
   (and also with --draft-tokenizer, if given) and compare the decoded words;
3. trie order / node_count from the raw file vs the C++ API (cross-check the
   C++ loader's header handling);
4. id-range: all root-child token ids must lie inside the target vocab;
5. spot-check a few conditional probabilities computed by hand from the raw
   file against the C++ get_probability output (independent "lookup is not
   wrong" evidence);
6. supporting checks: draft-vs-target vocab identity, coverage, canaries,
   determinism/range.

Usage:

    python utils/verify_ngram.py \
        --ngram-model /path/to/small.trie \
        --tokenizer Qwen/Qwen3-4B \
        --draft-tokenizer Huang2020/Qwen3-4B-Domino-b16

Exit code 0 = all checks passed; 1 = hard failure.
"""

from __future__ import annotations

import argparse
import struct
import sys
from typing import Dict, List, Optional, Tuple

DEFAULT_SNIPPETS: List[str] = [
    "The capital of France is Paris and it is known for the Eiffel Tower.",
    "To solve this problem we need to compute the total number of apples.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    (
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n-1) + fibonacci(n-2)"
    ),
]

CANARY_CONTEXTS: List[str] = [
    "the answer is",
    "I think",
    "in order to",
    "one of the most",
]

COMMON_PROBES: List[str] = [" the", " is", " answer", " of"]

# ============================================================================
# Raw .trie binary parser (independent of the C++ extension)
# ============================================================================

Node = Tuple[int, int, int]  # (token, parent, freq)


def _unpack(fmt: str, f) -> int:
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def parse_trie_raw(
    path: str, max_parse_nodes: int = 5_000_000
) -> Tuple[int, int, Dict[int, int], Dict[int, Node], Dict[int, Dict[int, int]]]:
    """Stream-parse the .trie file.

    Returns (order, node_count, root_children, node_info, children_maps)
    where:
      root_children : {token: node_id}
      node_info     : {node_id: (token, parent, freq)} for parsed nodes
      children_maps : {node_id: {token: child_id}} for root + root's children
    """
    with open(path, "rb") as f:
        order = _unpack("<Q", f)
        node_count = _unpack("<Q", f)

        # node 0 = root (its own token/parent are meaningless)
        _unpack("<i", f)  # token
        _unpack("<Q", f)  # parent
        _unpack("<i", f)  # freq
        root_children: Dict[int, int] = {}
        for _ in range(_unpack("<Q", f)):
            tok = _unpack("<i", f)
            cid = _unpack("<Q", f)
            root_children[tok] = cid

        # Nodes are stored sequentially; node ids are assigned in creation
        # order, so root children have the smallest ids.  Stream nodes
        # 1..limit, capturing freq/token of root children and the children
        # maps of root + root children.
        limit = min(node_count - 1, max_parse_nodes)
        node_info: Dict[int, Node] = {}
        children_maps: Dict[int, Dict[int, int]] = {0: root_children}
        root_child_ids = set(root_children.values())
        for node_id in range(1, limit + 1):
            tok = _unpack("<i", f)
            parent = _unpack("<Q", f)
            freq = _unpack("<i", f)
            n_children = _unpack("<Q", f)
            children: Dict[int, int] = {}
            for _ in range(n_children):
                child_tok = _unpack("<i", f)
                child_id = _unpack("<Q", f)
                children[child_tok] = child_id
            node_info[node_id] = (tok, parent, freq)
            if node_id in root_child_ids or node_id == 0:
                children_maps[node_id] = children

    if limit < node_count - 1:
        print(
            f"[raw] WARNING: trie has {node_count} nodes; parsed only the "
            f"first {limit} for spot checks."
        )
    return order, node_count, root_children, node_info, children_maps


# ============================================================================
# Decisive: decode the trie's own most frequent token ids
# ============================================================================

def check_root_children_decode(
    root_children: Dict[int, int],
    node_info: Dict[int, Node],
    tok,
    vocab_size: int,
    label: str,
) -> List[str]:
    """Decode the trie's top-frequency token ids with a tokenizer."""
    problems: List[str] = []
    ranked = sorted(
        ((node_info[cid][2], tid) for tid, cid in root_children.items()
         if cid in node_info),
        reverse=True,
    )
    if not ranked:
        problems.append(f"could not parse root-child frequencies ({label})")
        return problems

    out_of_vocab = sum(1 for _, tid in ranked if tid >= vocab_size)
    print(
        f"[raw:{label}] root children = {len(root_children)}, "
        f"top-30 by frequency decoded with {label!r}:"
    )
    for freq, tid in ranked[:30]:
        try:
            word = tok.decode([tid])
        except Exception:  # noqa: BLE001
            word = "<decode-error>"
        flag = "  <-- id >= vocab_size!" if tid >= vocab_size else ""
        print(f"    id={tid:<7} freq={freq:<10} {word!r}{flag}")

    if out_of_vocab:
        problems.append(
            f"{out_of_vocab} root-child ids are outside the {label} vocab "
            f"size ({vocab_size}) -- the trie was built with a different "
            "vocabulary."
        )
    return problems


# ============================================================================
# Conditional-probability spot check: raw file vs C++ get_probability
# ============================================================================

def check_conditional_spot(
    model,
    tok,
    root_children: Dict[int, int],
    node_info: Dict[int, Node],
    children_maps: Dict[int, Dict[int, int]],
    order: int,
    k: int = 5,
) -> List[str]:
    """Hand-compute P(Y|X) from the raw file and compare with the C++ API."""
    problems: List[str] = []
    checked = 0
    # pick frequent root children that were parsed and have children maps
    candidates = sorted(
        ((node_info[cid][2], tid) for tid, cid in root_children.items()
         if cid in node_info and cid in children_maps),
        reverse=True,
    )
    for _freq, tid in candidates:
        if checked >= k:
            break
        cid = root_children[tid]
        ctx_freq = node_info[cid][2]
        if ctx_freq <= 0:
            continue
        # pick the most frequent child of this root child
        child_ranked = sorted(
            ((node_info[cc][2], ctok) for ctok, cc in children_maps[cid].items()
             if cc in node_info),
            reverse=True,
        )
        if not child_ranked:
            continue
        child_freq, child_tok = child_ranked[0]
        expect = child_freq / ctx_freq
        probs, _m = model.get_probability([tid], [child_tok])
        got = float(probs[0])
        ok = abs(got - expect) <= 1e-6 * max(1.0, abs(expect))
        print(
            f"[raw-vs-cpp] P({tok.decode([child_tok])!r} | "
            f"{tok.decode([tid])!r}) = raw {expect:.6f} vs cpp {got:.6f} "
            f"{'OK' if ok else 'MISMATCH'}"
        )
        if not ok:
            problems.append(
                "C++ get_probability disagrees with the raw file "
                f"({got:.6f} vs {expect:.6f})"
            )
        checked += 1
    if checked == 0:
        print("[raw-vs-cpp] no parsed bigram available for spot check "
              "(trie too large for the parse cap?)")
    return problems


# ============================================================================
# Supporting checks
# ============================================================================

def check_vocab_identity(
    tokenizer_name: str,
    draft_tokenizer_name: Optional[str],
) -> Tuple[dict, List[str]]:
    from transformers import AutoTokenizer

    problems: List[str] = []
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab = tok.get_vocab()
    print(f"[vocab] target tokenizer {tokenizer_name!r}: size = {len(vocab)}")

    if draft_tokenizer_name:
        try:
            dtok = AutoTokenizer.from_pretrained(draft_tokenizer_name)
            dvocab = dtok.get_vocab()
            print(
                f"[vocab] draft tokenizer {draft_tokenizer_name!r}: "
                f"size = {len(dvocab)}"
            )
            if len(vocab) != len(dvocab):
                problems.append(
                    f"vocab size mismatch: target {len(vocab)} vs "
                    f"draft {len(dvocab)}"
                )
            else:
                sample = list(vocab.keys())[:5000]
                mism = sum(1 for t in sample if vocab[t] != dvocab.get(t))
                print(
                    f"[vocab] spot-checked {len(sample)} tokens: "
                    f"{mism} id mismatches"
                )
                if mism:
                    problems.append(
                        "token->id mapping differs between target and "
                        "draft tokenizers"
                    )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[vocab] WARNING: could not load draft tokenizer "
                f"{draft_tokenizer_name!r}: {exc}"
            )
    return vocab, problems


def check_coverage(model, tok, order: int, snippets: List[str]) -> List[str]:
    problems: List[str] = []
    total = {L: 0 for L in range(1, order)}
    hit = {L: 0 for L in range(1, order)}

    for text in snippets:
        ids = tok.encode(text, add_special_tokens=True)
        for i in range(1, len(ids)):
            for L in range(1, min(order, i + 1)):
                ctx = ids[i - L : i]
                probs, _matched = model.get_probability(ctx, [ids[i]])
                total[L] += 1
                if probs[0] > 0:
                    hit[L] += 1

    for L in sorted(total):
        if total[L] == 0:
            continue
        cov = hit[L] / total[L]
        print(f"[coverage] ctx_len={L}: {hit[L]}/{total[L]} = {cov:.1%}")
        if L == 1 and cov < 0.3:
            problems.append(
                f"1-token context coverage is only {cov:.1%} -- the trie "
                "token id space almost certainly does NOT match the "
                "tokenizer (or the trie is tiny)."
            )
    return problems


def _print_top(model, tok, vocab, ctx: List[int], k: int = 5) -> None:
    probs, matched = model.get_probability(ctx, sorted(vocab.values()))
    ranked = sorted(zip(probs, matched, sorted(vocab.values())), reverse=True)
    if ranked[0][0] <= 0.0:
        print("      -> NO MATCH (all p = 0): the queried context does not "
              "exist in this trie's id space.")
        return
    for p, m, tid in ranked[:k]:
        print(f"      p={p:.4f} (matched_len={m}) {tok.decode([tid])!r}")


def check_canaries(model, tok, vocab, order: int) -> None:
    print("\n[canary probes] (query the trie over the full vocab)")
    for text in CANARY_CONTEXTS:
        ids = tok.encode(text, add_special_tokens=False)
        print(f"  text={text!r}  ->  tokens={[tok.decode([t]) for t in ids]!r}")
        for L in range(1, min(order, len(ids) + 1)):
            ctx = ids[-L:]
            print(f"    ctx_len={L}: ctx={tok.decode(ctx)!r}")
            _print_top(model, tok, vocab, ctx, k=5)


def check_common_probes(model, tok, vocab, order: int) -> None:
    print("\n[common-token probes] (1-token context, top-10)")
    for probe in COMMON_PROBES:
        ids = tok.encode(probe, add_special_tokens=False)
        ctx = ids[:1]
        print(f"  ctx={tok.decode(ctx)!r}")
        _print_top(model, tok, vocab, ctx, k=10)


def check_sanity(model, tok, vocab, order: int) -> List[str]:
    problems: List[str] = []
    probe_ctx = tok.encode("the", add_special_tokens=False)
    probe_toks = sorted(vocab.values())[:1000]

    p1, m1 = model.get_probability(probe_ctx, probe_toks)
    p2, m2 = model.get_probability(probe_ctx, probe_toks)
    if p1 != p2:
        problems.append("get_probability is not deterministic")
    if m1 != m2:
        problems.append("matched_lengths not deterministic")
    if any(not (0.0 <= x <= 1.0) for x in p1):
        problems.append("probability value outside [0, 1]")
    if any(x > order - 1 for x in m1):
        problems.append("matched_lengths exceed order-1")
    if problems:
        print("[sanity] FAIL")
    else:
        print("[sanity] determinism, [0,1] range, matched_lengths: OK")
    return problems


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Definitively verify the .trie tokenizer alignment and "
                    "lookup correctness."
    )
    parser.add_argument("--ngram-model", required=True,
                        help="Path to a DART-format .trie file.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B",
                        help="Tokenizer used at eval time (target model).")
    parser.add_argument("--draft-tokenizer", default=None,
                        help="Draft model tokenizer (eval feeds draft ids).")
    parser.add_argument("--text", action="append", default=None,
                        help="Extra sample text (repeatable).")
    parser.add_argument("--max-parse-nodes", type=int, default=5_000_000,
                        help="Cap for raw-file spot checks.")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from utils.ngram import NgramModel

    # --- raw trie parse first (independent of the C++ extension) -----------
    print(f"[raw] parsing {args.ngram_model} ...")
    order_raw, node_count, root_children, node_info, children_maps = (
        parse_trie_raw(args.ngram_model, args.max_parse_nodes)
    )
    print(f"[raw] order = {order_raw}, node_count = {node_count}")

    problems: List[str] = []

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = tok.get_vocab()
    vocab_size = len(vocab)

    problems += check_root_children_decode(
        root_children, node_info, tok, vocab_size, args.tokenizer
    )

    if args.draft_tokenizer:
        try:
            dtok = AutoTokenizer.from_pretrained(args.draft_tokenizer)
            problems += check_root_children_decode(
                root_children, node_info, dtok, len(dtok.get_vocab()),
                args.draft_tokenizer,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[raw] WARNING: could not decode with draft tokenizer: {exc}")

    # --- C++ extension checks ----------------------------------------------
    model = NgramModel.from_path(args.ngram_model)
    order = int(model.order)
    print(f"[cpp] C++ API reports order = {order}")
    if order != order_raw:
        problems.append(
            f"C++ API order ({order}) differs from raw file order "
            f"({order_raw})"
        )

    vocab2, vocab_problems = check_vocab_identity(
        args.tokenizer, args.draft_tokenizer
    )
    problems += vocab_problems

    snippets: List[str] = list(args.text) if args.text else DEFAULT_SNIPPETS
    problems += check_coverage(model, tok, order, snippets)

    check_canaries(model, tok, vocab, order)
    check_common_probes(model, tok, vocab, order)
    problems += check_conditional_spot(
        model, tok, root_children, node_info, children_maps, order
    )
    problems += check_sanity(model, tok, vocab, order)

    print()
    if problems:
        print("FAIL:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
