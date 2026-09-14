#!/usr/bin/env python3
"""Sanity-check that a .trie n-gram model aligns with the tokenizers used at
eval time, i.e. that probability lookups are NOT silently wrong.

Run on the evaluation machine (needs transformers + the C++ extension):

    python utils/verify_ngram.py \
        --ngram-model /path/to/small.trie \
        --tokenizer Qwen/Qwen3-4B \
        --draft-tokenizer Huang2020/Qwen3-4B-Domino-b16

Checks:

1. trie order (from the .trie header).
2. tokenizer id-space identity between --tokenizer and --draft-tokenizer
   (vocab size + per-token id spot check).  The eval feeds the trie with token
   ids produced by the DRAFT model's logits, so both sides must agree.
3. coverage: tokenize real text (math / code / prose) and, for every
   consecutive n-gram window (ctx length 1 .. order-1), count how often the
   next token has p > 0.  High 1-token coverage => same id space; near-zero =>
   mismatch.
4. canary probes: top-N continuations (with p values, and the actual queried
   context decoded) for contexts that certainly occur in the training corpus,
   at both 1-token and (order-1)-token context lengths.
5. determinism and value-range sanity.

Exit code 0 = all checks passed; 1 = hard failure.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Tuple

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

# Contexts whose (order-1)-token tails are guaranteed common in a general
# web corpus, so a correctly-aligned trie MUST return real continuations.
CANARY_CONTEXTS: List[str] = [
    "the answer is",
    "I think",
    "in order to",
    "one of the most",
]

COMMON_PROBES: List[str] = [
    " the",
    " is",
    " answer",
    " of",
]


def check_vocab_identity(
    tokenizer_name: str,
    draft_tokenizer_name: Optional[str],
) -> Tuple[dict, List[str]]:
    """Compare the id spaces of the eval tokenizers; return (vocab, problems)."""
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
    """Report non-OOV coverage for each context length 1..order-1."""
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
    """Query the trie over the full vocab and print the top-k continuations."""
    probs, matched = model.get_probability(ctx, sorted(vocab.values()))
    ranked = sorted(zip(probs, matched, sorted(vocab.values())), reverse=True)
    if ranked[0][0] <= 0.0:
        print("      -> NO MATCH (all p = 0): the queried context does not "
              "exist in this trie's id space.")
        return
    for p, m, tid in ranked[:k]:
        print(f"      p={p:.4f} (matched_len={m}) {tok.decode([tid])!r}")


def check_canaries(model, tok, vocab, order: int) -> None:
    """Query guaranteed-common contexts at both 1-token and full-window
    lengths, so we can see whether the trie 'understands' real text."""
    print("\n[canary probes] (query the trie over the full vocab)")
    vocab_ids = sorted(vocab.values())
    for text in CANARY_CONTEXTS:
        ids = tok.encode(text, add_special_tokens=False)
        print(f"  text={text!r}  ->  tokens={[tok.decode([t]) for t in ids]!r}")
        for L in range(1, min(order, len(ids) + 1)):
            ctx = ids[-L:]
            print(f"    ctx_len={L}: ctx={tok.decode(ctx)!r}")
            _print_top(model, tok, vocab, ctx, k=5)


def check_common_probes(model, tok, vocab, order: int) -> None:
    """1-token probes on the most common English tokens."""
    print("\n[common-token probes] (1-token context, top-10)")
    for probe in COMMON_PROBES:
        ids = tok.encode(probe, add_special_tokens=False)
        ctx = ids[:1]
        print(f"  ctx={tok.decode(ctx)!r}")
        _print_top(model, tok, vocab, ctx, k=10)


def check_sanity(model, tok, vocab, order: int) -> List[str]:
    """Determinism and value-range checks."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a .trie n-gram model aligns with the eval "
                    "tokenizer (probability lookups are not silently wrong)."
    )
    parser.add_argument("--ngram-model", required=True,
                        help="Path to a DART-format .trie file.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B",
                        help="Tokenizer used at eval time (target model).")
    parser.add_argument("--draft-tokenizer", default=None,
                        help="Draft model tokenizer to compare id spaces "
                             "against (e.g. Huang2020/Qwen3-4B-Domino-b16).")
    parser.add_argument("--text", action="append", default=None,
                        help="Extra sample text (repeatable).")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from utils.ngram import NgramModel

    model = NgramModel.from_path(args.ngram_model)
    order = int(model.order)
    print(f"[trie] order = {order}")

    vocab, problems = check_vocab_identity(
        args.tokenizer, args.draft_tokenizer
    )

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    snippets: List[str] = list(args.text) if args.text else DEFAULT_SNIPPETS
    problems += check_coverage(model, tok, order, snippets)

    check_canaries(model, tok, vocab, order)
    check_common_probes(model, tok, vocab, order)
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
