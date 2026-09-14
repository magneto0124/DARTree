#!/usr/bin/env python3
"""Sanity-check that a .trie n-gram model aligns with the tokenizers used at
eval time, i.e. that probability lookups are NOT silently wrong.

Run on the evaluation machine (needs transformers + the C++ extension):

    python utils/verify_ngram.py \
        --ngram-model /path/to/small.trie \
        --tokenizer Qwen/Qwen3-4B \
        --draft-tokenizer Huang2020/Qwen3-4B-Domino-b16

Checks performed:

1. trie order (read from the .trie header).
2. tokenizer id-space identity between --tokenizer and --draft-tokenizer
   (vocab size + per-token id spot check) -- the eval queries the trie with
   token ids produced by the draft model, so both sides must agree.
3. coverage: tokenize real text (math / code / prose) and, for every
   consecutive n-gram window (ctx length 1 .. order-1), query the trie and
   count how often the next token has p > 0.  A correct Qwen3 tokenizer / trie
   pairing shows high coverage at ctx length 1; near-zero coverage means the
   id spaces do not match.
4. top-5 continuations per context, decoded back to text, for a human
   eyeball check.
5. determinism and value-range sanity.

Exit code 0 = all checks passed; 1 = a hard failure was found.
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

CANARY_CONTEXTS: List[str] = [
    "The capital of France is",
    "the answer is",
]


def check_vocab_identity(
    tokenizer_name: str,
    draft_tokenizer_name: Optional[str],
) -> Tuple[Optional[dict], List[str]]:
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
                mism = sum(
                    1 for t in sample if vocab[t] != dvocab.get(t)
                )
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


def check_top_continuations(model, tok, vocab, order: int) -> None:
    """Print the trie's most likely continuations for human inspection."""
    print("\n[top-5 continuations] (human eyeball check)")
    vocab_ids = sorted(vocab.values())
    for text in CANARY_CONTEXTS:
        ids = tok.encode(text, add_special_tokens=False)
        ctx = ids[-(order - 1):]
        probs, _matched = model.get_probability(ctx, vocab_ids)
        ranked = sorted(zip(probs, vocab_ids), reverse=True)[:5]
        print(f"  ctx={text!r}")
        for p, tid in ranked:
            print(f"    p={p:.4f}  {tok.decode([tid])!r}")


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

    check_top_continuations(model, tok, vocab, order)
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
