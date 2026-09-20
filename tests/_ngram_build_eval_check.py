"""Standalone check for ngram_build.py's --eval-output mode (no torch).

Static checks:
  * --eval-output is a defined, optional argparse flag;
  * main() folds collect_eval_texts(...) into the training items;
  * the eval path is inference-free: no dartree_generate / spec_generate /
    AutoModel anywhere in the script (inputs come from the dataset, outputs
    from the saved eval JSON);
  * collect_eval_texts reads the dataset (load_and_process_dataset) and
    builds conversations via build_conversation_text.

Dynamic checks (functions extracted via AST):
  * group_assistant_by_sample: maps sample_index -> {turn_index: text},
    drops rows without usable dartree.text;
  * build_conversation_text: interleaves user turns with saved responses;
  * collect_eval_texts: full end-to-end with a faked dataset -- reads the
    eval JSON, reconstructs the eval dataset order from summary.config,
    interleaves inputs+outputs, skips missing samples.
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
from types import SimpleNamespace

SRC = open("utils/ngram_build.py", encoding="utf-8").read()
tree = ast.parse(SRC)


def fn(name: str) -> ast.FunctionDef:
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Attribute)
    and n.func.attr == "add_argument"
    and any(
        isinstance(a, ast.Constant) and a.value == "--eval-output"
        for a in n.args
    )
    for n in ast.walk(tree)
), "ngram_build.py must define --eval-output"
assert "default=None" in SRC, "--eval-output must be optional"
print("OK  --eval-output defined as an optional flag")

# main() must fold eval texts into the training items
main_fn = fn("main")
assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Name)
    and n.func.id == "collect_eval_texts"
    for n in ast.walk(main_fn)
), "main() must call collect_eval_texts"
print("OK  main() folds eval texts into the training items")

# inference-free
for forbidden in ("dartree_generate", "spec_generate", "AutoModel"):
    assert forbidden not in SRC, (
        f"ngram_build.py must stay inference-free (found {forbidden!r})"
    )
print("OK  eval path is inference-free (dataset inputs + saved outputs)")

# collect_eval_texts must read the dataset and build conversations
collect_fn = fn("collect_eval_texts")
assert "load_and_process_dataset" in SRC, "must read the dataset"
assert any(
    isinstance(n, ast.Call)
    and isinstance(n.func, ast.Name)
    and n.func.id == "build_conversation_text"
    for n in ast.walk(collect_fn)
), "collect_eval_texts must build conversations"
print("OK  collect_eval_texts reads the dataset and builds conversations")

# ---------------------------------------------------------------------------
# Dynamic checks
# ---------------------------------------------------------------------------

ns: dict = {
    "Dict": __import__("typing").Dict,
    "List": __import__("typing").List,
    "Optional": __import__("typing").Optional,
    "Sequence": __import__("typing").Sequence,
    "argparse": __import__("argparse"),
}
for name in ("group_assistant_by_sample", "build_conversation_text"):
    exec(
        compile(ast.Module(body=[fn(name)], type_ignores=[]), f"<{name}>", "exec"),
        ns,
    )

group_assistant_by_sample = ns["group_assistant_by_sample"]
build_conversation_text = ns["build_conversation_text"]

# group_assistant_by_sample: matches sample_index/turn_index, drops unusable
rows = [
    {"sample_index": 1, "turn_index": 0, "dartree": {"text": "A0"}},
    {"sample_index": 1, "turn_index": 1, "dartree": {"text": "A1"}},
    {"sample_index": 2, "turn_index": 0, "dartree": {"text": "B"}},
    {"sample_index": 3, "turn_index": 0},                  # no response
    {"sample_index": 4, "turn_index": 0, "dartree": {}},   # no text
]
got = group_assistant_by_sample(rows)
assert got == {1: {0: "A0", 1: "A1"}, 2: {0: "B"}}, got
print("OK  group_assistant_by_sample matches rows and drops unusable ones")

# build_conversation_text: interleave, extra assistant turns, drop empties
assert (
    build_conversation_text(["U0", "U1"], {0: "A0", 1: "A1"})
    == "U0\nA0\nU1\nA1"
)
assert (
    build_conversation_text(["U0", "U1"], {0: "A0"})
    == "U0\nA0\nU1"
)
assert (
    build_conversation_text(["U0"], {0: "A0", 1: "A1"})
    == "U0\nA0\nA1"
)
assert build_conversation_text(["U0"], {}) == "U0"
assert build_conversation_text([], {0: "A0"}) == "A0"
assert build_conversation_text([], {}) == ""
print("OK  build_conversation_text interleaves turns and responses")

# collect_eval_texts end-to-end with a faked dataset
class _FakeDataset:
    """Plain list-like dataset; shuffle/select are only used when
    summary.config.max_samples is not None (here it is)."""

    def __init__(self, samples) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]

    def shuffle(self, seed: int) -> "_FakeDataset":
        return self  # seed irrelevant for the check

    def select(self, indices) -> "_FakeDataset":
        return _FakeDataset([self._samples[i] for i in indices])


fake_dataset = _FakeDataset(
    [
        {"turns": ["Q0"]},
        {"turns": ["Q1a", "Q1b"]},
        {"turns": ["Q2"]},
        {"turns": ["Q3"]},
    ]
)


def fake_load_and_process_dataset(name: str):
    assert name == "gsm8k"
    return fake_dataset


collect_ns: dict = {
    "json": json,
    "print": print,
    "argparse": __import__("argparse"),
    "group_assistant_by_sample": group_assistant_by_sample,
    "build_conversation_text": build_conversation_text,
    "load_and_process_dataset": fake_load_and_process_dataset,
}
# strip the internal `from utils.data import load_and_process_dataset` import
collect_fn.body = [
    n for n in collect_fn.body
    if not isinstance(n, (ast.Import, ast.ImportFrom))
]
exec(
    compile(ast.Module(body=[collect_fn], type_ignores=[]), "<collect>", "exec"),
    collect_ns,
)
collect_eval_texts = collect_ns["collect_eval_texts"]

with tempfile.TemporaryDirectory() as tmp:
    eval_json = os.path.join(tmp, "eval.json")
    eval_data = {
        "summary": {
            "config": {
                "dataset": "gsm8k",
                "dataset_shuffle_seed": 0,
                "max_samples": None,
            }
        },
        "rows": [
            {"sample_index": 0, "turn_index": 0, "dartree": {"text": "R0"}},
            {"sample_index": 1, "turn_index": 0, "dartree": {"text": "R1a"}},
            {"sample_index": 1, "turn_index": 1, "dartree": {"text": "R1b"}},
            {"sample_index": 2, "turn_index": 0, "dartree": {"text": "R2"}},
            {"sample_index": 99, "turn_index": 0, "dartree": {"text": "R99"}},
            {"sample_index": 3, "turn_index": 0},  # no usable output
        ],
    }
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(eval_data, f)

    args = SimpleNamespace(eval_output=eval_json)
    texts = collect_eval_texts(args)
    assert texts == [
        "Q0\nR0",
        "Q1a\nR1a\nQ1b\nR1b",
        "Q2\nR2",
    ], texts
    print("OK  collect_eval_texts builds input+output conversations end-to-end")

print("\nALL NGRAM-BUILD EVAL CHECKS PASSED")
