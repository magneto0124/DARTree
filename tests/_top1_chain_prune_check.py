"""Check the top1-test whole-tree prune strategy.

Strategy under test (branch top1-test):
  When pruning the whole supertree to top-b, first force-keep the chain of
  the deepest Top-1 node (best score at the maximum depth, plus every
  ancestor), then fill the remaining budget with the best-scoring nodes
  among the rest.

Checks:
  1. The returned set always has exactly ``budget`` nodes.
  2. The deepest-Top-1 chain is always a subset of the kept nodes
     (whenever the budget can hold it; otherwise the plain cut applies).
  3. The kept set is prefix-closed (every kept node's parent is kept/root).
  4. When torch is available, CPU (select_topb_prefix_tree) and tensor
     (select_topb_prefix_tree_tensor) paths agree on the kept node ids.

The CPU function is executed verbatim from eval_dartree.py via AST extraction
so the check also runs on hosts without torch; the tensor path is exercised
only when torch is importable.

Run: python tests/_top1_chain_prune_check.py
"""

import ast
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

torch = None
try:
    import torch  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    torch = None


def load_cpu_function() -> object:
    """Load the real select_topb_prefix_tree source without importing torch."""
    source = open(os.path.join(BASE, "eval_dartree.py"), encoding="utf-8").read()
    parsed = ast.parse(source)
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name == "select_topb_prefix_tree":
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(module, "eval_dartree.py", "exec")
            namespace: dict = {}
            exec(code, namespace)
            return namespace["select_topb_prefix_tree"]
    raise RuntimeError("select_topb_prefix_tree not found in eval_dartree.py")


def load_tensor_function() -> object:
    if torch is None:
        return None
    from eval_dartree import select_topb_prefix_tree_tensor

    return select_topb_prefix_tree_tensor


def build_supertree(
    rng: random.Random,
    depth_limit: int,
    node_count: int,
) -> tuple[list[int], list[float], list[int]]:
    """Build a random prefix-monotone tree in BFS order (root at 0).

    Node i (1-based) has parent parents[i] < i, depth parents-depth + 1 and a
    path score strictly <= its parent's (small random drop), so the
    non-increasing parent-child invariant the pruner relies on holds.
    """
    parents = [-1] * (node_count + 1)
    scores = [0.0] * (node_count + 1)
    depths = [0] * (node_count + 1)
    frontier = [(0, 0)]
    next_id = 1
    while next_id <= node_count:
        if not frontier:
            break
        parent, parent_depth = frontier.pop(0)
        children_here = rng.randint(1, max(1, node_count - next_id + 1))
        for _ in range(children_here):
            if next_id > node_count:
                break
            child_depth = parent_depth + 1
            if child_depth > depth_limit:
                continue
            parents[next_id] = parent
            depths[next_id] = child_depth
            scores[next_id] = scores[parent] - rng.uniform(0.02, 0.9)
            frontier.append((next_id, child_depth))
            next_id += 1
    return parents[:next_id], scores[:next_id], depths[:next_id]


def deepest_top1_chain(parents, scores, depths):
    max_depth = max(depths[1:])
    deepest = [i for i in range(1, len(parents)) if depths[i] == max_depth]
    top1 = min(deepest, key=lambda i: (-scores[i], i))
    chain = []
    cur = top1
    while cur > 0:
        chain.append(cur)
        cur = parents[cur]
    return set(chain), top1


def main() -> None:
    select_topb_prefix_tree = load_cpu_function()
    select_topb_prefix_tree_tensor = load_tensor_function()
    rng = random.Random(20240517)
    failures = 0
    trials = 0
    for depth_limit in (2, 4, 7):
        for node_count in (3, 10, 25, 60):
            for _ in range(40):
                parents, scores, depths = build_supertree(
                    rng, depth_limit, node_count
                )
                n = len(parents) - 1
                if n < 2:
                    continue
                chain, top1 = deepest_top1_chain(parents, scores, depths)
                for budget in range(1, n + 1):
                    trials += 1
                    kept = select_topb_prefix_tree(
                        parents, scores, budget, depths
                    )
                    kept_set = set(kept)
                    ok = len(kept) == budget
                    if len(chain) <= budget:
                        ok = ok and chain.issubset(kept_set)
                    ok = ok and all(
                        p == 0 or p in kept_set
                        for p in (parents[i] for i in kept)
                    )
                    if select_topb_prefix_tree_tensor is not None:
                        path_scores_t = torch.tensor(scores, dtype=torch.float32)
                        depths_t = torch.tensor(depths, dtype=torch.long)
                        parents_t = torch.tensor(parents, dtype=torch.long)
                        kept_t = select_topb_prefix_tree_tensor(
                            path_scores_t, depths_t, budget, 0.0, parents_t
                        )
                        ok = ok and kept_t.tolist() == kept
                    if not ok:
                        failures += 1
                        print(
                            f"FAIL depth_limit={depth_limit} n={n} budget={budget} "
                            f"top1={top1} chain={sorted(chain)} kept={kept} "
                            f"tensor={kept_t.tolist() if select_topb_prefix_tree_tensor is not None else 'n/a'}"
                        )
                        if failures > 5:
                            print("too many failures, aborting")
                            sys.exit(1)
    # Negative depth bonus: same-depth tie at the deepest level, so the top1
    # node is unchanged; CPU receives already-adjusted prune scores and the
    # tensor path applies the bonus internally.
    parents, scores, depths = build_supertree(rng, 5, 30)
    n = len(parents) - 1
    depth_bonus = -0.05
    for budget in (5, 12, n):
        prune_scores = [s + depth_bonus * d for s, d in zip(scores, depths)]
        kept = select_topb_prefix_tree(parents, prune_scores, budget, depths)
        if select_topb_prefix_tree_tensor is not None:
            path_scores_t = torch.tensor(scores, dtype=torch.float32)
            depths_t = torch.tensor(depths, dtype=torch.long)
            parents_t = torch.tensor(parents, dtype=torch.long)
            kept_t = select_topb_prefix_tree_tensor(
                path_scores_t, depths_t, budget, depth_bonus, parents_t
            )
            trials += 1
            if kept_t.tolist() != kept:
                failures += 1
                print(f"FAIL depth-bonus budget={budget} cpu={kept} tensor={kept_t.tolist()}")
        else:
            kept_set = set(kept)
            chain, _ = deepest_top1_chain(parents, prune_scores, depths)
            trials += 1
            if not (
                len(kept) == budget
                and (len(chain) > budget or chain.issubset(kept_set))
            ):
                failures += 1
                print(f"FAIL depth-bonus budget={budget} kept={kept}")

    if failures:
        print(f"\n{trials} trials, {failures} FAILURES")
        sys.exit(1)
    tensor_note = (
        "CPU==tensor"
        if select_topb_prefix_tree_tensor is not None
        else "tensor path skipped (torch unavailable)"
    )
    print(f"\nOK: {trials} trials passed (chain preserved, prefix-closed, {tensor_note})")


if __name__ == "__main__":
    main()
