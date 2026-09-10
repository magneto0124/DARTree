#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache


from utils import (
    DFlashDraftModel,
    DominoCorrectionScorer,
    DraftCorrectionGraphRunner,
    cuda_time,
    load_and_process_dataset,
    logits_entropy,
    resolve_graft_retain,
    sample,
)
from utils.device_backend import (
    accelerator_available,
    can_use_cuda_graphs,
    device_type,
    is_accelerator_device,
    seed_all,
    set_device,
    synchronize,
)


REPO_ROOT = Path(__file__).resolve().parent


STAGE_NAMES = ("draft", "tree_build", "tree_setup", "verify", "commit")


class DARTreeScoreSelectGraph:
    """CUDA graph for candidate scoring and frontier selection."""

    def __init__(
        self,
        scorer: DominoCorrectionScorer,
        *,
        device: torch.device,
        max_batch: int = 16,
        candidate_count: int = 64,
        select_candidate_count: int | None = None,
        max_select: int = 8,
        dtype: torch.dtype = torch.bfloat16,
        include_gru: bool = False,
        warm_pairs: list[tuple[int, int]] | None = None,
    ) -> None:
        self.scorer = scorer
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.candidate_count = max(1, int(candidate_count))
        self.select_candidate_count = max(
            1,
            min(
                int(self.candidate_count),
                (
                    int(select_candidate_count)
                    if select_candidate_count is not None
                    else int(self.candidate_count)
                ),
            ),
        )
        self.max_select = max(1, int(max_select))
        self.dtype = dtype
        self.include_gru = bool(include_gru)
        self.graphs: dict[tuple[int, int, int, torch.dtype], dict[str, Any]] = {}
        # CUDA-graph capture is NVIDIA-only; on an Ascend NPU the caller falls
        # back to the eager score/select path (run() returns None).
        self.enabled = can_use_cuda_graphs(device)
        if self.enabled and warm_pairs is not None:
            for batch_size, selected_count in sorted(set((int(b), int(s)) for b, s in warm_pairs)):
                if (
                    batch_size > 0
                    and batch_size <= self.max_batch
                    and selected_count > 0
                    and selected_count <= self.max_select
                    and selected_count <= batch_size * self.select_candidate_count
                ):
                    self._capture(
                        batch_size=batch_size,
                        selected_count=selected_count,
                        select_candidate_count=self.select_candidate_count,
                        dtype=dtype,
                    )
        elif self.enabled:
            for batch_size in range(1, self.max_batch + 1):
                max_for_batch = min(self.max_select, batch_size * self.select_candidate_count)
                for selected_count in range(1, max_for_batch + 1):
                    self._capture(
                        batch_size=batch_size,
                        selected_count=selected_count,
                        select_candidate_count=self.select_candidate_count,
                        dtype=dtype,
                    )

    def _capture(
        self,
        *,
        batch_size: int,
        selected_count: int,
        select_candidate_count: int,
        dtype: torch.dtype,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        batch_size = int(batch_size)
        selected_count = int(selected_count)
        select_candidate_count = max(1, min(int(select_candidate_count), int(self.candidate_count)))
        if (
            batch_size <= 0
            or batch_size > self.max_batch
            or selected_count <= 0
            or selected_count > self.max_select
            or selected_count > batch_size * select_candidate_count
        ):
            return None
        key = (batch_size, selected_count, select_candidate_count, dtype)
        cached = self.graphs.get(key)
        if cached is not None:
            return cached

        hidden_dim = int(self.scorer.w_s.shape[1])
        mid_dim = int(self.scorer.w_s.shape[0])
        h_static = torch.zeros((batch_size, hidden_dim), dtype=dtype, device=self.device)
        z_static = torch.zeros((batch_size, mid_dim), dtype=dtype, device=self.device)
        candidate_weight = torch.zeros(
            (self.candidate_count, mid_dim),
            dtype=dtype,
            device=self.device,
        )
        candidate_base = torch.zeros(
            (self.candidate_count,),
            dtype=torch.float32,
            device=self.device,
        )
        candidate_bias = torch.zeros((self.candidate_count,), dtype=dtype, device=self.device)
        candidate_ids = torch.arange(self.candidate_count, dtype=torch.long, device=self.device)
        parent_scores = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
        parent_indices = torch.arange(batch_size, dtype=torch.long, device=self.device)

        def graph_body() -> tuple[torch.Tensor, ...]:
            s_proj = F.linear(h_static, self.scorer.w_s, None)
            mid = F.silu((z_static + s_proj).unsqueeze(1)).squeeze(1)
            logits = (
                candidate_base.view(1, -1).to(mid.dtype)
                + candidate_bias.view(1, -1)
                + (mid @ candidate_weight.t())
            ).float()
            log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
            logits_for_select = logits[:, :select_candidate_count]
            path_scores = parent_scores.unsqueeze(1) + logits_for_select - log_z
            _selected_prios, selected_flat = torch.topk(
                path_scores.reshape(-1),
                k=selected_count,
                dim=0,
            )
            selected_parent_pos = torch.div(
                selected_flat,
                select_candidate_count,
                rounding_mode="floor",
            )
            selected_rank = selected_flat.remainder(select_candidate_count)
            selected_tokens = candidate_ids.index_select(0, selected_rank).contiguous()
            selected_path_scores = path_scores[selected_parent_pos, selected_rank].contiguous()
            selected_parent_indices = parent_indices.index_select(0, selected_parent_pos)
            outputs: tuple[torch.Tensor, ...] = (
                selected_parent_pos,
                selected_rank,
                selected_tokens,
                selected_path_scores,
                selected_parent_indices,
            )
            if self.include_gru:
                parent_h0 = h_static.index_select(0, selected_parent_pos)
                gi = self.scorer._gru_input_proj_table.index_select(0, selected_tokens)
                gh = F.linear(parent_h0, self.scorer.gru_w_hh, self.scorer.gru_b_hh)
                g = int(self.scorer.gru_hidden_dim)
                i_r, i_z, i_n = gi[:, :g], gi[:, g : 2 * g], gi[:, 2 * g :]
                h_r, h_z, h_n = gh[:, :g], gh[:, g : 2 * g], gh[:, 2 * g :]
                r = torch.sigmoid(i_r + h_r)
                z = torch.sigmoid(i_z + h_z)
                n = torch.tanh(i_n + r * h_n)
                child_hidden = (1.0 - z) * n + z * parent_h0
                outputs = outputs + (child_hidden,)
            return outputs

        for _ in range(3):
            graph_body()
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            outputs = graph_body()
        entry = {
            "graph": graph,
            "h": h_static,
            "z": z_static,
            "candidate_weight": candidate_weight,
            "candidate_base": candidate_base,
            "candidate_bias": candidate_bias,
            "candidate_ids": candidate_ids,
            "parent_scores": parent_scores,
            "parent_indices": parent_indices,
            "outputs": outputs,
        }
        self.graphs[key] = entry
        return entry

    def run(
        self,
        *,
        z_proj: torch.Tensor,
        h_state: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_base_vals: torch.Tensor,
        candidate_weight: torch.Tensor,
        candidate_bias: torch.Tensor | None,
        parent_scores: torch.Tensor,
        parent_indices: torch.Tensor,
        selected_count: int,
        select_candidate_count: int | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        if not self.enabled:
            return None
        batch_size = int(h_state.shape[0])
        if int(candidate_ids.numel()) != self.candidate_count:
            return None
        select_count = (
            int(self.select_candidate_count)
            if select_candidate_count is None
            else max(1, min(int(select_candidate_count), int(self.candidate_count)))
        )
        entry = self._capture(
            batch_size=batch_size,
            selected_count=int(selected_count),
            select_candidate_count=int(select_count),
            dtype=h_state.dtype,
        )
        if entry is None:
            return None
        entry["h"].copy_(h_state)
        entry["z"].copy_(z_proj)
        entry["candidate_weight"].copy_(candidate_weight)
        entry["candidate_base"].copy_(candidate_base_vals.float())
        entry["candidate_ids"].copy_(candidate_ids.long())
        entry["parent_scores"].copy_(parent_scores.float())
        entry["parent_indices"].copy_(parent_indices.long())
        if candidate_bias is not None:
            entry["candidate_bias"].copy_(candidate_bias)
        else:
            entry["candidate_bias"].zero_()
        entry["graph"].replay()
        return entry["outputs"]


def stage_dict() -> dict[str, float]:
    return {name: 0.0 for name in STAGE_NAMES}


def add_elapsed(
    times: dict[str, float] | None,
    name: str,
    start: float,
    device: torch.device,
) -> None:
    if times is not None:
        times[name] = float(times.get(name, 0.0) + cuda_time(device) - start)


def detail_start(times: dict[str, float] | None, device: torch.device) -> float:
    return cuda_time(device) if times is not None else 0.0


class SelectedHiddenCollector:
    def __init__(self, target: AutoModelForCausalLM, layer_ids: list[int]):
        self.target = target
        self.layer_ids = [int(layer_id) for layer_id in layer_ids]
        self.handles: list[Any] = []
        self.states: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "SelectedHiddenCollector":
        layers = self.target.model.layers
        for layer_id in self.layer_ids:
            handle = layers[layer_id].register_forward_hook(self._make_hook(layer_id))
            self.handles.append(handle)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.states.clear()

    def _make_hook(self, layer_id: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            self.states[int(layer_id)] = output[0] if isinstance(output, tuple) else output

        return hook

    def clear(self) -> None:
        self.states.clear()

    def cat(self) -> torch.Tensor:
        return torch.cat([self.states[layer_id] for layer_id in self.layer_ids], dim=-1)

    def index_select_cat(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        selected = [
            self.states[layer_id].index_select(dim, index)
            for layer_id in self.layer_ids
        ]
        if len(selected) == 1:
            return selected[0]
        return torch.cat(selected, dim=-1)


def normalize_draft_config(config: Any) -> Any:
    dflash_config = dict(getattr(config, "dflash_config", {}) or {})
    if dflash_config.get("projector_type") == "causal_v5":
        dflash_config["projector_type"] = "domino"
    if "emb_dim" not in dflash_config and hasattr(config, "emb_dim"):
        dflash_config["emb_dim"] = int(config.emb_dim)
    if "gru_hidden_dim" not in dflash_config:
        if hasattr(config, "gru_hidden_dim"):
            dflash_config["gru_hidden_dim"] = int(config.gru_hidden_dim)
        elif "emb_dim" in dflash_config:
            dflash_config["gru_hidden_dim"] = int(dflash_config["emb_dim"])
    config.dflash_config = dflash_config
    return config


def supertree_width_schedule(
    depth_limit: int,
    candidate_count: int,
    width: int,
) -> list[int]:
    width = max(1, min(int(width), int(candidate_count)))
    return [width] * int(depth_limit)


def select_topb_prefix_tree(
    parents: list[int],
    scores: list[float],
    budget: int,
) -> list[int]:
    """Select global Top-B nodes when scores are prefix monotone."""
    node_count = len(parents) - 1
    if node_count < 0 or len(scores) != len(parents):
        raise ValueError("parents and scores must include the root and have equal length")
    if int(budget) < 0 or int(budget) > int(node_count):
        raise ValueError(f"budget {budget} is outside [0, {node_count}]")
    for node_index in range(1, node_count + 1):
        parent_index = int(parents[node_index])
        if parent_index < 0 or parent_index >= node_index:
            raise ValueError(
                f"invalid parent {parent_index} for node {node_index}"
            )
        if float(scores[node_index]) > float(scores[parent_index]):
            raise ValueError(
                "Top-B prefix selection requires non-increasing parent-child scores: "
                f"node {node_index} has {scores[node_index]} > parent "
                f"{parent_index} with {scores[parent_index]}"
            )

    selected = sorted(
        range(1, node_count + 1),
        key=lambda node_index: (-float(scores[node_index]), int(node_index)),
    )[: int(budget)]
    selected_set = {0, *selected}
    if any(int(parents[node_index]) not in selected_set for node_index in selected):
        raise RuntimeError("prefix-monotone Top-B selection produced a non-prefix-closed tree")
    return sorted(selected)


def select_topb_prefix_tree_tensor(
    path_scores: torch.Tensor,
    depths: torch.Tensor,
    budget: int,
    depth_bonus: float,
) -> torch.Tensor:
    """Select global Top-B with one device topk and restore topological order."""
    node_count = int(depths.numel())
    if int(path_scores.numel()) < node_count + 1:
        raise ValueError("path_scores must include the root and every candidate node")
    if int(budget) < 0 or int(budget) > node_count:
        raise ValueError(f"budget {budget} is outside [0, {node_count}]")
    if int(budget) == 0:
        return torch.empty((0,), dtype=torch.long, device=depths.device)

    candidate_scores = (
        path_scores[1 : node_count + 1].float()
        + float(depth_bonus) * depths[:node_count].float()
    )
    selected = torch.topk(
        candidate_scores,
        k=int(budget),
        largest=True,
        sorted=False,
    ).indices
    return torch.sort(selected.add_(1)).values


def planned_score_select_pairs(
    *,
    budget: int,
    depth_limit: int,
    candidate_count: int,
    per_layer_widths: list[int] | None,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    depth_limit = max(1, int(depth_limit))
    candidate_count = max(1, int(candidate_count))
    budget = max(1, int(budget))
    node_count = 0
    frontier_len = 1
    for child_depth in range(1, depth_limit + 1):
        if frontier_len <= 0 or node_count >= budget:
            break
        remaining = budget - node_count
        remaining_depths = max(1, depth_limit - child_depth + 1)
        quota = (
            int(per_layer_widths[child_depth - 1])
            if per_layer_widths is not None
            else int(np.ceil(remaining / remaining_depths))
        )
        quota = max(1, min(remaining, quota))
        selected_count = min(quota, frontier_len * candidate_count)
        if selected_count <= 0:
            break
        pairs.add((frontier_len, selected_count))
        frontier_len = selected_count
        node_count += selected_count
    return pairs


def prepare_tree_attention_inputs(
    *,
    root_token_id: torch.Tensor | int,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[
            0,
            0,
            :previous_tree_length,
            previous_tree_start : previous_tree_start + previous_tree_length,
        ] = 0

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[
        0,
        0,
        :current_length,
        past_length : past_length + current_length,
    ]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = (
        int(root_token_id)
        if not torch.is_tensor(root_token_id)
        else root_token_id
    )
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_verified_tree(
    child_maps: list[dict[int, int]],
    posterior: torch.Tensor,
) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])
    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])
    return accepted_indices, next_token


def _compact_appended_window(
    cache_tensor: torch.Tensor,
    past_length: int,
    keep_current_indices: torch.Tensor,
) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return
    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return
    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(
        -2,
        keep_current_indices,
    )
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(
    past_key_values: DynamicCache,
    past_length: int,
    keep_current_indices: list[int],
) -> None:
    if not keep_current_indices:
        past_key_values.crop(past_length)
        return
    if all(int(idx) == offset for offset, idx in enumerate(keep_current_indices)):
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(
                keep_current_indices,
                dtype=torch.long,
                device=device,
            )
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for tree cache compaction.")


def build_visibility(parents: list[int]) -> torch.Tensor:
    current_length = len(parents)
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    return torch.from_numpy(visibility_np)


@torch.inference_mode()
def build_dartree_supertree(
    *,
    root_token_id: torch.Tensor,
    parallel_hiddens: torch.Tensor,
    base_logits: torch.Tensor,
    budget: int,
    expansion_k: int,
    prefix_len: int,
    supertree_width: int,
    depth_bonus: float,
    correction_scorer: DominoCorrectionScorer,
    z_parts: torch.Tensor,
    candidate_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    score_select_graph: DARTreeScoreSelectGraph,
    tree_buffers: dict[str, torch.Tensor],
    pruned: bool,
    retain_budget: int | None = None,
    detail_times: dict[str, float] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    list[dict[int, int]],
    torch.Tensor,
    dict[str, float],
]:
    """Build a fixed-width or globally pruned DARTree."""
    device = parallel_hiddens.device
    depth_limit = int(base_logits.shape[1])
    # Graft: the pruning target may be smaller than `budget` (`retain_budget`),
    # leaving ``budget - retain_budget`` slots for the retrieval subtree.  When
    # `retain_budget` is None the function keeps its original behaviour (prune to
    # `budget`), so `fixed`/`pruned` variants are unaffected.
    prune_target = int(budget) if retain_budget is None else int(retain_budget)
    if prune_target < 0 or prune_target > int(budget):
        raise ValueError(
            f"retain_budget {prune_target} must be within [0, budget={budget}]"
        )
    if budget <= 0 or depth_limit <= 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            torch.ones((1, 1), dtype=torch.bool),
            {"node_count": 0.0, "tree_height": 0.0},
        )

    expansion_count = max(
        1,
        min(int(expansion_k), int(base_logits.shape[-1]), int(budget)),
    )

    def reusable_buffer(
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = tree_buffers.get(name)
        if (
            tensor is None
            or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            tensor = torch.empty(shape, dtype=dtype, device=device)
            tree_buffers[name] = tensor
        return tensor

    t_detail = detail_start(detail_times, device)
    root_h0 = reusable_buffer(
        "root_zero_hidden",
        (1, int(correction_scorer.gru_hidden_dim)),
        correction_scorer.gru_w_hh.dtype,
    )
    root_h0.zero_()
    root_hidden = correction_scorer.update_hidden(
        root_token_id.view(-1), root_h0
    ).view(1, 1, -1)
    add_elapsed(detail_times, "tree_setup_root_gru", t_detail, device)

    t_root_probability = detail_start(detail_times, device)
    slot = 0
    if slot >= prefix_len:
        candidate_ids, candidate_base_vals, candidate_weights, candidate_biases = (
            candidate_tables
        )
        z_i = z_parts[:, slot : slot + 1, :][:, 0, :]
        h_state = root_hidden.squeeze(0)
        bias_i = (
            None
            if candidate_biases is None
            else candidate_biases[slot]
        )
        top_vals, _top_ids, log_z = (
            correction_scorer.candidate_topk_from_precomputed(
                z_i,
                h_state,
                candidate_ids[slot],
                candidate_base_vals[slot],
                candidate_weights[slot],
                bias_i,
                2,
            )
        )
        top_vals = top_vals[0]
        log_z = log_z[0, 0]
    else:
        logits = base_logits[:, slot : slot + 1, :]
        logits_1d = logits[0, 0].float()
        top_vals, _top_ids = torch.topk(
            logits_1d, k=2, dim=-1
        )
        log_z = torch.logsumexp(logits_1d, dim=-1)
    top1_logprob = float((top_vals[0] - log_z).item())
    root_top1_probability = float(np.exp(top1_logprob))
    add_elapsed(
        detail_times,
        "tree_root_probability",
        t_root_probability,
        device,
    )

    def update_child_hidden(
        token_ids: torch.Tensor,
        parent_h0: torch.Tensor,
    ) -> torch.Tensor:
        return correction_scorer.update_hidden(
            token_ids.view(-1),
            parent_h0.squeeze(0),
        ).unsqueeze(0)

    supertree_widths = (
        supertree_width_schedule(
            depth_limit, expansion_count, supertree_width
        )
        if pruned
        else []
    )
    max_supertree_width = max(supertree_widths, default=1)
    expanded_layers = 0
    scored_parent_count = 0
    max_frontier_width = 0
    fused_score_select_layers = 0
    fused_gru_layers = 0
    fallback_gru_layers = 0
    graph_miss_batch_layers = 0
    graph_miss_select_layers = 0
    graph_miss_other_layers = 0
    supertree_node_count = 0
    supertree_budget = (
        sum(int(width) for width in supertree_widths)
        if pruned
        else int(budget)
    )
    max_nodes = int(supertree_budget) + 1
    hidden_size = int(root_hidden.shape[-1])
    hidden_states = reusable_buffer(
        "hidden_states", (max_nodes, hidden_size), root_hidden.dtype
    )
    path_scores_t = reusable_buffer(
        "path_scores", (max_nodes,), torch.float32
    )
    parents_t = reusable_buffer("parents", (max_nodes,), torch.long)
    tokens_t = reusable_buffer(
        "tokens", (int(supertree_budget),), torch.long
    )
    depths_t = reusable_buffer(
        "depths", (int(supertree_budget),), torch.long
    )
    node_index_range = tree_buffers.get("node_index_range")
    if (
        node_index_range is None
        or tuple(node_index_range.shape) != (max_nodes,)
        or node_index_range.dtype != torch.long
        or node_index_range.device != device
    ):
        node_index_range = torch.arange(
            max_nodes, dtype=torch.long, device=device
        )
        tree_buffers["node_index_range"] = node_index_range

    hidden_states[0].copy_(root_hidden.view(-1))
    parents_t[:1] = torch.tensor([-1], dtype=torch.long, device=device)
    path_scores_t[:1] = torch.tensor(
        [0.0], dtype=torch.float32, device=device
    )

    node_count = 0
    frontier_t = torch.tensor([0], dtype=torch.long, device=device)
    frontier_len = 1
    for child_depth in range(1, depth_limit + 1):
        if frontier_len <= 0 or node_count >= int(supertree_budget):
            break

        remaining = int(supertree_budget) - int(node_count)
        remaining_depths = max(1, depth_limit - child_depth + 1)
        if pruned:
            quota = int(supertree_widths[child_depth - 1])
        else:
            quota = int(np.ceil(remaining / remaining_depths))
        quota = max(1, min(remaining, quota))

        parent_indices = frontier_t[:frontier_len]
        parent_hidden_2d = hidden_states.index_select(0, parent_indices)
        parent_count = int(frontier_len)
        expanded_layers += 1
        scored_parent_count += int(parent_count)
        max_frontier_width = max(
            int(max_frontier_width), int(parent_count)
        )
        slot = child_depth - 1
        parent_scores_1d = path_scores_t.index_select(0, parent_indices)
        selected_count = min(
            int(quota), int(parent_count) * int(expansion_count)
        )
        if selected_count <= 0:
            break
        fused_selection = None
        fused_child_hidden_2d = None

        t_score = detail_start(detail_times, device)
        if slot < prefix_len:
            candidate_ids, candidate_base_vals, candidate_weights, _ = (
                candidate_tables
            )
            if int(expansion_count) <= int(candidate_ids.shape[-1]):
                z_i = z_parts[:, slot : slot + 1, :].expand(
                    parent_count, -1, -1
                )[:, 0, :]
                zero_candidate_weight = reusable_buffer(
                    "zero_candidate_weight",
                    tuple(candidate_weights[slot].shape),
                    candidate_weights.dtype,
                )
                zero_candidate_weight.zero_()
                zero_candidate_bias = reusable_buffer(
                    "zero_candidate_bias",
                    tuple(candidate_base_vals[slot].shape),
                    candidate_weights.dtype,
                )
                zero_candidate_bias.zero_()
                fused_selection = score_select_graph.run(
                    z_proj=z_i,
                    h_state=parent_hidden_2d,
                    candidate_ids=candidate_ids[slot],
                    candidate_base_vals=candidate_base_vals[slot],
                    candidate_weight=zero_candidate_weight,
                    candidate_bias=zero_candidate_bias,
                    parent_scores=parent_scores_1d,
                    parent_indices=parent_indices,
                    selected_count=selected_count,
                    select_candidate_count=int(expansion_count),
                )
                if fused_selection is not None:
                    fused_score_select_layers += 1
                elif parent_count > int(
                    score_select_graph.max_batch
                ):
                    graph_miss_batch_layers += 1
                elif selected_count > int(
                    score_select_graph.max_select
                ):
                    graph_miss_select_layers += 1
                else:
                    graph_miss_other_layers += 1

        if fused_selection is None and slot >= prefix_len:
            z_i = z_parts[:, slot : slot + 1, :].expand(
                parent_count, -1, -1
            )[:, 0, :]
            candidate_ids, candidate_base_vals, candidate_weights, candidate_biases = (
                candidate_tables
            )
            bias_i = (
                None
                if candidate_biases is None
                else candidate_biases[slot]
            )
            if int(expansion_count) <= int(candidate_ids.shape[-1]):
                fused_selection = score_select_graph.run(
                    z_proj=z_i,
                    h_state=parent_hidden_2d,
                    candidate_ids=candidate_ids[slot],
                    candidate_base_vals=candidate_base_vals[slot],
                    candidate_weight=candidate_weights[slot],
                    candidate_bias=bias_i,
                    parent_scores=parent_scores_1d,
                    parent_indices=parent_indices,
                    selected_count=selected_count,
                    select_candidate_count=int(expansion_count),
                )
                if fused_selection is not None:
                    fused_score_select_layers += 1
                elif parent_count > int(
                    score_select_graph.max_batch
                ):
                    graph_miss_batch_layers += 1
                elif selected_count > int(
                    score_select_graph.max_select
                ):
                    graph_miss_select_layers += 1
                else:
                    graph_miss_other_layers += 1
            if fused_selection is None:
                top_vals, top_ids, log_z = (
                    correction_scorer.candidate_topk_from_precomputed(
                        z_i,
                        parent_hidden_2d,
                        candidate_ids[slot],
                        candidate_base_vals[slot],
                        candidate_weights[slot],
                        bias_i,
                        expansion_count,
                        sort_result=False,
                        compute_log_z=True,
                    )
                )
                top_scores = top_vals.float() - log_z.float()
        elif fused_selection is None:
            logits = base_logits[:, slot : slot + 1, :].expand(
                parent_count, -1, -1
            )
            logits_2d = logits[:, 0, :].float()
            top_vals, top_ids = torch.topk(
                logits_2d, k=expansion_count, dim=-1
            )
            log_z = torch.logsumexp(
                logits_2d, dim=-1, keepdim=True
            )
            top_scores = top_vals - log_z
        add_elapsed(
            detail_times,
            (
                "tree_score_select_fused"
                if fused_selection is not None
                else "tree_score"
            ),
            t_score,
            device,
        )

        t_select = detail_start(detail_times, device)
        scored_candidate_count = (
            int(top_scores.shape[1])
            if fused_selection is None
            else int(expansion_count)
        )
        if fused_selection is None:
            selected_count = min(
                int(selected_count),
                int(parent_count) * max(1, scored_candidate_count),
            )
            if selected_count <= 0:
                break

        if fused_selection is not None:
            if len(fused_selection) == 6:
                (
                    selected_parent_pos,
                    _selected_rank,
                    selected_tokens,
                    selected_path_scores,
                    selected_parent_indices,
                    fused_child_hidden_2d,
                ) = fused_selection
            else:
                (
                    selected_parent_pos,
                    _selected_rank,
                    selected_tokens,
                    selected_path_scores,
                    selected_parent_indices,
                ) = fused_selection
        else:
            path_scores = parent_scores_1d.unsqueeze(1) + top_scores
            _selected_prios, selected_flat = torch.topk(
                path_scores.reshape(-1),
                k=selected_count,
                dim=0,
            )
            selected_parent_pos = torch.div(
                selected_flat,
                scored_candidate_count,
                rounding_mode="floor",
            )
            selected_rank = selected_flat.remainder(
                scored_candidate_count
            )
            selected_parent_indices = parent_indices.index_select(
                0, selected_parent_pos
            )
            selected_tokens = top_ids[
                selected_parent_pos, selected_rank
            ].contiguous()
            selected_path_scores = path_scores[
                selected_parent_pos, selected_rank
            ].contiguous()
        add_elapsed(
            detail_times, "tree_select", t_select, device
        )

        t_gru_batch = detail_start(detail_times, device)
        if fused_child_hidden_2d is not None:
            child_hidden_batch = fused_child_hidden_2d.unsqueeze(0)
            fused_gru_layers += 1
        else:
            parent_h0 = parent_hidden_2d.index_select(
                0, selected_parent_pos
            ).unsqueeze(0)
            child_hidden_batch = update_child_hidden(
                selected_tokens.view(-1, 1), parent_h0
            )
            fallback_gru_layers += 1
        add_elapsed(
            detail_times,
            "tree_gru_update",
            t_gru_batch,
            device,
        )

        t_write = detail_start(detail_times, device)
        new_start = node_count + 1
        new_end = node_count + selected_count + 1
        new_indices = node_index_range[new_start:new_end]
        hidden_states[new_start:new_end].copy_(
            child_hidden_batch.squeeze(0)
        )
        path_scores_t[new_start:new_end].copy_(
            selected_path_scores.float()
        )
        parents_t[new_start:new_end].copy_(
            selected_parent_indices.long()
        )
        tokens_t[node_count : node_count + selected_count] = (
            selected_tokens.long()
        )
        depths_t[node_count : node_count + selected_count] = int(
            child_depth
        )

        frontier_t = new_indices
        frontier_len = int(selected_count)
        node_count += int(selected_count)
        add_elapsed(
            detail_times, "tree_write", t_write, device
        )

    tree_metadata_on_host = False
    prune_wall_ms = 0.0
    used_gpu_topb = False
    if pruned and node_count > prune_target:
        synchronize(device)
        prune_wall_start = time.perf_counter()
        t_prune = detail_start(detail_times, device)
        supertree_node_count = int(node_count)
        used_gpu_topb = bool(
            is_accelerator_device(device) and float(depth_bonus) <= 0.0
        )
        if used_gpu_topb:
            kept_old_indices_t = select_topb_prefix_tree_tensor(
                path_scores_t[: node_count + 1],
                depths_t[:node_count],
                prune_target,
                float(depth_bonus),
            )
            selected_parent_old_t = parents_t[kept_old_indices_t]
            selected_token_t = tokens_t[kept_old_indices_t - 1]
            selected_depth_t = depths_t[kept_old_indices_t - 1]
            selected_metadata = torch.stack(
                (
                    kept_old_indices_t,
                    selected_parent_old_t,
                    selected_token_t,
                    selected_depth_t,
                ),
                dim=0,
            ).detach().cpu().tolist()
            kept_old_indices_list = [
                int(value) for value in selected_metadata[0]
            ]
            all_parents = [0] * (int(node_count) + 1)
            all_tokens = [0] * (int(node_count) + 1)
            all_depths = [0] * (int(node_count) + 1)
            for metadata_index, old_index in enumerate(
                kept_old_indices_list
            ):
                all_parents[old_index] = int(
                    selected_metadata[1][metadata_index]
                )
                all_tokens[old_index] = int(
                    selected_metadata[2][metadata_index]
                )
                all_depths[old_index] = int(
                    selected_metadata[3][metadata_index]
                )
        else:
            prune_metadata = torch.stack(
                (
                    parents_t[: node_count + 1].float(),
                    path_scores_t[: node_count + 1].float(),
                    torch.cat(
                        (
                            torch.zeros(
                                (1,),
                                dtype=torch.float32,
                                device=device,
                            ),
                            tokens_t[:node_count].float(),
                        ),
                        dim=0,
                    ),
                    torch.cat(
                        (
                            torch.zeros(
                                (1,),
                                dtype=torch.float32,
                                device=device,
                            ),
                            depths_t[:node_count].float(),
                        ),
                        dim=0,
                    ),
                ),
                dim=0,
            ).detach().cpu().tolist()
            all_parents = [int(value) for value in prune_metadata[0]]
            all_scores = [float(value) for value in prune_metadata[1]]
            all_tokens = [int(value) for value in prune_metadata[2]]
            all_depths = [int(value) for value in prune_metadata[3]]
            prune_scores = [
                float(score) + float(depth_bonus) * int(depth)
                for score, depth in zip(all_scores, all_depths)
            ]
            kept_old_indices_list = select_topb_prefix_tree(
                all_parents,
                prune_scores,
                prune_target,
            )
        if len(kept_old_indices_list) != prune_target:
            raise RuntimeError(
                "Top-B pruning selected "
                f"{len(kept_old_indices_list)} nodes, expected {prune_target}"
            )
        old_to_new = {0: 0}
        old_to_new.update(
            {
                int(old_index): int(new_index)
                for new_index, old_index in enumerate(
                    kept_old_indices_list, start=1
                )
            }
        )
        node_token_ids = [
            all_tokens[index] for index in kept_old_indices_list
        ]
        node_depths = [
            all_depths[index] for index in kept_old_indices_list
        ]
        parents = [-1] + [
            old_to_new[all_parents[index]]
            for index in kept_old_indices_list
        ]
        node_count = prune_target
        tree_metadata_on_host = True
        add_elapsed(detail_times, "tree_topb_prune", t_prune, device)
        prune_wall_ms = (time.perf_counter() - prune_wall_start) * 1000.0

    t_to_cpu = detail_start(detail_times, device)
    if not tree_metadata_on_host:
        node_token_ids = [
            int(x)
            for x in tokens_t[:node_count].detach().cpu().tolist()
        ]
        node_depths = [
            int(x)
            for x in depths_t[:node_count].detach().cpu().tolist()
        ]
        parents = [
            int(x)
            for x in parents_t[: node_count + 1]
            .detach()
            .cpu()
            .tolist()
        ]
        if parents:
            parents[0] = -1
    child_maps = [dict() for _ in range(node_count + 1)]
    for node_index, token_id in enumerate(node_token_ids, start=1):
        parent_index = int(parents[node_index])
        child_maps[parent_index][int(token_id)] = int(node_index)
    add_elapsed(detail_times, "tree_to_host", t_to_cpu, device)

    selected_depth_counts = [0] * (int(depth_limit) + 1)
    for selected_depth in node_depths:
        selected_depth_counts[int(selected_depth)] += 1
    final_layer_widths = selected_depth_counts[1 : int(depth_limit) + 1]

    t_visibility = detail_start(detail_times, device)
    node_token_tensor = (
        torch.tensor(node_token_ids, dtype=torch.long)
        if tree_metadata_on_host
        else tokens_t[:node_count]
    )
    node_depth_tensor = (
        torch.tensor(node_depths, dtype=torch.long)
        if tree_metadata_on_host
        else depths_t[:node_count]
    )
    visibility = build_visibility(parents)
    add_elapsed(detail_times, "tree_visibility", t_visibility, device)
    stats = {
        "node_count": float(len(node_token_ids)),
        "tree_height": float(max(node_depths) if node_depths else 0),
        "root_top1_probability": float(root_top1_probability),
        "expanded_layers": float(expanded_layers),
        "scored_parent_count": float(scored_parent_count),
        "max_frontier_width": float(max_frontier_width),
        "fused_score_select_layers": float(fused_score_select_layers),
        "fused_gru_layers": float(fused_gru_layers),
        "fallback_gru_layers": float(fallback_gru_layers),
        "graph_miss_batch_layers": float(graph_miss_batch_layers),
        "graph_miss_select_layers": float(graph_miss_select_layers),
        "graph_miss_other_layers": float(graph_miss_other_layers),
        "pruned": 1.0 if pruned else 0.0,
        "supertree_width": float(max_supertree_width) if pruned else 0.0,
        "supertree_node_count": float(supertree_node_count),
        "prune_wall_ms": float(prune_wall_ms),
        "used_gpu_topb": 1.0 if used_gpu_topb else 0.0,
    }
    stats.update(
        {
            f"final_layer_width_{index + 1}": float(width)
            for index, width in enumerate(final_layer_widths)
        }
    )
    return (
        node_token_tensor,
        node_depth_tensor,
        parents,
        child_maps,
        visibility,
        stats,
    )


@torch.inference_mode()
def dartree_generate(
    *,
    draft_model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    tree_budget: int,
    expansion_k: int,
    depth_bonus: float,
    variant: str,
    supertree_width: int,
    graft_ratio: float = 1.0,
    correction_scorer: DominoCorrectionScorer,
    candidate_vocab_size: int,
    score_select_graph: DARTreeScoreSelectGraph,
    temperature: float,
    stop_token_ids: list[int] | None,
    record_round_trace: bool = False,
    record_entropy: bool = False,
    verify_buffer_nodes: int = 0,
) -> SimpleNamespace:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "dartree_generate only supports batch size 1."
        )

    device = draft_model.device
    input_ids = input_ids.to(device)
    mask_token_id = int(draft_model.mask_token_id)
    shift_label = bool(
        getattr(draft_model.config, "dflash_config", {}).get(
            "shift_label", False
        )
    )
    prefix_len = int(
        getattr(draft_model, "pure_draft_prefix_len", 0)
    )
    k_draft = block_size if shift_label else block_size - 1
    if k_draft <= 0:
        raise ValueError("tree mode requires block_size > 1.")

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)
    max_tree_nodes = max(
        1 + int(tree_budget), int(verify_buffer_nodes)
    )
    output_ids = torch.full(
        (1, max_length + max_tree_nodes + block_size + 1),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(
        output_ids.shape[1], device=device
    ).unsqueeze(0)
    stop_tensor = (
        None
        if stop_token_ids is None
        else torch.tensor(stop_token_ids, device=device)
    )

    verify_input_ids_buffer = torch.empty(
        (1, max_tree_nodes), dtype=torch.long, device=device
    )
    verify_position_ids_buffer = torch.empty(
        (1, max_tree_nodes), dtype=torch.long, device=device
    )
    attention_mask_buffer = torch.zeros(
        (
            1,
            1,
            max_tree_nodes,
            max_length + max_tree_nodes + block_size + 1,
        ),
        dtype=target.dtype,
        device=device,
    )
    tree_visibility_buffer = torch.empty(
        (max_tree_nodes, max_tree_nodes),
        dtype=torch.bool,
        device=device,
    )

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = stage_dict()
    detail_times: dict[str, float] | None = None
    tree_stat_totals: dict[str, float] = defaultdict(float)
    tree_heights: list[int] = []
    round_trace: list[dict[str, Any]] = []
    tree_buffers: dict[str, torch.Tensor] = {}
    # Target entropy (nats) at each accepted draft-chain node, DARTree path.
    entropy_target: list[float] = []

    hidden_collector = SelectedHiddenCollector(
        target, draft_model.target_layer_ids
    )
    hidden_collector.__enter__()
    prefill_start = cuda_time(device)
    hidden_collector.clear()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=False,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[
        :, num_input_tokens : num_input_tokens + 1
    ] = sample(output.logits, temperature)
    target_hidden = hidden_collector.cat()
    time_to_first_token = cuda_time(device) - prefill_start

    start = num_input_tokens
    acceptance_lengths: list[int] = []
    previous_tree_start = 0
    previous_tree_length = 0
    draft_prefill = True
    decode_start = cuda_time(device)

    while start < max_length:
        round_index = len(acceptance_lengths)
        round_start = int(start)
        generated_start = int(start - num_input_tokens)

        output_ids[
            :, start + 1 : start + block_size
        ] = mask_token_id
        block_output_ids = output_ids[
            :, start : start + block_size
        ].clone()
        round_root_token_id = (
            output_ids[0, start].detach().clone()
        )
        round_root_token_value = int(
            round_root_token_id.detach().cpu().item()
        )

        draft_start = cuda_time(device)
        t_draft_embed = detail_start(detail_times, device)
        noise_embedding = target.model.embed_tokens(
            block_output_ids
        )
        add_elapsed(
            detail_times,
            "draft_input_embed",
            t_draft_embed,
            device,
        )
        t_draft_backbone = detail_start(detail_times, device)
        parallel_hiddens = draft_model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[
                :,
                past_key_values_draft.get_seq_length() :
                start + block_size,
            ],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )
        add_elapsed(
            detail_times,
            "draft_backbone",
            t_draft_backbone,
            device,
        )
        if not shift_label:
            parallel_hiddens = parallel_hiddens[
                :, -block_size + 1 :, :
            ]
        t_draft_cache = detail_start(detail_times, device)
        past_key_values_draft.crop(start)
        add_elapsed(
            detail_times,
            "draft_cache_crop",
            t_draft_cache,
            device,
        )
        t_base_head = detail_start(detail_times, device)
        base_logits = target.lm_head(
            parallel_hiddens[:, :k_draft, :]
        )
        add_elapsed(
            detail_times,
            "draft_base_lm_head",
            t_base_head,
            device,
        )
        t_zproj = detail_start(detail_times, device)
        z_parts = correction_scorer.project_z(
            parallel_hiddens[:, :k_draft, :]
        )
        add_elapsed(
            detail_times,
            "draft_z_project",
            t_zproj,
            device,
        )

        candidate_count = max(
            int(expansion_k),
            min(
                int(candidate_vocab_size),
                int(base_logits.shape[-1]),
            ),
        )
        candidate_count = max(
            1,
            min(int(candidate_count), int(base_logits.shape[-1])),
        )
        t_candidate_topk = detail_start(detail_times, device)
        candidate_base_vals, candidate_ids = torch.topk(
            base_logits[0, :k_draft, :].float(),
            k=candidate_count,
            dim=-1,
        )
        add_elapsed(
            detail_times,
            "draft_candidate_topk",
            t_candidate_topk,
            device,
        )
        t_candidate_gather = detail_start(
            detail_times, device
        )
        flat_candidate_ids = candidate_ids.reshape(-1)
        candidate_weights = (
            correction_scorer.fc2_weight.index_select(
                0, flat_candidate_ids
            )
            .view(k_draft, candidate_count, -1)
            .contiguous()
        )
        candidate_biases = (
            correction_scorer.fc2_bias.index_select(
                0, flat_candidate_ids
            )
            .view(k_draft, candidate_count)
            .contiguous()
            if correction_scorer.fc2_bias is not None
            else None
        )
        add_elapsed(
            detail_times,
            "draft_candidate_gather",
            t_candidate_gather,
            device,
        )
        candidate_tables = (
            candidate_ids.contiguous(),
            candidate_base_vals.contiguous(),
            candidate_weights,
            candidate_biases,
        )

        draft_elapsed = cuda_time(device) - draft_start
        draft_counted = not draft_prefill
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time(device)
        else:
            stage_times["draft"] += draft_elapsed

        tree_start = cuda_time(device)
        if variant == "graft":
            prune_target, _ret_budget = resolve_graft_retain(
                tree_budget, graft_ratio
            )
            _retain_budget = prune_target
        else:
            _retain_budget = None
        (
            node_token_ids,
            node_depths,
            parents,
            child_maps,
            visibility_cpu,
            tree_stats,
        ) = build_dartree_supertree(
            root_token_id=round_root_token_id,
            parallel_hiddens=parallel_hiddens[
                :, :k_draft, :
            ],
            base_logits=base_logits,
            budget=tree_budget,
            expansion_k=expansion_k,
            prefix_len=prefix_len,
            supertree_width=supertree_width,
            depth_bonus=depth_bonus,
            correction_scorer=correction_scorer,
            z_parts=z_parts,
            candidate_tables=candidate_tables,
            score_select_graph=score_select_graph,
            tree_buffers=tree_buffers,
            pruned=(variant in ("pruned", "graft")),
            retain_budget=_retain_budget,
            detail_times=detail_times,
        )
        tree_build_elapsed = cuda_time(device) - tree_start
        stage_times["tree_build"] += tree_build_elapsed
        tree_heights.append(int(tree_stats["tree_height"]))
        for key, value in tree_stats.items():
            if isinstance(value, (int, float)):
                tree_stat_totals[key] += float(value)

        compile_start = cuda_time(device)
        (
            verify_input_ids,
            verify_position_ids,
            verify_attention_mask,
            previous_tree_start,
            previous_tree_length,
        ) = prepare_tree_attention_inputs(
            root_token_id=round_root_token_value,
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=(
                verify_position_ids_buffer
            ),
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        compile_elapsed = cuda_time(device) - compile_start
        stage_times["tree_setup"] += compile_elapsed

        verify_start = cuda_time(device)
        hidden_collector.clear()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=False,
        )
        posterior = sample(output.logits, temperature)
        verify_elapsed = cuda_time(device) - verify_start
        stage_times["verify"] += verify_elapsed

        commit_start = cuda_time(device)
        t_commit_follow = detail_start(detail_times, device)
        accepted_indices, next_token = follow_verified_tree(
            child_maps, posterior
        )
        add_elapsed(
            detail_times,
            "commit_follow_tree",
            t_commit_follow,
            device,
        )
        accepted_len = int(len(accepted_indices))
        if record_entropy and accepted_len > 1:
            # Accepted draft-chain nodes are accepted_indices[1:].  Node i is a
            # draft token the target accepted using its logit slot parents[i]
            # (which predicts the token at node i).  Report the target model's
            # word-distribution entropy at that slot for each chain position.
            for k in range(1, accepted_len):
                node = int(accepted_indices[k])
                parent = int(parents[node])
                ent = float(
                    logits_entropy(
                        output.logits[:, parent : parent + 1, :], temperature
                    )[0, 0].item()
                )
                entropy_target.append(ent)
                print(
                    f"[dartree-entropy] out_pos={start + k} "
                    f"chain_index={k} tree_node={node} target={ent:.4f}"
                )
        t_commit_tensor = detail_start(detail_times, device)
        accepted_index_tensor = torch.tensor(
            accepted_indices, dtype=torch.long, device=device
        )
        accepted_tokens = verify_input_ids.index_select(
            1, accepted_index_tensor
        )
        output_ids[
            :, start : start + accepted_len
        ] = accepted_tokens
        output_ids[:, start + accepted_len] = int(next_token)
        add_elapsed(
            detail_times,
            "commit_write_tokens",
            t_commit_tensor,
            device,
        )

        t_commit_cache = detail_start(detail_times, device)
        compact_dynamic_cache(
            past_key_values_target, start, accepted_indices
        )
        add_elapsed(
            detail_times,
            "commit_compact_cache",
            t_commit_cache,
            device,
        )
        t_commit_hidden = detail_start(detail_times, device)
        target_hidden = hidden_collector.index_select_cat(
            1, accepted_index_tensor
        )
        add_elapsed(
            detail_times,
            "commit_select_hidden",
            t_commit_hidden,
            device,
        )

        acceptance_lengths.append(accepted_len)
        start += accepted_len
        commit_elapsed = cuda_time(device) - commit_start
        stage_times["commit"] += commit_elapsed
        round_total_elapsed_ms = float(
            (
                float(draft_elapsed)
                + float(tree_build_elapsed)
                + float(compile_elapsed)
                + float(verify_elapsed)
                + float(commit_elapsed)
            )
            * 1000.0
        )

        if bool(record_round_trace):
            round_record: dict[str, Any] = {
                "round": int(round_index),
                "absolute_start": int(round_start),
                "generated_start": int(generated_start),
                "accept": int(accepted_len),
                "next_token": int(next_token),
                "tree_height": int(
                    tree_stats.get("tree_height", 0)
                ),
                "tree_nodes": int(
                    tree_stats.get("node_count", 0)
                ),
                "expanded_layers": int(
                    tree_stats.get("expanded_layers", 0)
                ),
                "scored_parent_count": int(
                    tree_stats.get("scored_parent_count", 0)
                ),
                "max_frontier_width": int(
                    tree_stats.get("max_frontier_width", 0)
                ),
                "draft_elapsed": float(draft_elapsed),
                "draft_counted": bool(draft_counted),
                "tree_build_elapsed": float(
                    tree_build_elapsed
                ),
                "tree_setup_elapsed": float(
                    compile_elapsed
                ),
                "verify_elapsed": float(verify_elapsed),
                "commit_elapsed": float(commit_elapsed),
                "total_elapsed_ms": float(
                    round_total_elapsed_ms
                ),
            }
            final_layer_widths = [
                int(tree_stats.get(f"final_layer_width_{index}", 0))
                for index in range(1, int(k_draft) + 1)
            ]
            if any(final_layer_widths):
                round_record["final_layer_widths"] = final_layer_widths
            round_trace.append(round_record)

        if stop_tensor is not None:
            new_tokens = output_ids[
                :, start - accepted_len : start + 1
            ]
            if torch.isin(new_tokens[0], stop_tensor).any():
                break

    hidden_collector.__exit__(None, None, None)

    valid_length = min(int(start) + 1, int(max_length))
    output_ids = output_ids[:, :valid_length]
    output_ids = output_ids[
        :, output_ids[0] != mask_token_id
    ]
    if stop_tensor is not None:
        stop_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[
                :,
                : num_input_tokens
                + int(stop_indices[0].item())
                + 1,
            ]

    num_output_tokens = int(
        output_ids.shape[1] - num_input_tokens
    )
    decode_time = cuda_time(device) - decode_start
    entropy_summary: dict[str, float] = {}
    if record_entropy and entropy_target:
        entropy_summary["mean_target_entropy"] = float(
            sum(entropy_target) / len(entropy_target)
        )
        entropy_summary["n_recorded"] = float(len(entropy_target))
        print(
            "[dartree-entropy-summary] "
            f"n={len(entropy_target)} "
            f"mean target={sum(entropy_target)/len(entropy_target):.4f}"
        )
    return SimpleNamespace(
        output_ids=output_ids.detach().cpu(),
        num_input_tokens=int(num_input_tokens),
        num_output_tokens=num_output_tokens,
        time_to_first_token=float(time_to_first_token),
        time_per_output_token=float(
            decode_time / max(1, num_output_tokens)
        ),
        acceptance_lengths=[
            int(x) for x in acceptance_lengths
        ],
        stage_times={
            key: float(value)
            for key, value in stage_times.items()
        },
        detail_times={
            key: float(value)
            for key, value in (detail_times or {}).items()
        },
        tree_stat_totals={
            key: float(value)
            for key, value in tree_stat_totals.items()
        },
        tree_heights=[int(x) for x in tree_heights],
        round_trace=round_trace,
        entropy=entropy_summary,
        entropy_target=entropy_target,
    )


def summarize_choice(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    chosen = [row[key] for row in rows if key in row and row[key] is not None]
    new_tokens = [int(x["new_tokens"]) for x in chosen]
    decode_times = [float(x["decode_time_s"]) for x in chosen]
    acceptance = [int(v) for x in chosen for v in x.get("acceptance_lengths", [])]
    tree_heights = [int(v) for x in chosen for v in x.get("tree_heights", [])]
    stage_sum: dict[str, float] = defaultdict(float)
    detail_sum: dict[str, float] = defaultdict(float)
    tree_stat_totals: dict[str, float] = defaultdict(float)
    for x in chosen:
        for name, value in x.get("stage_times", {}).items():
            stage_sum[name] += float(value)
        for name, value in x.get("detail_times", {}).items():
            detail_sum[name] += float(value)
        for name, value in x.get("tree_stat_totals", {}).items():
            tree_stat_totals[name] += float(value)
    total_tokens = max(1, sum(new_tokens))
    rounds = len(acceptance)
    return {
        "samples": len(chosen),
        "new_tokens": int(sum(new_tokens)),
        "decode_time_s": float(sum(decode_times)),
        "tpot_ms": float(sum(decode_times) / total_tokens * 1000.0),
        "decoding_rounds": int(rounds),
        "micro_acceptance_length": (
            float(np.mean(acceptance)) if acceptance else 0.0
        ),
        "mean_acceptance_length": (
            float(
                np.mean(
                    [
                        np.mean(x.get("acceptance_lengths", [0]))
                        for x in chosen
                    ]
                )
            )
            if chosen
            else 0.0
        ),
        "acceptance_length_histogram": {
            str(i): int(acceptance.count(i))
            for i in range((max(acceptance) if acceptance else 0) + 1)
        },
        "mean_tree_height": float(np.mean(tree_heights)) if tree_heights else 0.0,
        "max_tree_height": int(max(tree_heights)) if tree_heights else 0,
        "stage_tpot_ms": {
            name: float(stage_sum[name] / total_tokens * 1000.0)
            for name in sorted(stage_sum)
        },
        "detail_tpot_ms": {
            name: float(detail_sum[name] / total_tokens * 1000.0)
            for name in sorted(detail_sum)
        },
        "mean_tree_stats": {
            name: float(tree_stat_totals[name] / max(1, rounds))
            for name in sorted(tree_stat_totals)
        },
    }


def row_from_response(
    response: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    generated_ids = response.output_ids[
        0, response.num_input_tokens :
    ]
    vocab_size = int(
        len(tokenizer) if hasattr(tokenizer, "__len__") else 0
    )
    if vocab_size <= 0:
        vocab_size = int(
            getattr(tokenizer, "vocab_size", 0) or 0
        )
    generated_list = [
        int(x)
        for x in generated_ids.detach().cpu().tolist()
    ]
    output_flat = (
        response.output_ids.detach()
        .cpu()
        .to(dtype=torch.int64)
        .contiguous()
        .view(-1)
    )
    valid_generated = [
        token_id
        for token_id in generated_list
        if 0 <= int(token_id) < int(vocab_size)
    ]
    invalid_generated = [
        token_id
        for token_id in generated_list
        if not (0 <= int(token_id) < int(vocab_size))
    ]
    row = {
        "new_tokens": int(response.num_output_tokens),
        "decode_time_s": (
            float(response.time_per_output_token)
            * int(response.num_output_tokens)
        ),
        "prefill_time_s": float(response.time_to_first_token),
        "acceptance_lengths": [
            int(x)
            for x in getattr(
                response, "acceptance_lengths", []
            )
        ],
        "stage_times": {
            key: float(value)
            for key, value in getattr(
                response, "stage_times", {}
            ).items()
        },
        "detail_times": {
            key: float(value)
            for key, value in getattr(
                response, "detail_times", {}
            ).items()
        },
        "tree_stat_totals": {
            key: float(value)
            for key, value in getattr(
                response, "tree_stat_totals", {}
            ).items()
        },
        "tree_heights": [
            int(x)
            for x in getattr(response, "tree_heights", [])
        ],
        "round_trace": list(
            getattr(response, "round_trace", [])
        ),
        "invalid_generated_token_count": int(
            len(invalid_generated)
        ),
        "output_sha1": hashlib.sha1(
            output_flat.numpy().tobytes()
        ).hexdigest(),
        "output_new_ids": generated_list,
        "text": tokenizer.decode(
            valid_generated, skip_special_tokens=True
        ),
    }
    if invalid_generated:
        row["invalid_generated_token_min"] = int(
            min(invalid_generated)
        )
        row["invalid_generated_token_max"] = int(
            max(invalid_generated)
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fixed or pruned DARTree decoding."
        )
    )
    parser.add_argument(
        "--target-model", default="Qwen/Qwen3-4B"
    )
    parser.add_argument(
        "--draft-model",
        # Public Domino checkpoint on Hugging Face.
        default="Huang2020/Qwen3-4B-Domino-b16",
    )
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument(
        "--dataset-shuffle-seed", type=int, default=0
    )
    parser.add_argument(
        "--max-samples", type=int, default=4
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256
    )
    parser.add_argument(
        "--block-size", type=int, default=16
    )
    parser.add_argument(
        "--tree-budget", type=int, default=64
    )
    parser.add_argument(
        "--expansion-k", type=int, default=64
    )
    parser.add_argument(
        "--candidate-vocab-size", type=int, default=64
    )
    parser.add_argument(
        "--variant",
        choices=["fixed", "pruned", "graft"],
        default="pruned",
    )
    parser.add_argument(
        "--supertree-width", type=int, default=12
    )
    parser.add_argument(
        "--graft-ratio", type=float, default=0.6,
        help=(
            "Graft variant only: fraction of tree-budget retained as draft nodes "
            "after Top-B pruning (in (0, 1]); the rest goes to the retrieval subtree."
        ),
    )
    parser.add_argument(
        "--depth-bonus", type=float
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument(
        "--record-round-trace", action="store_true"
    )
    parser.add_argument(
        "--record-entropy", action="store_true",
        help=(
            "Print the TARGET model word-distribution entropy at each accepted "
            "draft-chain position (DARTree path and Domino chain baseline)."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT / "results" / "dartree.json"
        ),
    )
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> None:
    device_kind = device_type(args.device)
    if device_kind not in ("cuda", "npu"):
        raise ValueError(
            "DARTree requires a CUDA (NVIDIA) or Ascend NPU device; "
            f"got {args.device!r}"
        )
    if not accelerator_available(args.device):
        raise ValueError(
            f"device {args.device!r} was requested but its backend is "
            "unavailable (no CUDA-enabled torch build, or no CANN/torch_npu "
            "install for Ascend)"
        )
    positive_args = {
        "--tree-budget": args.tree_budget,
        "--expansion-k": args.expansion_k,
        "--candidate-vocab-size": args.candidate_vocab_size,
        "--supertree-width": args.supertree_width,
    }
    for name, value in positive_args.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(args.candidate_vocab_size) < int(args.expansion_k):
        raise ValueError(
            "--candidate-vocab-size must be at least --expansion-k"
        )
    if args.variant == "graft":
        if not (0.0 < float(args.graft_ratio) <= 1.0):
            raise ValueError(
                "--graft-ratio must be in (0, 1]"
            )
    if args.depth_bonus is None:
        args.depth_bonus = 0.0 if args.variant == "fixed" else -0.2
    if args.variant == "fixed":
        if abs(float(args.depth_bonus)) > 1e-12:
            raise ValueError(
                "fixed DARTree requires --depth-bonus 0"
            )
    else:
        if float(args.depth_bonus) > 0.0:
            raise ValueError(
                "pruned/graft DARTree requires a non-positive --depth-bonus"
            )
        if args.run_baselines:
            raise ValueError(
                "--run-baselines is only available with --variant fixed"
            )


def main() -> None:
    args = parse_args()
    validate_contract(args)

    random.seed(0)
    np.random.seed(0)
    seed_all(0)

    device = torch.device(args.device)
    set_device(device)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()
    draft_config = normalize_draft_config(
        AutoConfig.from_pretrained(args.draft_model)
    )
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model
    )

    prefix_len = int(
        getattr(draft_model, "pure_draft_prefix_len", 0)
    )
    shift_label = bool(
        getattr(draft_model.config, "dflash_config", {}).get(
            "shift_label", False
        )
    )
    k_draft = (
        int(args.block_size)
        if shift_label
        else int(args.block_size) - 1
    )
    steps = k_draft - prefix_len
    if steps <= 0:
        raise ValueError(
            "cannot create Domino graph runner: "
            f"block_size={args.block_size}, "
            f"k_draft={k_draft}, prefix_len={prefix_len}"
        )

    chain_graph_runner = None
    if args.run_baselines:
        chain_graph_runner = DraftCorrectionGraphRunner(
            draft_model=draft_model,
            target_model=target,
            batch_size=1,
            steps=steps,
            hidden_dim=int(
                target.lm_head.weight.shape[1]
            ),
            gru_hidden_dim=(
                draft_model.prefix_gru.hidden_size
            ),
            vocab_size=int(
                target.lm_head.weight.shape[0]
            ),
            prefix_token_count=1 + prefix_len,
            device=device,
        )

    correction_scorer = DominoCorrectionScorer(
        draft_model=draft_model,
        target_model=target,
        hidden_dim=int(target.lm_head.weight.shape[1]),
        gru_hidden_dim=draft_model.prefix_gru.hidden_size,
    )
    dummy_token = torch.zeros(
        (1,), dtype=torch.long, device=device
    )
    dummy_hidden = torch.zeros(
        (1, int(draft_model.prefix_gru.hidden_size)),
        dtype=next(draft_model.parameters()).dtype,
        device=device,
    )
    for _ in range(3):
        correction_scorer.update_hidden(
            dummy_token, dummy_hidden
        )
    synchronize(device)

    candidate_vocab_size = int(args.candidate_vocab_size)
    expansion_k = min(
        int(args.expansion_k), candidate_vocab_size
    )
    per_layer_widths = (
        supertree_width_schedule(
            k_draft, expansion_k, int(args.supertree_width)
        )
        if args.variant in ("pruned", "graft")
        else None
    )
    construction_budget = (
        sum(per_layer_widths)
        if per_layer_widths is not None
        else int(args.tree_budget)
    )
    planned_pairs = planned_score_select_pairs(
        budget=construction_budget,
        depth_limit=k_draft,
        candidate_count=expansion_k,
        per_layer_widths=per_layer_widths,
    )
    planned_max_batch = max(
        (
            batch_size
            for batch_size, _selected in planned_pairs
        ),
        default=1,
    )
    planned_max_select = max(
        (
            selected
            for _batch_size, selected in planned_pairs
        ),
        default=1,
    )
    score_select_graph = (
        DARTreeScoreSelectGraph(
            correction_scorer,
            device=device,
            max_batch=planned_max_batch,
            candidate_count=candidate_vocab_size,
            select_candidate_count=expansion_k,
            max_select=planned_max_select,
            dtype=next(draft_model.parameters()).dtype,
            include_gru=True,
            warm_pairs=sorted(planned_pairs),
        )
    )

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None:
        count = max(0, int(args.max_samples))
        if len(dataset) > count:
            dataset = dataset.shuffle(
                seed=int(args.dataset_shuffle_seed)
            )
        dataset = dataset.select(
            range(min(len(dataset), count))
        )

    rows: list[dict[str, Any]] = []
    for idx, instance in enumerate(
        tqdm(dataset, desc="dartree-eval")
    ):
        messages: list[dict[str, str]] = []
        for turn_index, user_content in enumerate(
            instance["turns"]
        ):
            messages.append(
                {"role": "user", "content": user_content}
            )
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(device)
            row: dict[str, Any] = {
                "sample_index": int(idx),
                "turn_index": int(turn_index),
            }

            if args.run_baselines:
                ar_response = draft_model.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=1,
                    stop_token_ids=[
                        tokenizer.eos_token_id
                    ],
                    temperature=args.temperature,
                    graph_runner=None,
                    use_bias=True,
                    return_dict=True,
                )
                row["ar"] = row_from_response(
                    ar_response, tokenizer
                )

            if args.run_baselines:
                chain_response = draft_model.spec_generate(
                    target=target,
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    stop_token_ids=[
                        tokenizer.eos_token_id
                    ],
                    temperature=args.temperature,
                    graph_runner=chain_graph_runner,
                    use_bias=True,
                    record_entropy=args.record_entropy,
                    return_dict=True,
                )
                row["domino"] = row_from_response(
                    chain_response, tokenizer
                )

            tree_response = dartree_generate(
                draft_model=draft_model,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                tree_budget=args.tree_budget,
                expansion_k=args.expansion_k,
                depth_bonus=args.depth_bonus,
                variant=args.variant,
                supertree_width=args.supertree_width,
                graft_ratio=args.graft_ratio,
                correction_scorer=correction_scorer,
                candidate_vocab_size=args.candidate_vocab_size,
                score_select_graph=score_select_graph,
                temperature=args.temperature,
                stop_token_ids=[
                    tokenizer.eos_token_id
                ],
                record_round_trace=(
                    args.record_round_trace
                ),
                record_entropy=args.record_entropy,
                verify_buffer_nodes=(
                    1 + int(args.tree_budget)
                ),
            )
            row["dartree"] = row_from_response(
                tree_response, tokenizer
            )
            rows.append(row)
            messages.append(
                {
                    "role": "assistant",
                    "content": row["dartree"]["text"],
                }
            )

    summary: dict[str, Any] = {
        "config": vars(args),
        "domino": (
            summarize_choice(rows, "domino")
            if args.run_baselines
            else None
        ),
        "ar": (
            summarize_choice(rows, "ar")
            if args.run_baselines
            else None
        ),
        "dartree": summarize_choice(rows, "dartree"),
    }
    if summary.get("ar"):
        summary["dartree_speedup_vs_ar_pct"] = 100.0 * (
            summary["ar"]["tpot_ms"]
            / summary["dartree"]["tpot_ms"]
            - 1.0
        )
    if summary.get("domino"):
        summary[
            "domino_speedup_vs_ar_pct"
        ] = 100.0 * (
            summary["ar"]["tpot_ms"]
            / summary["domino"]["tpot_ms"]
            - 1.0
        )
        summary[
            "dartree_tpot_delta_vs_domino_pct"
        ] = 100.0 * (
            summary["dartree"]["tpot_ms"]
            / summary["domino"]["tpot_ms"]
            - 1.0
        )
        summary["dartree_accept_delta_vs_domino"] = (
            summary["dartree"]["mean_acceptance_length"]
            - summary["domino"]["mean_acceptance_length"]
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "rows": rows},
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
