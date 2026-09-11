# Graft 方法融入 DARTree — 修改大纲

> 论文：Shen et al., 2026, *"Draft Less, Retrieve More: Hybrid Tree Construction for
> Speculative Decoding"* (Graft). arXiv:2605.20104
> 所有路径相对于仓库根 `E:\repos\DARTree`。
> 状态标记：`[ ]` 未开始 · `[x]` 已完成。

---

## 1. 方法理解（论文要点）

Graft 核心是「**先剪枝、后嫁接**（prune-then-graft）」的**固定预算补偿**框架：

1. **预算释放（pruning）**：在标定的剪枝检查点用累积路径分数算置信度
   `c_d = exp(max_j S_{d,j})`，低于阈值 `τ_d` 时剪掉低置信 draft 分支，释放候选预算。
2. **检索嫁接（retrieval grafting）**：用释放的预算，从 **GPU 常驻邻接矩阵**
   `M ∈ V^{|V|×k}`（每行存某 token 的 top-k 后继）做表查找，按 stage-adaptive 模板
   生成检索分支，与 drafting 并行预取，不占关键路径。
3. **混合树验证 + 在线更新**：`T_draft^s ∪ G_ret^s`，总预算 `K_max = K_draft^s + K_ret^s`
   不变，走标准 tree-attention 单次验证；验证得到的 target 分布刷新 `M[x̃_i] = argtop_k(p̃_{i+1})`
   （接受和拒绝的节点都刷）。

关键结论：剪枝不是「删候选」而是「**释放预算**」；检索不是独立 drafter 而是「**补偿被
判错的覆盖**」。两者互补才能同时提速度（剪枝省钱）和提 MAT（检索补覆盖），打破纯剪枝的
Pareto 边界。全程训练 free、lossless（不改 target、不改验收规则）。

---

## 2. 论文对 DFlash 的结论（与本项目最直接相关，Appendix F / Sec 4.5）

DARTree 前身/近亲是 **DFlash 式 block drafting**（一次并行 diffusion 出整块 token）。
论文给出 **Graft-DFLASH(16)** 初步结果（Qwen3-8B 五 benchmark 平均 3.40×→3.71×，+9.1%）：

- DFlash 一次并行 draft 16 token，但平均接受长度常只有 5~7，块内存在
  **双向去噪 vs 自回归验证**的结构性不匹配，部分位置验证效用低。
- DFlash 在验证前产出 **token 级 draft logits** → 可作每 token 置信度（论文观察其与
  接受率正相关）→ 天然适合置信度引导的块级剪枝。
- 因 DFlash 是 **chain 式** draft-and-verify（非树），论文 demo 用 **Graft(TAIL)**：
  保留高置信前缀 + 检索 token 填尾 `B_ret = B − B_df`，总预算 `B` 不变，lossless。
- 论文留白：块 drafter 的拓扑/置信度标定与树 drafter 不同，需更系统设计（本项目要做的）。

> **本项目比论文 demo 更强，但表述要准确**：DARTree 本就是**树式验证**
> （`build_dartree_supertree` + tree-attention 单次验证），所以可直接采用论文主方法的
> 「**root 中心检索子树**」，而不是退化到 tail 链式（Graft(TAIL)）。「树内嫁接」只是本文档对
> "在树式验证里做 prune-then-graft"的泛称，具体实现是 **root 子树共享根合并**，不是把
> 检索 token 逐位填回被剪节点的父级空位（后者见 §4.2 ablation）。

---

## 3. 现有代码框架分析

### 3.1 目录结构
```
eval_dartree.py         # 主评估入口 + 核心解码逻辑 + 树构造 + 图 runner（2351 行）
run_dartree.py          # CLI 包装，转调 eval_dartree
run_entropy.py          # 熵记录小工具
utils/
  __init__.py           # 导出 DFlashDraftModel / DominoCorrectionScorer / DraftCorrectionGraphRunner
  correction.py         # correction MLP + GRU 的权重展开、scorer、chain 图 runner + Triton 内核
  draft_model.py        # DFlashDraftModel（block 并行 draft）+ spec_generate（Domino chain 基线）
  data.py               # 数据集加载
  device_backend.py     # CUDA / Ascend NPU 抽象
  retrieval.py          # （新增）GraftAdjacencyMatrix + 检索模板
tests/
  conftest.py           # 保证 pytest 可导入 utils
  test_retrieval.py     # 检索原语 CPU 单测
```

### 3.2 关键组件与职责

| 组件 | 位置 | 职责 |
|---|---|---|
| `DFlashDraftModel` | `utils/draft_model.py` | Qwen3 扩散 drafter；`forward` 一次块并行产出 `parallel_hiddens`；`spec_generate` 是 Domino **链式**基线 |
| `DominoCorrectionScorer` | `utils/correction.py` | 展开 correction MLP（`w_z/w_s/fc1/fc2`）+ `prefix_gru`；提供 `project_z`、`candidate_topk_from_precomputed`、`update_hidden`、`_gru_input_proj_table` |
| `DraftCorrectionGraphRunner` | `utils/correction.py` | Domino **单链**校正的 CUDA graph runner（链式基线用） |
| `DARTreeScoreSelectGraph` | `eval_dartree.py` | 树扩展「打分+选前沿」CUDA graph（含 GRU 融合），`run()` |
| `build_dartree_supertree` | `eval_dartree.py` | **核心**：逐深度 best-first 扩展候选树 →（pruned）Top-B 剪枝 → 输出 token/depth/parents/child_maps/visibility |
| `prepare_tree_attention_inputs` | `eval_dartree.py` | 由 parents 构造 tree position ids、祖先 attention mask、visibility |
| `build_visibility` | `eval_dartree.py` | 由 parents 构造祖先可见性矩阵 |
| `follow_verified_tree` / `compact_dynamic_cache` | `eval_dartree.py` | 验收路径追踪 + KV cache 压缩 |
| `dartree_generate` | `eval_dartree.py` | 顶层解码循环：prefill → draft block → 建树 → 验证 → commit |
| `main` | `eval_dartree.py` | 装配模型、scorer、图 runner、数据集、汇总输出 |

### 3.3 每轮 decoding 数据流（`dartree_generate`）

1. **prefill**：target 前向，填 `output_ids`，采样首 token，`SelectedHiddenCollector` 抓 `target_hidden`。
2. **draft**：`block_output_ids` → `noise_embedding` → `draft_model(...)` 得 `parallel_hiddens`；
   `base_logits = target.lm_head(parallel_hiddens[:, :k_draft])`；`z_parts = project_z(...)`；
   `candidate_base_vals/candidate_ids = topk(base_logits, k=candidate_count)` → gather 出
   `candidate_weights/candidate_biases` → `candidate_tables`。
3. **tree build**：`build_dartree_supertree(...)` 逐深度打分选前沿，产出
   `node_token_ids/node_depths/parents/child_maps/visibility_cpu/stats`。
   - `fixed`：平均分配预算到各深度；`pruned`：先扩 supertree 再 Top-B 剪枝（`depth_bonus` 微调）。
4. **verify**：`prepare_tree_attention_inputs` → `target(...)` 单次验证，`posterior = sample(logits)`。
5. **commit**：`follow_verified_tree` 追踪验收路径 → 写 token → `compact_dynamic_cache` →
   `hidden_collector.index_select_cat` → 推进 `start`。

### 3.4 可复用 / 需扩展的接口（对 Graft 而言）

**直接可复用**：
- `build_dartree_supertree` 已产出全套树结构；Graft 只需在它之后**额外构造 root 中心检索子树
  并与保留 draft 树共享根合并**，后续 verify/commit 全不改。
- `prepare_tree_attention_inputs` / `build_visibility` 只吃 `parents` + token id，**对节点来源
  无感知**，检索节点可无缝并入 tree-attention。
- `candidate_tables` + `candidate_topk_from_precomputed` 已能算出 corrected top-k 分数（含 `log_z`），
  可作「置信度/累积路径分数」来源。

**需要新增**：
1. **GPU 常驻邻接矩阵 `M`**（`[vocab_size, k]` 的 token-id 张量）— 已实现（Phase 0）。
2. **线上更新 `M`**：验证后对每个验证节点（接受+拒绝）`argtop_k` 刷新行 — Phase 3。
3. **检索模板**：rank-path 模板 + 宽度分配 — 已实现（Phase 0）。
4. **剪枝阈值/置信度**：V1 用固定比例 `ratio`；V2 用 draft logits 置信度 checkpoint。

---

## 4. 融入方案（设计决策 + 分阶段）

### 4.1 总体定位

- **不改 draft/verify/commit 主循环骨架**，只在「树构建」与「验证」之间插入 Graft 层：
  `build_dartree_supertree(...)` →（新）`graft_hybrid_tree(...)` → 原有 verify/commit。
- Graft 作为 **第三种 variant**（`variant=graft`），`fixed`/`pruned` 保持不变，便于 A/B 对比
  （`--run-baselines` 已能出 Domino/AR 基线）。

### 4.2 核心设计：什么是「剪枝」、什么是「嫁接」

- **候选块**：DFlash 每轮给 `k_draft` 个位置、每位置 `candidate_count` 个候选。
- **剪枝（预算释放）—— V1 固定比例（简化默认）**：
  - 沿用 `pruned` 路径：构 supertree → Top-B 剪枝，但剪枝目标从 `budget` 改为
    `draft_retain = min(round(ratio × budget), supertree_node_count)`，`ratio ∈ (0,1]`
    （新参数 `--graft-ratio`，默认 0.6）。
  - `K_ret = budget − draft_retain` 即释放给检索的预算；`draft_retain + K_ret = budget` 不变。
  - 改动极小：`build_dartree_supertree` 尾部 `select_topb_prefix_tree_tensor(..., int(budget), ...)`
    的 `int(budget)` 换成 `int(draft_retain)`。
  - 语义：论文正式方法的**固定比例退化版**（≈ Figure 5 "w/o prune" 镜像）；牺牲了置信度自适应。
    作为 V1 先跑通全链路。
- **剪枝（预算释放）—— V2 置信度自适应（可选升级）**：
  用 draft 侧置信度决定剪枝 stage `s`（对标 checkpoint d0/d1/d5），使 `K_draft^s` 随每轮变化，
  取代固定 `ratio`。V1 稳定后做。
- **嫁接（检索）—— 论文正式方法**：
  - 检索报文是 **root 中心子树 `G_ret^s`**：从当前轮 root token `x_t` 出发，按 stage 模板
    查 `M`（`x_u = M[x_parent, r_u]`），BFS 批量查表生成 `K_ret^s` 节点；检索只依赖 `x_t`，
    可在 drafting 期间并行预取。
  - 将 `T_draft^s` 与 `G_ret^s` 合并：`G_ret^s` 作为 root 下的独立检索分支与保留 draft 树
    并排（Algorithm 2："Append K_ret retrieved nodes to the retained draft nodes"）。
  - **"填充剪枝空位"指预算个数**，而非父—子拓扑位置：检索预算属于「被剪掉的弱 draft 延续」，
    不与强 draft 抢预算（区别于 Graft(ROOT)）；检索节点**不继承**被剪节点的父节点。
- **可选 ablation（非论文方法）—— 逐空位填充 / 树内嫁接**：
  把检索 token 顶替到被剪分支**原父节点下、同深度**，等价 Graft(TAIL) 树化变体，前缀依赖更强，
  仅作对照，**不作为默认**。

#### Graft(ROOT) 与正式 prune-then-graft 的区别（易混淆）

两者**最终拓扑几乎一样**（检索都从 root 展开、都并 root 下），区别在**检索预算从谁手里拿**：

| 维度 | Graft(ROOT) | prune-then-graft（正式） |
|---|---|---|
| 是否先剪枝 | ❌ 不剪，保留全量 draft | ✅ 先剪低置信 draft 分支 |
| 检索预算来源 | 挤占 root 附近**最强 draft**名额 | 接盘**被剪掉的弱 draft**名额 |
| draft/检索切分 | 固定（≈ Fig.5 "w/o prune"） | 随剪枝阶段 `s` 自适应 |
| Fig.2 结果 | 平均低于 EAGLE3 | 最高（省钱 + 补覆盖） |

论文 §2.2 的 *"inserted exactly where pruning creates space"* 指**剪枝腾出的预算名额**，而非
父节点拓扑位。本项目默认 `prune-then-graft`；`--graft-no-prune` 可退化到 Graft(ROOT) 作对照。

### 4.3 合并语义：共享根 + 同父 token 去重（关键）

**共享根，不做"第二个根"**：检索树无独立根，与 draft 树共享 root `x_t`（node 0）。合并时检索的
depth-1 节点（`M[x_t, 0..w1-1]`）作为 root 的**新增 sibling** 与保留 draft 子节点并排；
depth≥2 挂检索 sibling 下。因此「根第一层已被 draft 占满」不是冲突（同层宽度相加，
`build_visibility` 天然支持多兄弟）。

**真正的坑：`child_maps[parent][token]` 是单值索引**。`follow_verified_tree` 靠 token→唯一 child
查验收路径；检索 token 与同父下已有 child 重复时 dict 覆盖会造成验收歧义。两种处理：

- **方案 A（去重跳过，推荐默认）**：逐 rank 查 `M[parent, rank]`，遇重复 rank+1 继续；
  不改 `child_maps` / `follow_verified_tree`，实现最简。
- **方案 B（去重 redirect，可选增强）**：重复 token 的检索后代改挂到已有节点下续深链；
  保留预算、更贴"补偿"语义，但需重算整条子树 parents/索引，复杂度高。

Stage 模板的 **root 层宽度 `w1`** 既避免 root 层被检索占满（防 Graft(ROOT) 覆辙），也为
方案 A skip 预留备选 rank。Phase 2 先落 A，B 留作 ablation。

### 4.4 预算参数（V1 固定 ratio，参数化，勿硬编码）

V1（简化默认，先跑通全链路）：
- `K_max = tree_budget`（默认 64）。
- `draft_retain = min(round(ratio × K_max), supertree_node_count)`，`ratio=0.6`（`--graft-ratio`）。
- `K_ret = K_max − draft_retain`。
- 检索模板：单一满封套模板（深度 ≤ `k_draft`），BFS 取前 `K_ret` 节点；`level_widths` 显式可
  复现论文 Table 7。

V2（可选升级）：置信度 checkpoint `s` 选 `(K_draft^s, K_ret^s)`，取代固定 ratio（预留
`--graft-stages`）。

### 4.5 验收不变（lossless）

合并后仍 `follow_verified_tree` + 标准 SD 验收；`M` 更新不影响输出 token，只影响候选建议 →
lossless。检索节点被 target 拒绝时同样刷 `M`。

---

## 5. 分阶段实施任务清单

### Phase 0 — 基建 ✅ 已完成
- [x] `utils/retrieval.py`（新文件）：`GraftAdjacencyMatrix` 类
  - `__init__(vocab_size, k, device, pad_token_id, dtype)`：分配 `M = full([V,k], pad_token)`。
  - `update(token_ids, logits)`：对每验证节点 `topk(logits, k)` 覆盖相应行；越界 id 跳过；
    维护 `initialized` 布尔掩码（区分「未初始化行」与「legit 含 pad 的行」）。
  - `lookup(parent_ids, ranks)`：`M[parent_ids, ranks]` 广播查表（支持标量 rank 广播 + 逐节点
    `[N]` 并行查表；rank 越界抛 `IndexError`）。
  - `is_ready(token_ids)` / `ready_rows()`：未初始化行兜底/统计。
  - `state_dict()` / `load_state_dict()`：保存/加载。
  - 注：`from_warmup` 未单独实现——暖机可复用 `update()` 喂外部语料 logits，Phase 3 再定。
- [x] 检索模板 `build_retrieval_template(level_widths)` + `default_level_widths(budget, max_depth, root_width)` —— **并入 `retrieval.py`，不单设 `templates.py`**。
  - `build_retrieval_template`：BFS 生成不平衡 rank-path 模板，rank0 贪心链自动延伸最深；
    保证前缀闭合（父索引 < 子索引），可直转 `parents`。
  - `default_level_widths`：`K_ret` 折成 root 层 `w1` + 余层均分的每层宽度列表；显式
    `level_widths=[...]` 可复现论文 Table 7 精确形状；保留接口供 V2 stage 表。
- [x] `utils/__init__.py`：导出 `GraftAdjacencyMatrix` / `build_retrieval_template` / `default_level_widths`。
- [x] `tests/test_retrieval.py` + `tests/conftest.py`：CPU 单测。
- [ ] ⚠️ 待办：本机无 torch，`pytest tests/` 尚未实跑，需在有 torch 环境验证。

### Phase 1 — 固定比例剪枝（budget release，V1）✅ 已完成
- [x] `build_dartree_supertree` 增加 `retain_budget`（=`draft_retain`）参数，替换尾部剪枝
  `int(budget)`（共 4 处：阈值判断 / GPU topb / CPU topb / `node_count`），产出 `T_draft^s` 元数据；
  `None` 时退化为 `budget`（向后兼容，`fixed`/`pruned` 行为不变）。
- [x] 新增纯函数 `resolve_graft_retain(budget, ratio, supertree_node_count=None) -> (draft_retain, k_ret)`
  （位于 `utils/retrieval.py`，非 `eval_dartree.py`，便于无模型 import 单测）。
- [x] `utils/__init__.py` 导出 `resolve_graft_retain`。
- [x] `tests/test_graft_budget.py`（新）：sum 恒等、ratio=1 退化、0.6 舍入、supertree clamp、下限、非法输入。
- [ ] ⚠️ 待办：本机无 torch，pytest 未实跑；`dartree_generate` 尚未接 `retain_budget`（等 Phase 2 一起）。

#### Phase 1.5 — 接线（ratio 贯通，已补）
- [x] `eval_dartree.py::dartree_generate` 新增 `graft_ratio: float = 1.0` 参数；当
  `variant == "graft"` 时调用 `resolve_graft_retain(tree_budget, graft_ratio)` 得到
  `_retain_budget`，并传给 `build_dartree_supertree(retain_budget=...)`；`pruned` 分支改
  `variant in ("pruned", "graft")`。
- [x] `main`/`parse_args`：`--variant` 增加 `graft` 选项；新增 `--graft-ratio`（默认 0.6）；
  call site 传入 `graft_ratio=args.graft_ratio`；`per_layer_widths` 触发条件改为
  `variant in ("pruned", "graft")`。
- [x] `validate_contract`：graft 时校验 `--graft-ratio ∈ (0,1]`；报错文案加 "graft"。
- [ ] ⚠️ 待办：graft 目前只完成了「按 ratio 剪枝」，检索嫁接部分（Phase 2）尚未接，
  故 graft variant 还不能产生正确输出（剪枝后未补检索，只是预算缩小）。

### Phase 2 — 检索嫁接 + 合并（retrieval grafting）✅ 已完成（除 into_slot ablation）
- [x] `build_retrieval_subtree(root_token, M, template, k_ret)`（`utils/retrieval.py`）：root 中心按模板 BFS 查 `M`，
  产出 token/depth/parents/ranks/stats；空行/无效哨兵回退（rank 向上扫描、节点连同子树丢弃），
  前缀闭合合法树；`k_ret` 封顶。
- [x] `graft_hybrid_tree(draft_tree, retrieval_tree, k_max, matrix=None, root_token_id=None, dedup="skip", slots=None)`
  （`utils/retrieval.py`）：`T_draft^s` 与 `G_ret^s` **共享根合并**、重排索引、重建
  `parents`/`node_token_ids`/`node_depths`/`child_maps`/`visibility`（纯 torch 重建，无 numpy 依赖）。
  - 默认去重 **方案 A**（`dedup="skip"`）：冲突时用 `M` 重扫后继 rank 救援节点，否则丢节点+子树
    （`graft_rank_rescanned_nodes`/`dedup_skipped_nodes` 统计）；
  - ablation 分支 2 `--graft-dedup redirect`（方案 B：重复后代续到已有节点，`dedup_redirected_nodes`）；
  - ablation 分支 1 `--graft-insert into_slot`（逐空位填充/树内嫁接，非论文方法对照）：
    `build_dartree_supertree` 新增 `pruned_slots_out` 参数（Top-B 剪枝后暴露被剪节点的
    (kept 父新索引, 深度) 槽位，GPU/CPU topb 两路径均支持），`_merge_into_slots` 按
    `M[父token, rank]` 逐槽位填充、rank 逐父递增避让哨兵与同父重复、槽位深度一致性校验；
  - `k_max` 防御性检查：合并后节点数 ≤ `K_max`。
- [x] `dartree_generate` 在 `build_dartree_supertree` 后调用 `build_retrieval_subtree` + `graft_hybrid_tree`，
  输出与 `prepare_tree_attention_inputs` 兼容（token/depth 转张量、`parents`/`child_maps` 列表、CPU visibility）；
  新增 `stage_times["graft"]` 与 `graft_*` 统计随 `tree_stat_totals` 汇总；`--graft-insert` 选择
  `root`（默认，root 中心检索子树）或 `into_slot`（逐空位填充）两种合并策略。
- [x] CLI：`--graft-k`（默认 8）、`--graft-template-depth`、`--graft-root-width`（默认 8）、
  `--graft-dedup {skip,redirect}`、`--graft-insert {root,into_slot}`；`validate_contract` 校验；
  `main` 为 graft variant 实例化 `GraftAdjacencyMatrix`（`vocab_size`/`pad_token_id` 取自 target/tokenizer）；
  `run_dartree.py` 透传。
- [x] `tests/test_graft_merge.py`（新）：检索子树原语 + 合并一致性单测（前缀闭合 / child_maps /
  visibility 参照 `build_visibility` 递推）+ 去重 skip/redirect/救援 + into_slot 填充/避让/深度校验 +
  k_max 越界。
- [ ] ⚠️ 待办：本机无 torch，`pytest tests/` 未实跑；且 Phase 3 未接时 `M` 全空，
  graft 运行时会退化为空检索子树（== pruned），需 Phase 3 更新后才有实际检索命中。

### Phase 3 — 在线更新 M（online update）
- [ ] 验证后用 `output.logits`（全验证节点）调用 `M.update(...)`。
- [ ] prefill 用 prefill logits 初始化 `M`。
- [ ] 可选 `--graft-warmup` 用暖机轮次/外部语料预填。

### Phase 4 — 接线与评估
- [x] （部分，Phase 1.5 已完成）`--variant graft`、`--graft-ratio` 已接（parse_args→validate→dartree_generate→build_dartree_supertree）；
  其余 `--graft-k`、`--graft-template-depth`、`--graft-warmup`、`--graft-dedup`、
  预留 `--graft-stages`、`--graft-no-prune` 尚未加。
- [ ] `validate_contract` / `planned_score_select_pairs` 适配 graft variant。
- [ ] 汇总输出新增 `graft_stage_histogram`、`retrieved_node_count`、`retrieval_hit_rate`、
  `matrix_updated_rows`、`dedup_skipped_nodes`/`dedup_redirected_nodes`、`graft_tpot_ms`。
- [ ] 与 `fixed`/`pruned`/Domino/AR 对比（`--run-baselines`），验证 speedup 与 MAT。

### Phase 5 — 性能与健壮性
- [ ] 检索查表走 GPU gather，保持零 host-device 同步。
- [ ] 邻接矩阵 `int32` 存储；空行哨兵避免非法 token 进验证。
- [ ] `device_backend` 通路：检索/合并全程 torch eager，CUDA/NPU 均可用，不依赖 NVIDIA Triton。
- [ ] 单测：`build_retrieval_subtree`/`graft_hybrid_tree` 与 `build_visibility` 一致性。

---

## 6. 关键风险 / 待确认点

1. **confidence 信号选择**：论文 DFlash demo 用 draft logits；本项目有 `candidate_base_vals + correction bias`。
   需实验确认 base_logits top1 还是 Domino corrected top1 更贴接受率（论文只给正相关结论）。
2. **剪枝阶段对齐**：DARTree「深度」对应 block 内位置；paper checkpoint d0/d1/d5 需重映射到
   `k_draft`，建议 V2 从 2~3 阶段起步。
3. **检索冷启动**：无暖机时 `M` 空行多、命中率低。需 prefill 更新 + 可选暖机（论文 5 rounds ≈ 0.37MB）。
4. **嫁接位置 / 前缀依赖（已澄清）**：正式方法是 root 中心检索子树（前缀依赖最浅）；「逐空位填充」
   非论文方法，是更强前缀依赖的 ablation。root 子树若深剪阶段占满 root 层名额可能重蹈
   Graft(ROOT) 覆辙——靠 stage 模板控制 root 层宽度 + 分 stage 监控 `retrieval_hit_rate`。
5. **同父 token 去重（共享根合并）**：检索 token 与同父已有 child 重复会覆盖 `child_maps`
   单值索引、造成验收歧义。默认方案 A（skip-and-advance）；方案 B（redirect）留 ablation。
   用 `dedup_skipped_nodes`/`dedup_redirected_nodes` 监控。
6. **`tree_budget` 语义**：确保 `K_draft^s + K_ret^s ≤ K_max` 严格成立，且
   `prepare_tree_attention_inputs` 的 `max_tree_nodes` 覆盖合并后节点数。

---

## 7. 验收口径（客观指标）

- **lossless**：greedy 下 Graft 输出 `output_sha1` 与 AR 一致（T>0 下抽样一致）。
- **speedup**：`dartree_speedup_vs_ar_pct`、`dartree_tpot_delta_vs_domino_pct`、`graft` variant 横向对比。
- **MAT**：`mean_acceptance_length` 应 ≥ `pruned`（检索补覆盖），接近或超 `fixed`。
- **新统计**：`retrieved_node_count`、`retrieval_hit_rate` > 0，`matrix_updated_rows` 随轮次增长，
  `dedup_skipped_nodes`/`dedup_redirected_nodes` 记录去重开销。