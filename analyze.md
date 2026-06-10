# STAR-v3 — Algorithm & Code Analysis

Full technical analysis of the training system: the math of every component, where it lives in
code, the paper + official implementation it follows, fidelity notes, and numerical caveats.

> Confidence tags: **[MEASURED]** proven in a paper · **[STANDARD]** community default ·
> **[INFERRED]** our extrapolation, verify on VAL-B. GitHub repos marked **(verify)** are unconfirmed.

**Section index** (numbers match the `analyze.md §N` references in the source docstrings):
§0 foundations · §1 metrics · §2 backbone · §3 LoRA · §4 ITC · §5 Smooth-AP · §6 ITM ·
§7 MLM (removed) · §8 hard-neg · §9 XBM (removed) · §10 LHP · §11 pose · §12 optimizer · §13 smart sampler ·
§14 loss assembly & the plan · §15 code architecture, validation & risks.

---

## §0 Foundations

**Embedding & cosine.** Each image/text → vector in ℝ^d (d=256 X-VLM proj). After L2-norm,
`s(I,T) = f_V · f_T ∈ [-1,1]`. All retrieval rests on this dot product.

**Metric = mAP, and mAP = MRR here.** General AP for one query with positive set P:
$$\text{AP}=\frac{1}{|P|}\sum_k \text{Prec}(k)\,\text{rel}(k).$$
Each query has exactly **one** GT at rank r ⇒ `AP = 1/r` ⇒ `mAP = mean(1/r) = MRR`.
Consequence: **rank-1 is everything** (3→1 gains 0.67; 10→9 gains 0.01). This is why we add
Smooth-AP (trains the rank directly) and, at inference, a cross-encoder re-rank.

---

## §1 Metrics — `src/star/metrics.py`
- `rank_of_gt(sim, gt_index)`: 1-based rank, **pessimistic on ties** (`#greater + #ties + 1`) to
  avoid optimistic scoring when many gallery items tie.
- `recall_at_k`, `mean_ap_single` (= MRR), `mean_ap_multi` (general, vectorized).
- **Validation:** `tests/test_metrics.py` checks hand values **and** cross-checks `mean_ap_multi`
  against `sklearn.metrics.average_precision_score` (`assert abs(ours-ref)<1e-5`).
- Ref: standard IR; competition spec.

---

## §2 Backbone — `src/star/models/backbone.py`
X-VLM = **Swin-B** image encoder + **BERT[0:6]** text encoder + **BERT[6:12]** cross-encoder
(cross-attention) + ITC/ITM heads. Window attention `softmax(QKᵀ/√d + B)V` (relative bias B);
cross-attention `softmax(Q_text K_imgᵀ/√d)V_img`.
- **[MEASURED]** X-VLM zero-shot 77.86 on PAB ([2502.03230]) ≫ EVA-CLIP 60.01.
- **Interface the rest of the code depends on:** `tokenizer`, `encode_image`, `encode_text`,
  `itm_logits`, `setup_finetuning`. `DummyBackbone` implements it as a small *real* trainable model
  (offline tests); **`XVLMBackbone` is WIRED + validated** against `third_party/X-VLM` + the 16M
  checkpoint (`get_vision_embeds`/`get_text_embeds`/`get_cross_embeds`/`itm_head`), 48 LoRA layers,
  text frozen, ITC reusing the pretrained `temp`. Runs in a pinned venv (README §5); import is lazy.
- Papers: X-VLM [2111.08276], Swin [2103.14030], BERT [1810.04805]. Code: `zengyan-97/X-VLM`.

---

## §3 LoRA — `src/star/models/lora.py`
`h = W₀x + (α/r)·B(Ax)`, `A∼N(0,·)`, `B=0` (starts as the pretrained map). Only A,B train; W₀ frozen.
Trainable fraction `r(d+k)/dk ≈ 4.2%` for d=k=768, r=16. Inference: **merge** `W₀ += (α/r)BA`.
- **Fidelity:** matches `microsoft/LoRA` exactly — `kaiming_uniform_(A)`, `zeros_(B)`, scaling α/r,
  dropout on x before A. `inject_lora(..., exclude=...)` lets us scope adapters to image+cross and
  **skip the text encoder** (the plan freezes text).
- `tests/test_transforms.py::test_lora_starts_as_identity_then_merges` proves init-identity + merge.
- Papers: LoRA [2106.09685]; full-FT distorts OOD [2202.10054] **[MEASURED]**.

---

## §4 ITC — `src/star/losses/itc.py`  *(faithful to ALBEF / X-VLM)*
**Identity soft targets**, not plain diagonal InfoNCE. With candidate bank 𝓑 (in-batch, all_gathered across GPUs):
$$p_{ij}=\frac{\mathbb 1[\text{id}_i=\text{id}_j]}{\sum_{k\in\mathcal B}\mathbb 1[\text{id}_i=\text{id}_k]},\quad
\mathcal L_\text{i2t}=-\frac1N\sum_i\sum_{j}p_{ij}\log\frac{e^{s_{ij}/\tau}}{\sum_k e^{s_{ik}/\tau}},\quad
\mathcal L_\text{ITC}=\tfrac12(\mathcal L_\text{i2t}+\mathcal L_\text{t2i}).$$
Unique ids ⇒ ordinary InfoNCE. Same-`sequence_id` frames are positives (no false negatives).
- **Fidelity to official code:** temperature is a learnable scalar used by **division** and
  `clamp_(0.001, 0.5)`. **Review fix #6:** when wrapping real X-VLM we reuse the backbone's
  *pretrained* `temp` (`external_temp`) instead of a fresh one. **Review fix #2:** `all_gather`
  across GPUs is implemented (gradient-preserving `GatherLayer`, no-op on 1 process) so negatives
  = batch × world_size, matching X-VLM `allgather`. Momentum distillation (ALBEF α) omitted.
- **Review fix #4:** the XBM memory bank was removed; ITC negatives = in-batch + all_gather, the
  X-VLM regime (see §9).
- **Validation:** `test_losses.py` — reduces exactly to symmetric InfoNCE for unique ids
  (`allclose` vs `F.cross_entropy`), temp clamp enforced, ids change the loss, memory path finite.
- Papers/code: ALBEF [2107.07651] `salesforce/ALBEF`; CLIP [2103.00020]; X-VLM [2111.08276].

---

## §5 Smooth-AP — `src/star/losses/smooth_ap.py`
Differentiable AP: replace the Heaviside rank-step with a sigmoid.
$$D_{ij}=s_j-s_i,\quad \mathcal R(i,S)=1+\sum_{j\ne i}\sigma(D_{ij}/\tau),\quad
\text{AP}_q=\frac{1}{|P_q|}\sum_{i\in P_q}\frac{\mathcal R(i,P_q)}{\mathcal R(i,\Omega)},\quad
\mathcal L=1-\overline{\text{AP}_q}.$$
- **Fidelity:** matches `Andrew-Brown1/Smooth_AP` (sigmoid with temperature `anneal≈0.01`,
  self-masked diagonal). We **generalize** it to the cross-modal case: per-query the relevance
  is the row's positives (single-GT ⇒ smooth `1/rank`). Single-positive AP = smooth MRR.
- **Validation:** loss ≈ 0 when perfectly ranked; larger when the positive is at the bottom.
- Status: **[MEASURED on image retrieval] → text-person [INFERRED]** (verify VAL-B). Paper [2007.12163].

---

## §6 ITM — `src/star/losses/itm.py`
Cross-encoder fuses a pair → `[CLS]` → 2-way head; `L_ITM = CE(logits, label)`. Per batch item we
build **3N pairs**: N positives `(i,i,1)`, N hard-neg texts `(i, t⁻, 0)`, N hard-neg images `(i⁻, i, 0)`.
- **Fidelity to ALBEF/X-VLM:** negatives sampled from the similarity distribution with the true
  match (and duplicates) masked — `weights = softmax(sim/temp)+1e-5; mask; multinomial`; label
  order `[1]*N + [0]*2N`. **Review fix #1:** the sampling now uses `softmax(sim / temp)` (peaked on
  the HARDEST negatives) like X-VLM, not the earlier `softmax(cos / 0.5)`; **#10:** a `1e-5` floor
  guards multinomial. Split into `build_itm_pairs` (testable) + `ITMLoss`.
- **[MEASURED]** CMP IHNM +1.01% on PAB. Papers: ALBEF [2107.07651], BLIP [2201.12086], monoBERT [1901.04085].

---

## §7 MLM — **REMOVED** (per the plan)
The original draft had an MLM auxiliary (mask 15% tokens, predict conditioned on the image). The
annotated plan **removes it** (and its head). The objective is now ITC + λ₁·ITM + λ₂·Smooth-AP.
Rationale: MLM is a pretraining-style grounding aux; for retrieval fine-tuning it is optional, and
the plan prioritizes the rank-aligned terms. (X-VLM [2111.08276] used MLM in *pretraining*.)

---

## §8 Hard-negative mining — `src/star/modules/hard_neg.py`
`P(neg=j | i) = softmax(s_{ij}/τ)` over candidates, **forbidding** the true match + duplicate
captions (PAB has 171 dup captions ⇒ false negatives). `multinomial` sampling (not argmax) keeps
diversity; an all-forbidden row falls back to uniform.
- Papers: ALBEF [2107.07651]; Robinson [2010.04592]; Kalantidis [2010.01028].

---

## §9 XBM — **REMOVED** (review fix #4)
An earlier draft added a Cross-Batch Memory bank (XBM, 1912.06798) of past embeddings as extra
ITC negatives. **It was removed** because no reference for our setup supports it: X-VLM uses
**all_gather** (no bank); ALBEF uses a **momentum** queue (slowly-updated encoder), not raw
current-encoder features. With a non-momentum bank we empirically saw **ITC degrade** (1.8→5.4 at
high LR) from stale-feature drift. ITC negatives now come from the **in-batch set + all_gather**
across GPUs (§4) — exactly the X-VLM regime.
- **Single-GPU consequence:** all_gather is a no-op, so negatives = batch−1. Use a larger batch
  and/or multi-GPU for a strong contrastive. (If this proves too few, the principled add-back is a
  *momentum* queue à la ALBEF — option #4b — not the raw XBM.)
- Papers (for reference): XBM [1912.06798]; MoCo [1911.05722]; ALBEF momentum queue [2107.07651].

---

## §10 LHP — Local-global Hybrid augmentation — `src/star/data/transforms.py`  *(toggle, train only)*
Per image: `p∼N(0.5, 1/6)`; `p>0.5` → LOCAL (person-bbox RandomResizedCrop, scale≥0.5) else GLOBAL.
Inference uses GLOBAL only → no detail lost at scoring time.
- **Safety (why it doesn't lose the subject):** stochastic (GLOBAL also seen), scale≥0.5, crop
  centered on the person bbox (fallback center crop), test uses the full image.
- Status: **[MEASURED small]** PAB +0.18% (0.1M) / ~0% (1M) [2511.22470]; person-aware variant [INFERRED].

---

## §11 Pose branch — `src/star/models/pose.py`  *(toggle)*
17 keypoints (x,y,conf) → MLP → `f_pose`; gated fuse into the image feature:
`f_V' = LayerNorm(f_V + g⊙W_p f_pose)`, `g=σ(gate)`. **Fused into the image branch, no separate
loss** (trained through the ITC/Smooth-AP gradient on f_V), as drawn in the plan.
- Status: pose component of CMP +0.66% (OpenPose) **[MEASURED small]**; ViTPose→PAB **[UNPROVEN]**.
- Papers: ViTPose [2204.12484]; ST-GCN [1801.07455]. Keypoints come from the data manifest.

---

## §12 Optimizer / schedule — `src/star/engine/optim.py`
AdamW, **β=(0.9, 0.999)** (X-VLM/ALBEF AdamW default), wd 0.02 with **no-decay** on
bias/LayerNorm/temp/gate. Differential LR: LoRA 2e-4, heads 4e-4 (text params are frozen ⇒ never
enter the optimizer). **Linear warmup → cosine decay**:
$$\eta_t = \eta_\max\tfrac{t}{t_w}\ (t<t_w),\qquad \tfrac12\eta_\max\big(1+\cos\tfrac{\pi(t-t_w)}{T-t_w}\big)\ (t\ge t_w).$$
- grad-clip 1.0; AMP (bf16, or fp16+GradScaler on RTX 3090); grad-checkpointing on.
- Papers: AdamW [1711.05101]; cosine [1608.03983]; warmup [1706.02677].

---

## §13 Smart sampler — `src/star/data/sampler.py`
Groups same-`scene`/`action` items into a batch → in-batch negatives become genuinely hard (free
hard negatives), mixed with a random fraction (`group_fraction`) so diversity is preserved.

---

## §14 Loss assembly & the plan — `src/star/models/star_model.py`
$$\boxed{\;L = w_\text{itc}\,\text{ITC} + \lambda_1\,\text{ITM(hard-neg)} + \lambda_2\,\text{Smooth-AP}\;}$$
(defaults `w_itc=1, λ₁=1, λ₂=0.3`). The plan changes vs the original draft:

| Change | Where |
|---|---|
| **Text encoder frozen** (no LoRA) | `backbone.setup_finetuning` excludes text + freezes it; `mark_only_lora_trainable` has no `txt_proj` |
| **MLM removed** (head + loss) | dropped from `star_model`, `losses/__init__`, backbone |
| **Loss = ITC + λ₁·ITM + λ₂·Smooth-AP** | `star_model.forward`, `config.LossConfig` |
| **Pose fused into image branch, no pose loss** | `star_model.forward` fuses `f_V`; no `λ_pose` term |

`forward` returns `{loss, loss_itc, loss_itm, loss_smap}`. Trainable share with text frozen ≈ 50%
on the dummy; with real LoRA only adapters + ITM head + image proj + pose + temp train.

---

## §15 Code architecture, validation & risks

**Dependency direction** (low→high): `utils → metrics, modules, losses, data → models → engine → scripts`.
Rule: **losses/metrics/modules never import models** — they take plain tensors, so the math is
unit-testable without a GPU or X-VLM.

**Backbone interface (the seam):** `tokenizer`, `encode_image`, `encode_text`, `itm_logits`,
`setup_finetuning(cfg)`. Implement these in `XVLMBackbone` and nothing else changes.

**Validation (all green):**
- `pytest` → **28 tests pass** (metrics vs sklearn; ITC reduces to InfoNCE + temp clamp + identity
  targets; Smooth-AP=0 when perfectly ranked; ITM pair builder; hard-neg prefers the hardest; LoRA
  init-identity+merge; LHP shapes; **text tower frozen**; forward returns plan losses; overfit-one-batch
  drives loss down; **distractor-aware evaluator** decouples queries from gallery and lets a
  distractor steal rank — review fix #3).
- End-to-end: model builds from the real config (dummy backbone), text frozen (trainable ≈ 50%),
  optimizer uses β=(0.9,0.999) over only trainable params, full training path optimizes.

**Top execution risks** (engineering, not modeling):
1. **VAL-B representativeness** — you tune blind if VAL-B ≠ test distribution. Build several, trust stable winners.
2. **Dataloader I/O** — `.webp` decode can bottleneck; benchmark imgs/s, cache/LMDB if needed.
3. **Few contrastive negatives on 1 GPU** (XBM removed, §9) — all_gather is a no-op single-process;
   raise batch size or go multi-GPU, else ITC is weak against the 36K-distractor task.
4. **Frozen text underfit** — if VAL-B shows the text side can't match PAB paraphrase style, the
   fallback is LoRA on just the text projection (currently fully frozen per the plan).
5. **Silent data bugs** — the data team owns the manifest gates; training adds `--overfit-one-batch`.

**Honesty.** The design is competitive, not guaranteed #1. Smooth-AP(text), LHP, the pose branch and
the frozen-text choice are unproven on PAB — each is a toggle, kept only if VAL-B confirms.

---

## Paper ↔ arXiv ↔ code
| Component | arXiv | GitHub |
|---|---|---|
| X-VLM | 2111.08276 | zengyan-97/X-VLM |
| CMP (PAB baseline) | 2411.17776 | (check paper page) |
| Efficient (SCA) / Hybrid (LHP) | 2502.03230 · 2511.22470 | (check paper pages) |
| Swin · BERT | 2103.14030 · 1810.04805 | microsoft/Swin-Transformer · google-research/bert |
| LoRA · OOD | 2106.09685 · 2202.10054 | microsoft/LoRA · huggingface/peft |
| CLIP · ALBEF · BLIP · monoBERT | 2103.00020 · 2107.07651 · 2201.12086 · 1901.04085 | openai/CLIP · salesforce/ALBEF · salesforce/BLIP · castorini/pygaggle |
| Smooth-AP | 2007.12163 | Andrew-Brown1/Smooth_AP |
| Hard-neg (Robinson/Kalantidis) | 2010.04592 · 2010.01028 | — |
| XBM · MoCo | 1912.06798 · 1911.05722 | MalongTech/research-xbm · facebookresearch/moco |
| ViTPose · ST-GCN | 2204.12484 · 1801.07455 | ViTAE-Transformer/ViTPose · yysijie/st-gcn |
| AdamW · cosine · warmup | 1711.05101 · 1608.03983 · 1706.02677 | — |
