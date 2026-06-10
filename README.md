# STAR-v3 — Training Codebase
### AI City 2026 Track 4 — Text-Based Person Anomaly Retrieval (Sim2Real, 36K-distractor gallery)

Production training pipeline for the **STAR-v3** model. Built to the annotated plan:
**X-VLM** backbone with **LoRA on the image + cross encoders only** (the **text encoder is frozen**),
trained with **`L = ITC + λ₁·ITM + λ₂·Smooth-AP`** (MLM removed), with **hard-negative mining** and
**all_gather** contrastive (XBM removed — review fix #4), optional **LHP** augmentation and a
**pose branch** fused into the image side.

> Full math + code walkthrough: **[`analyze.md`](analyze.md)**.

---

## 1. Architecture (training)

```
 Synthetic Img ~100K (.webp 384)             Caption (LLM-rewritten)
        │                                            │
   (A) LHP augmentation [toggle]                     │
        │                                            │
 (B) X-VLM Image Encoder · Swin-B  [LoRA]    (E) X-VLM Text Encoder · BERT[0:6]  [FROZEN]
        │   └─(K) Pose branch [toggle] → fuse f_V    │
        │   → f_V                                    │   → f_T
        └───────────────────┬────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │ (F) ITC  cos(f_V,f_T) · identity soft  │  ← (J) Hard-Neg + all_gather + smart sampler
        │     targets (ALBEF/X-VLM)              │
        │ (G+) Smooth-AP  (train ≈ mAP)          │
        └───────────────────┬───────────────────┘
                            │
            (G) X-VLM Cross-Encoder · BERT[6:12]  [LoRA]
                            │
                  (H) ITM head → ITM loss   (hard negatives)
                            │
        L = ITC + λ₁·ITM + λ₂·Smooth-AP
                            │
   Backprop → LoRA(Swin image) + LoRA(Cross) + ITM head (+ pose).  Text tower stays FROZEN.
```

| Component | State | Trains? |
|---|---|---|
| Image encoder (Swin-B) | **LoRA** | ✅ adapters |
| Cross-encoder (BERT 6–12) | **LoRA** | ✅ adapters |
| **Text encoder (BERT 0–6)** | **FROZEN** | ❌ (preserves language prior; domain shift is image-side) |
| ITM head, image proj, ITC temp | TRAIN | ✅ |
| Pose branch | toggle | ✅ via ITC gradient (no separate loss) |
| LHP augmentation | toggle | — (dataloader only) |
| MLM | **removed** | — |

**Loss:** `L = w_itc·ITC + λ₁·ITM(hard-neg) + λ₂·Smooth-AP`  (defaults `w_itc=1, λ₁=1, λ₂=0.3`).

---

## 2. Repo layout
```
train2/
├── README.md            # this file (architecture + how to run)
├── analyze.md           # full algorithm + code analysis (math, papers, fidelity)
├── configs/star_v3_100k.yaml
├── src/star/
│   ├── config.py        # typed config + YAML loader
│   ├── metrics.py       # mAP / MRR / R@K
│   ├── losses/          # itc (ALBEF/X-VLM faithful), smooth_ap, itm
│   ├── modules/         # hard_neg (similarity-based negative sampling)
│   ├── models/          # lora, pose, backbone (X-VLM wrapper + dummy), star_model
│   ├── data/            # dataset (consumes manifest), transforms (LHP), sampler
│   ├── engine/          # optim (AdamW + warmup-cosine), evaluator, trainer
│   └── utils/           # seed, logging, checkpoint
├── scripts/             # train.py, evaluate.py
└── tests/               # 29 pytest unit tests (math-critical)
```

---

## 3. How to run

```bash
# install
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements.txt && pip install -e .

# 1) sanity: math unit tests (no GPU, no data)
pytest -q

# 2) the DATA TEAM drops a ready manifest at manifests/star_v3.parquet
#    + the .webp images under data.image_root  (schema in §4)

# 3) sanity: overfit one batch (loss must fall toward 0 -> wiring is correct)
python scripts/train.py --config configs/star_v3_100k.yaml --overfit-one-batch

# 4) train
python scripts/train.py --config configs/star_v3_100k.yaml
#    override anything:  --set optim.lr_lora=1e-4 loss.lambda_smooth_ap=0.1

# 5) evaluate a checkpoint on VAL-B (mAP / MRR / R@K)
python scripts/evaluate.py --config configs/star_v3_100k.yaml --ckpt outputs/star_v3/best.pth
```

Without real X-VLM weights the code runs on a built-in **dummy backbone** so tests and the
overfit check work offline. To use the real model, set `model.checkpoint` and implement
`XVLMBackbone` (one integration point — see §5).

---

## 4. Data contract (delivered by the DATA TEAM)

This repo does **not** build or clean data. The data team provides a parquet **manifest**
(one row per image–caption pair) + the images:

| column | required | use |
|---|---|---|
| `image_path` | ✅ | path to the `.webp` (abs, or relative to `data.image_root`) |
| `caption` | ✅ | text paired with the image. **In VAL-B, leave it empty for distractor rows** → that image joins the gallery but is never a query (review fix #3). |
| `split` | ✅ | `train` / `valb` — **VAL-B must not share scene/identity with train** |
| `image_id` | ⬜ | gallery identity (defaults to `image_path`); rows sharing it dedup to one gallery image |
| `sequence_id` | ✅ | instance id for ITC / hard-neg masking (same-sequence frames aren't negatives) |
| `scene` | ⬜ | smart-sampler grouping |
| `action` | ⬜ | logging only |
| `bbox` | ⬜ | normalized `[x,y,w,h]`; needed only if LHP person-crop is on |
| `keypoints` | ⬜ | 17×3 flattened; needed only if the pose branch is on |

Quick check the data team can run:
```python
import pandas as pd
df = pd.read_parquet("manifests/star_v3.parquet")
assert {"image_path","caption","split","sequence_id"} <= set(df.columns)
tr, va = set(df[df.split=="train"].scene), set(df[df.split=="valb"].scene)
assert tr.isdisjoint(va)   # no VAL-B leakage
```

---

## 5. Real X-VLM backbone (WIRED ✅)

`src/star/models/backbone.py::XVLMBackbone` is implemented and **validated** against
`third_party/X-VLM` + the **X-VLM 16M** checkpoint: it builds the model, loads the checkpoint
(`missing_keys: []`), and the full STARModel trains with **48 LoRA layers** (Swin `qkv` + BERT
`query/value` on fusion layers 6–11), **text tower frozen**, ITC reusing the pretrained `temp`.

X-VLM needs `transformers==4.12.5` (incompatible with the main env), so it runs in a **separate
pinned venv**. One-time setup (Python 3.11):

```bash
# 1. source code + checkpoint (≈825 MB)
git clone --depth 1 https://github.com/zengyan-97/X-VLM third_party/X-VLM
python -m gdown 1iXgITaSbQ1oGPPvGaV0Hlae4QiJG5gx0 -O data/checkpoints/xvlm_16m_base.th

# 2. pinned venv (CPU wheel shown; for GPU use the cu121 torch wheel)
python -m venv .venv-xvlm
V=.venv-xvlm/Scripts/python.exe
$V -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
$V -m pip install -r requirements-xvlm.txt
$V -m pip install --no-deps transformers==4.12.5 timm==0.4.9   # old `tokenizers` is unbuildable on py311; we use a modern one

# 3a. relax transformers' hard tokenizers pin (X-VLM uses its own pure-python tokenizer)
$V -c "import transformers,re,pathlib as p; f=p.Path(transformers.__file__).parent/'dependency_versions_table.py'; f.write_text(re.sub(r'\"tokenizers\":[^,]+', '\"tokenizers\": \"tokenizers\"', f.read_text()))"
# 3b. make X-VLM's CIDEr (captioning, unused) import optional
$V -c "import pathlib as p; f=p.Path('third_party/X-VLM/utils/__init__.py'); s=f.read_text(); old='from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD'; f.write_text(s.replace(old, 'try:\n    '+old+'\nexcept Exception:\n    CiderD = None')) if 'try:\n    '+old not in s else None"

# 4. validate end-to-end, then train (in the pinned venv, pointing at the checkpoint)
$V third_party/validate_star_xvlm.py
$V scripts/train.py --config configs/star_v3_100k.yaml \
   --set model.backbone=xvlm model.checkpoint=data/checkpoints/xvlm_16m_base.th
```
Probe/validation scripts: `third_party/probe_xvlm.py` (build+load+forward) and
`third_party/validate_star_xvlm.py` (full STARModel LoRA + losses). The main-venv `pytest`
keeps using the dummy backbone (the X-VLM import is lazy).

---

## 6. Status & honesty
- ✅ losses / metrics / hard-neg / LoRA / optimizer / trainer / distractor-aware evaluator —
  complete, **28 unit tests pass**.
- ✅ **real X-VLM backbone wired + validated** (16M checkpoint, 48 LoRA layers, text frozen) — §5.
- ⚠️ Smooth-AP(text), LHP, the pose branch, and the frozen-text choice are **not yet proven on PAB** —
  every one is a toggle and must be confirmed on a distractor-heavy **VAL-B**. Do not trust the
  leaderboard as validation. See `analyze.md` §15 (risks).
