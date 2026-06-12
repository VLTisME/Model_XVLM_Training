# Hướng dẫn dùng `best.pth` cho TEAM INFERENCE

> Checkpoint từ Run #1 (Kaggle T4, 2026-06-12): **mAP 0.6615** trên VAL-B nội bộ
> (chi tiết run: `reports/run_2026-06-12_10k_hard_t4.md`).
> Mọi quy trình dưới đây đã **test thật** (load vào X-VLM nguyên bản: `missing=[] unexpected=[]`).

---

## 0. Dùng file nào?
**Dùng `best.pth`** (checkpoint tại đỉnh mAP 0.6615, 859MB) — **KHÔNG dùng `last.pth`**
(trạng thái lúc early-stop, mAP 0.6575, kém hơn).

## 1. Bên trong `best.pth` có gì
```python
ckpt = torch.load("best.pth", map_location="cpu", weights_only=False)
ckpt["model"]          # state_dict cua STARModel (LoRA CHUA merge: <ten>.base.weight + lora_A/B)
ckpt["extra"]["cfg"]   # TOAN BO config cua run (tu dung lai dung kien truc)
ckpt["extra"]["report"]# diem VAL-B luc luu
```
Các phần tách theo **prefix của key** trong `ckpt["model"]`:

| Prefix | Là phần nào |
|---|---|
| `backbone.model.vision_encoder.*` | **Swin-B** (image encoder, có LoRA) |
| `backbone.model.text_encoder.embeddings.*` + `...encoder.layer.0–5.*` | **BERT text encoder** (frozen, y nguyên pretrain) |
| `backbone.model.text_encoder.encoder.layer.6–11.*` | **Cross-encoder** (có LoRA) |
| `backbone.model.vision_proj.*` / `text_proj.*` | 2 projection của ITC (stage-1) |
| `backbone.model.itm_head.*` | **ITM head — dùng cho cross-encoder re-rank (stage-2)** |
| `backbone.model.temp` | temperature đã học |
| `pose.*` | **pose branch** (run này pose ON — bắt buộc dùng, xem §4) |

⚠️ Vì LoRA chưa merge, **không load thẳng vào X-VLM nguyên bản được** → chọn 1 trong 2 cách dưới.

---

## 2. CÁCH A — Dùng qua repo này (đơn giản nhất, khuyên dùng)
Repo tự dựng đúng kiến trúc từ config nhúng trong checkpoint. Mẫu chuẩn là `scripts/evaluate.py`; rút gọn:

```python
import sys, torch
sys.path.insert(0, "src")                      # repo root
from star.config import load_config, _merge
from star.models import STARModel

cfg = load_config("configs/star_v3_10k_kaggle.yaml")
raw = torch.load("best.pth", map_location="cpu", weights_only=False)
_merge(cfg.model, raw["extra"]["cfg"]["model"])    # dung lai DUNG kien truc da train
model = STARModel(cfg).to("cuda").eval()
model.load_state_dict(raw["model"])                # da verify: missing=0 unexpected=0

# ---- 3 ham inference can dung ----
img_embeds, f_V = model.backbone.encode_image(images)        # [B,145,1024], [B,256]
txt_embeds, f_T = model.backbone.encode_text(ids, mask)      # [B,L,768],  [B,256]
f_V = model.pose(f_V, keypoints)                              # BAT BUOC (pose ON) — xem §4
scores_stage1 = f_T @ f_V.T                                   # cosine retrieve (cache f_V gallery 1 lan)
itm_logits = model.backbone.itm_logits(img_embeds_k, txt_embeds_q, mask_q)  # re-rank top-K
score_rerank = itm_logits.softmax(-1)[:, 1]
```
Lưu ý môi trường: cần transformers 4.12.5 pinned — chạy `python scripts/kaggle_setup.py` (Kaggle/Linux)
hoặc theo README §5 (local).

## 3. CÁCH B — Export sang X-VLM NGUYÊN BẢN (không cần code repo này)
Đã có sẵn script, **đã test** (merge 48 LoRA layers, load vào vanilla XVLM `missing=[] unexpected=[]`,
xác minh weight thay đổi thật so với pretrain):

```bash
python scripts/export_for_inference.py --ckpt best.pth --out export_infer/
# export_infer/
#   xvlm_merged.th     # key Y HET X-VLM goc (vision_encoder.*, text_encoder.*, itm_head.*, temp...)
#                      # LoRA DA merge: W = W0 + (alpha/r)·B@A  -> khong can code adapter
#   pose_branch.pth    # 9 tensor cua pose branch (encoder/proj/gate/norm)
#   export_info.json   # inventory + config run + ghi chu inference
```
Load vào X-VLM gốc (repo zengyan-97/X-VLM):
```python
from models.model_retrieval import XVLM
model = XVLM(config={...config_swinB_384 + config_bert, embed_dim=256, temp=0.07...})
model.load_state_dict(torch.load("export_infer/xvlm_merged.th"), strict=True)   # khop 100%
# stage-1:  f_V = model.get_features(image_embeds=model.get_vision_embeds(img)[0])
#           f_T = model.get_features(text_embeds=model.get_text_embeds(ids, mask))
# stage-2:  cross = model.get_cross_embeds(img_embeds, img_atts, text_embeds=txt_embeds, text_atts=mask)
#           score = model.itm_head(cross[:, 0]).softmax(-1)[:, 1]
```

---

## 4. ⚠️ BA QUY TẮC BẮT BUỘC (sai là lệch không gian embedding)
1. **POSE ON**: run này train với pose branch → **mọi ảnh gallery phải có ViTPose keypoints**
   (17×[x,y,conf], x/y chia W/H) và fuse vào f_V đúng công thức training:
   `f_V' = LayerNorm(f_V + sigmoid(gate) · proj(MLP(kpts)))` — dùng `pose.*` trong checkpoint
   (Cách A: gọi `model.pose(f_V, kpts)`; Cách B: tự dựng MLP nhỏ từ `pose_branch.pth`, cấu trúc trong
   `src/star/models/pose.py`). Bỏ qua bước này = train/eval mismatch, điểm tụt.
2. **Ảnh GLOBAL full 384×384** (resize thẳng, normalize ImageNet) — LHP crop chỉ là augmentation
   lúc train, inference KHÔNG crop.
3. **Tokenizer `bert-base-uncased`, max_length=100**, padding max_length.

## 5. Pipeline inference theo thiết kế (việc của team inference)
```
1,978 query + 36,773 gallery
  → encode (cache f_V gallery 1 lần; nhớ fuse pose §4)
  → Stage-1: cosine [Q×G]
  → Top-K (K=100–200, lấy theo điểm THÔ — đừng để Sinkhorn gate Top-K)
  → Stage-2: CROSS-ENCODER ITM RE-RANK (itm_head — vũ khí #1, đã train sẵn trong checkpoint này)
  → (ablation: Sinkhorn blend / ensemble Min-Max / Gale-Shapley-SCA — quyết bằng VAL-B)
  → top-10/query
```
Chi tiết toán từng bước: `analyze.md` §B + thiết kế tổng `../architecture.md`.
Điểm 0.6615 hiện tại là **cosine thuần chưa re-rank** — stage-2 là phần điểm còn bỏ ngỏ.

## 6. Checklist bàn giao
- [ ] `best.pth` (859MB) — tải từ Kaggle Output, lưu nơi an toàn
- [ ] (nếu dùng Cách B) chạy export 1 lần → gửi `export_infer/` (3 file)
- [ ] ViTPose keypoints cho TOÀN BỘ gallery test (hỏi team data — pipeline trích đã có)
- [ ] Xác nhận lại 3 quy tắc §4 trước khi chấm bất kỳ điểm nào
