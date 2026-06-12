# Báo cáo Run #2 (V2) — Experiment sweep trên Kaggle T4 (2026-06-12)

> Notebook `kaggle_train_10k_v2.ipynb` — 2 thí nghiệm chạy tuần tự, tổng **168.5 phút**.
> Mọi con số lấy trực tiếp từ log Kaggle. So với mốc V1: **mAP 0.6615** (LHP on, λ₂=0.2, batch 16, dừng @2.5 epoch).

---

## 1. Cấu hình V2 (khác V1)
| Knob | V1 | V2 |
|---|---|---|
| LHP | ON | **OFF** |
| λ₂ Smooth-AP | 0.2 | **0.5** |
| Batch | 16 | **20** (accum 2) |
| Lịch học | 6 epoch, patience 2 (dừng @2.5) | **10 epoch, patience 6 eval** (=3 epoch không cải thiện) |
| LR | 2e-4 | **sweep {2e-4, 1e-4}** |
| Pose / data / phần cứng | ON / 10K-hard / T4 fp16 | giữ nguyên |

Gate overfit: 2.986 → 0.885 tại step 21 → `[overfit] OK` (ngưỡng 70% mới hoạt động đúng, hết báo giả).

## 2. Kết quả

| Exp | best mAP | R@1 | R@5 | R@10 | Thời gian | Epoch đạt đỉnh |
|---|---|---|---|---|---|---|
| **v2a** λ₂=0.5, lr 2e-4 | **0.6652** | **0.5362** | 0.8309 | 0.9018 | 84.2′ | **~epoch 3** (giữa e3), sau đó 6 eval liên tiếp không vượt → early stop @e6 |
| v2b λ₂=0.5, lr 1e-4 | 0.6593 | 0.5233 | 0.8277 | 0.9147 | 84.3′ | ~epoch 3, early stop @e6 |
| *(mốc V1)* | *0.6615* | *0.5298* | *0.8213* | *0.9098* | *31.3′* | *epoch 1* |

**Diễn biến mAP v2a:** 0.563 → 0.637 → 0.643 → 0.631 → 0.644 → 0.641 → **0.6652 (đỉnh, e3)** → 0.644 → 0.645 → 0.650 → 0.649 → 0.653 → 0.641 → stop.
**v2b:** 0.539 → 0.629 → 0.634 → 0.633 → 0.654 → 0.649 → **0.6593 (đỉnh, e3)** → … → stop.

## 3. Phần cứng / VRAM (đo thật)
- Batch 20 @384 fp16: **VRAM peak 13.5G / 15.36G** (ổn định suốt 168 phút, không OOM).
- → Headroom còn ~1.9G: **batch 24 nhiều khả năng OOM** — batch 20 đã gần trần T4. Câu hỏi "tăng batch được không" đã có đáp số bằng đo đạc.

## 4. KẾT LUẬN — đọc cho đúng (quan trọng hơn con số)

### 4.1 Ba run đều HÒA nhau → bi-encoder cosine trên 10K-hard đã BÃO HOÀ ~0.66
0.6615 (v1) · 0.6652 (v2a) · 0.6593 (v2b) — chênh lệch lớn nhất 0.0059, **dưới ngưỡng nhiễu ~0.01** của VAL-B 621 query. Tổ hợp thay đổi (bỏ LHP + λ₂ 0.2→0.5 + batch 16→20 + lịch dài gấp đôi + sweep LR) **không tạo khác biệt có ý nghĩa**.

### 4.2 Câu hỏi "dừng sớm ở cực trị địa phương?" — ĐÃ TRẢ LỜI: KHÔNG
v2a chạy tới epoch 6 với patience 6: đỉnh vẫn rơi ở **epoch ~3**, sau đó **6 lần eval liên tiếp** không vượt nổi dù LR cosine đã hạ dần. Plateau ~0.66 là **thật**, không phải artefact của early stop. (V1 dừng @2.5 epoch là gần đúng đỉnh rồi — và rẻ hơn 2.7×.)

### 4.3 LHP: tắt đi không mất gì
LHP OFF (v2a 0.6652) ≈ LHP ON (v1 0.6615) → nhất quán với số đo paper (+0.18%/~0%). **Khuyến nghị: để LHP OFF** cho đơn giản. (Đóng góp của **pose** vẫn chưa đo — ON ở cả 3 run; muốn biết phải chạy đối chứng pose OFF.)

### 4.4 LR: 2e-4 ≥ 1e-4 (cùng hướng ở cả mAP lẫn R@1), giữ 2e-4.

### 4.5 Hàm ý chiến lược (điểm mấu chốt của run này)
Train-side tuning trên 10K **đã hết nạc**. Nguồn điểm còn lại nằm ở:
1. **Cross-encoder ITM re-rank** ở inference — ITM head đã train tốt (loss ~0.02–0.1) mà điểm hiện tại chưa hề dùng. Đây là việc số 1.
2. **Scale data 50K/100K-hard** — nhiều tín hiệu hơn cho bi-encoder.
3. Ablation pose OFF (đơn giản hoá inference nếu hòa).

## 5. Checkpoint
- `/kaggle/working/v2a_smap05_lr2e-4/best.pth` (859MB) — **mAP 0.6652, checkpoint tốt nhất hiện tại** (hơn v1 trong vùng nhiễu; bàn giao theo `reports/instruct_inference.md`, pose vẫn ON → cần ViTPose cho gallery).
- `/kaggle/working/v2b_smap05_lr1e-4/best.pth` — 0.6593.

## 6. Ghi chú kỹ thuật từ log
- Loss train dao động lớn giữa các step (0.08 ↔ 1.8): bình thường — sampler gom theo video nên batch "cùng video" có floor ITC cao hơn batch trộn; không phải bất ổn.
- Grad-norm: ITC vẫn chiếm 42–93% tuỳ batch; Smooth-AP 1–15% (λ₂=0.5 cũng không làm nó lấn át).
- Cảnh báo `lr_scheduler.step() before optimizer.step()` xuất hiện 1 lần đầu run: do GradScaler skip optimizer step đầu (chuẩn với AMP) — vô hại.
