"""Qwen2-VL top-10 re-rank — chay nhu MOT TIEN TRINH CON o cuoi notebook inference X-VLM.

Vi sao subprocess: X-VLM pin transformers cu, Qwen2-VL can transformers moi -> khong chung 1
kernel. Chay `!python qwen_rerank.py ...` la tien trinh Python rieng -> import transformers moi
sach se, KHONG dung X-VLM da nap trong kernel.

Doc answer.txt pipeline VUA xuat (top-N image_id/query, thu tu = query_index), cho Qwen2-VL chon
anh khop nhat trong top-K -> dua len rank-1, ghi answer_vlm.txt, in mAP/R@1/R@5/R@10 truoc/sau.
Resumable (luu picks json) -> bi dung thi chay lai chay tiep.

    python qwen_rerank.py --answer /kaggle/working/outputs/answer.txt \
        --query-json query_text.json --query-index query_index.txt --gt ground_truth.txt \
        --image-root /kaggle/input --model <qwen_dir|HF id> --out /kaggle/working/outputs/answer_vlm.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time
from pathlib import Path

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_queries(answer, query_json, query_index, gt, image_root, topk, skip_first_col=False):
    raw = open(query_json, encoding="utf-8").read().strip()
    recs = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    cap_of = {str(r["query_index"]): r["caption"] for r in recs}
    qorder = [l.strip() for l in open(query_index, encoding="utf-8").read().strip().splitlines()]
    gts = [l.strip() for l in open(gt, encoding="utf-8").read().strip().splitlines()]
    lines = [l.split() for l in open(answer, encoding="utf-8").read().strip().splitlines()]
    if skip_first_col:
        lines = [t[1:] for t in lines]
    assert len(qorder) == len(gts) == len(lines), \
        f"lech dong: query_index={len(qorder)} gt={len(gts)} answer={len(lines)}"
    stem2path = {Path(p).stem: p for p in glob.glob(os.path.join(image_root, "**", "*"), recursive=True)
                 if p.lower().endswith(_EXTS)}
    queries = []
    for i, full in enumerate(lines):
        cands = full[:topk]
        queries.append(dict(full=full, candidates=cands,
                            paths=[stem2path[c] for c in cands if c in stem2path],
                            caption=cap_of.get(qorder[i], ""), gt=gts[i]))
    return queries, len(stem2path)


def rerank_metrics(queries, picks):
    """Pure (testable): before = X-VLM order; after = VLM-picked promoted to rank-1."""
    def before(q):
        return q["full"].index(q["gt"]) + 1 if q["gt"] in q["full"] else None

    def after(i, q):
        p = picks.get(str(i))
        new = ([p] + [x for x in q["full"] if x != p]) if p in q["full"] else q["full"]
        return new.index(q["gt"]) + 1 if q["gt"] in new else None

    def agg(ranks):
        rr = [1.0 / r if r else 0.0 for r in ranks]
        valid = [r for r in ranks if r]
        hit = lambda k: (sum(1 for r in valid if r <= k) / len(valid)) if valid else 0.0
        return dict(mAP=sum(rr) / len(rr) if rr else 0.0, R1=hit(1), R5=hit(5), R10=hit(10))

    return agg([before(q) for q in queries]), agg([after(i, q) for i, q in enumerate(queries)])


def new_order(full, picked):
    return ([picked] + [x for x in full if x != picked]) if picked in full else full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", required=True)
    ap.add_argument("--query-json", required=True)
    ap.add_argument("--query-index", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--image-root", default="/kaggle/input")
    ap.add_argument("--model", required=True, help="local Qwen dir or HF id (Qwen/Qwen2-VL-2B-Instruct)")
    ap.add_argument("--out", default="/kaggle/working/outputs/answer_vlm.txt")
    ap.add_argument("--picks", default="/kaggle/working/vlm_picks.json")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--quant", default="none", choices=["none", "8bit"])
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--skip-first-col", action="store_true", help="set if answer.txt lines start with a query id")
    args = ap.parse_args()

    queries, n_img = load_queries(args.answer, args.query_json, args.query_index, args.gt,
                                  args.image_root, args.topk, args.skip_first_col)
    print(f"queries={len(queries)} | images found={n_img} | topk={args.topk}")

    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as VLMClass
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration as VLMClass
    from qwen_vl_utils import process_vision_info

    kw = dict(device_map="auto", torch_dtype=torch.float16)
    if args.quant == "8bit":
        from transformers import BitsAndBytesConfig
        kw = dict(device_map="auto", quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    model = VLMClass.from_pretrained(args.model, **kw).eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    print("VLM ready | quant =", args.quant)

    picks = json.load(open(args.picks)) if Path(args.picks).exists() else {}
    random.seed(0)

    def prompt(n, cap):
        return ("Below are " + str(n) + " candidate images of a person, numbered 1 to " + str(n) +
                ", then a description. Pick the ONE image that best matches it. "
                "Reply with ONLY the number (1-" + str(n) + ").\nDescription: " + str(cap))

    t0, done0 = time.time(), len(picks)
    for i, q in enumerate(queries):
        if str(i) in picks:
            continue
        paths, cands = q["paths"], q["candidates"]
        if not paths:
            picks[str(i)] = cands[0] if cands else ""
            continue
        order = list(range(len(paths)))
        if not args.no_shuffle:
            random.shuffle(order)
        content = [{"type": "image", "image": paths[o]} for o in order]
        content.append({"type": "text", "text": prompt(len(paths), q["caption"])})
        msgs = [{"role": "user", "content": content}]
        try:
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ii, vi = process_vision_info(msgs)
            inp = processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=8, do_sample=False)
            out = processor.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
            m = re.search(r"\d+", out)
            sel = int(m.group()) - 1 if m else 0
            sel = sel if 0 <= sel < len(paths) else 0
            picks[str(i)] = cands[order[sel]]
        except Exception as e:
            picks[str(i)] = cands[0]
            if i % 200 == 0:
                print("warn q", i, repr(e)[:120])
        if i % 50 == 0:
            json.dump(picks, open(args.picks, "w"))
            torch.cuda.empty_cache()
            el = (time.time() - t0) / 60
            eta = el / max(len(picks) - done0, 1) * (len(queries) - len(picks))
            print(f"{len(picks)}/{len(queries)}  {el:.1f}m  ETA ~{eta:.0f}m", flush=True)
    json.dump(picks, open(args.picks, "w"))

    B, A = rerank_metrics(queries, picks)
    print(f"\n{'':8s}{'mAP':>9}{'R@1':>9}{'R@5':>9}{'R@10':>9}")
    print(f"{'X-VLM':8s}{B['mAP']:9.4f}{B['R1']:9.4f}{B['R5']:9.4f}{B['R10']:9.4f}")
    print(f"{'+VLM':8s}{A['mAP']:9.4f}{A['R1']:9.4f}{A['R5']:9.4f}{A['R10']:9.4f}")
    print(f"{'delta':8s}{A['mAP']-B['mAP']:+9.4f}{A['R1']-B['R1']:+9.4f}{A['R5']-B['R5']:+9.4f}{A['R10']-B['R10']:+9.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i, q in enumerate(queries):
            f.write(" ".join(new_order(q["full"], picks.get(str(i)))) + "\n")
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
