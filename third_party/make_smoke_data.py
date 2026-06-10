"""Generate a tiny synthetic manifest + images to smoke-test the FULL scripts/train.py path
(manifest -> dataset -> sampler -> trainer -> VAL-B eval -> checkpoint), backbone-agnostic.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OUT = Path("outputs/smoke")
(OUT / "img").mkdir(parents=True, exist_ok=True)

ACTIONS = ["falling", "running", "fighting", "lying", "waving"]
rng = np.random.default_rng(0)
rows = []


def make_img(name):
    arr = (rng.random((96, 96, 3)) * 255).astype("uint8")
    p = OUT / "img" / name
    Image.fromarray(arr).save(p, format="WEBP")
    return f"img/{name}"


# 30 train rows across 10 sequences/scenes
for s in range(10):
    for k in range(3):
        a = ACTIONS[s % len(ACTIONS)]
        rows.append(dict(image_path=make_img(f"tr_{s}_{k}.webp"),
                         caption=f"a person {a} in scene {s}", split="train",
                         sequence_id=f"seq{s}", scene=f"scene{s}", action=a, bbox=None, keypoints=None))

# 8 valb query rows (separate scenes) + 2 distractor rows (image-only, empty caption)
for s in range(100, 108):
    a = ACTIONS[s % len(ACTIONS)]
    rows.append(dict(image_path=make_img(f"va_{s}.webp"),
                     caption=f"a person {a} on the street", split="valb",
                     sequence_id=f"seq{s}", scene=f"scene{s}", action=a, bbox=None, keypoints=None))
for d in range(2):
    rows.append(dict(image_path=make_img(f"dist_{d}.webp"), caption="", split="valb",
                     sequence_id=f"dist{d}", scene=f"distscene{d}", action="none", bbox=None, keypoints=None))

df = pd.DataFrame(rows)
df.to_parquet(OUT / "manifest.parquet", index=False)
print(f"wrote {len(df)} rows -> {OUT/'manifest.parquet'}  (train={sum(df.split=='train')}, "
      f"valb_query={sum((df.split=='valb') & (df.caption!=''))}, distractors={sum(df.caption=='')})")
sys.exit(0)
