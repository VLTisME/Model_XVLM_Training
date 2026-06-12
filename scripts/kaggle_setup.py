"""One-shot, idempotent Kaggle environment setup for X-VLM training.

Run from the repo root:  python scripts/kaggle_setup.py

Keeps Kaggle's CUDA torch + modern tokenizers. Installs the few pinned deps X-VLM
(transformers 4.12.5 era) needs, clones the X-VLM source, and applies 4 patches:
  1. transformers: relax the hard `tokenizers<0.11` pin (X-VLM uses its own pure-python
     BertTokenizer; the modern preinstalled tokenizers is never exercised). Line-based and
     idempotent — also REPAIRS a previously corrupted table file.
  2. X-VLM utils: make the CIDEr (captioning) import optional — unused for retrieval.
  3. X-VLM swin: replace scipy.interp2d rel-pos interpolation (removed in scipy>=1.14)
     with torch bicubic (timm-standard; affects only the 224->384 init of the bias table).
  4. X-VLM xvlm.py: torch>=2.6 defaults torch.load(weights_only=True) which rejects the
     checkpoint pickle -> pass weights_only=False (trusted file from the official release).
Every pip install runs separately so one failure cannot take down the rest (lesson from
the first Kaggle run: tokenizers had no cp312 wheel and killed the whole install line).
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

BICUBIC_PATCH = '''

# --- STAR patch: torch bicubic rel-pos interpolation (scipy-free), overrides def above ---
import torch as _torch
def interpolate_relative_pos_embed(rel_pos_bias, dst_num_pos, param_name=""):
    src_num_pos, num_heads = rel_pos_bias.size()
    src_size = int(src_num_pos ** 0.5); dst_size = int(dst_num_pos ** 0.5)
    if src_size != dst_size:
        print("Position interpolate %s from %dx%d to %dx%d (torch bicubic)"
              % (param_name, src_size, src_size, dst_size, dst_size))
        rel = rel_pos_bias.detach().float().permute(1, 0).reshape(1, num_heads, src_size, src_size)
        rel = _torch.nn.functional.interpolate(rel, size=(dst_size, dst_size),
                                               mode="bicubic", align_corners=False)
        rel_pos_bias = rel.reshape(num_heads, dst_size * dst_size).permute(1, 0).to(rel_pos_bias.dtype)
    return rel_pos_bias
'''


def pip(*args: str) -> None:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args])
    print(f"[setup] pip install {' '.join(args)} -> {'OK' if r.returncode == 0 else 'FAILED (continuing)'}")


def main() -> None:
    # ---- 1) pinned deps, each on its own line ----
    pip("huggingface_hub==0.10.1")          # API surface expected by transformers 4.12.5
    pip("sacremoses", "gdown", "zstandard", "ruamel.yaml")
    pip("--no-deps", "transformers==4.12.5", "timm==0.4.9")
    assert importlib.util.find_spec("transformers"), "transformers failed to install"

    # ---- 2) patch transformers tokenizers pin (idempotent, repairs corruption) ----
    spec = importlib.util.find_spec("transformers")
    table = pathlib.Path(spec.origin).parent / "dependency_versions_table.py"
    fixed = re.sub(r'^(\s*)"tokenizers":.*$', r'\1"tokenizers": "tokenizers",',
                   table.read_text(), flags=re.M)
    compile(fixed, str(table), "exec")      # hard guarantee: never write invalid python
    table.write_text(fixed)
    print("[setup] transformers tokenizers pin relaxed:", table)

    # ---- 3) X-VLM source ----
    xvlm = REPO / "third_party" / "X-VLM"
    if not xvlm.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/zengyan-97/X-VLM", str(xvlm)], check=True)
        print("[setup] cloned X-VLM")

    # ---- 4) patch: CIDEr optional ----
    u = xvlm / "utils" / "__init__.py"
    s = u.read_text()
    old = "from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD"
    if "try:\n    " + old not in s:
        u.write_text(s.replace(old, "try:\n    " + old + "\nexcept Exception:\n    CiderD = None"))
        print("[setup] CIDEr import made optional")

    # ---- 5) patch: scipy-free rel-pos interpolation ----
    sw = xvlm / "models" / "swin_transformer.py"
    if "STAR patch: torch bicubic" not in sw.read_text():
        sw.write_text(sw.read_text() + BICUBIC_PATCH)
        print("[setup] swin rel-pos interpolation -> torch bicubic")

    # ---- 6) patch: torch>=2.6 weights_only default ----
    xv = xvlm / "models" / "xvlm.py"
    s = xv.read_text()
    s2 = s.replace("torch.load(ckpt_rpath, map_location='cpu')",
                   "torch.load(ckpt_rpath, map_location='cpu', weights_only=False)")
    if s2 != s:
        xv.write_text(s2)
        print("[setup] xvlm torch.load -> weights_only=False")

    print("[setup] DONE — env + X-VLM source + 4 patches ready")


if __name__ == "__main__":
    main()
