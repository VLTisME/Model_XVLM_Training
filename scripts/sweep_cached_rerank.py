#!/usr/bin/env python3
"""CPU-only fusion sweep over a cached X-VLM Top-K ITM matrix."""
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from star.inference import evaluate_cached_rerank  # noqa: E402


RETRIEVAL_CONFIGS = [
    {"name": "rrf_equal_reference", "constant": 60, "pe": 1.0, "siglip2": 1.0, "dfn": 1.0},
    {"name": "rrf_pe_primary", "constant": 20, "pe": 3.0, "siglip2": 0.75, "dfn": 1.0},
    {"name": "rrf_pe_conservative", "constant": 60, "pe": 3.0, "siglip2": 0.5, "dfn": 1.0},
    {"name": "rrf_map_candidate", "constant": 10, "pe": 2.0, "siglip2": 0.5, "dfn": 0.75},
]

FUSION_CONFIGS = (
    [("legacy", value) for value in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)]
    + [("calibrated", value) for value in (0.25, 0.5, 1.0, 2.0)]
    + [("rank", value) for value in (0.25, 0.5, 1.0, 2.0)]
)


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rank_matrix(scores: torch.Tensor) -> torch.Tensor:
    order = scores.float().argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(1, scores.size(1) + 1).expand_as(order))
    return ranks


def auxiliary_scores(features: dict, cache: dict) -> torch.Tensor:
    query_ids = [str(value) for value in features["query_image_ids"]]
    if query_ids != [str(value) for value in cache["query_image_ids"]]:
        raise ValueError(f"Query order mismatch for {features.get('model_id')}")
    if features.get("candidate_hash") != cache["metadata"].get("candidate_hash"):
        raise ValueError(f"Candidate hash mismatch for {features.get('model_id')}")
    pos = {str(value): i for i, value in enumerate(features["gallery_ids"])}
    gallery = F.normalize(torch.as_tensor(features["gallery_feats"]).float(), dim=1)
    text = F.normalize(torch.as_tensor(features["txt_feats"]).float(), dim=1)
    output = torch.empty(
        len(query_ids), len(cache["candidate_image_ids"][0]), dtype=torch.float32
    )
    for query_index, row in enumerate(cache["candidate_image_ids"]):
        try:
            indices = torch.tensor([pos[str(value)] for value in row], dtype=torch.long)
        except KeyError as exc:
            raise ValueError(f"Auxiliary feature missing candidate {exc}") from exc
        output[query_index] = gallery[indices] @ text[query_index]
    return output


def retrieval_score_sets(cache: dict, ensemble_mode: str,
                         siglip2_path: str | None, dfn_path: str | None) -> list[dict]:
    pe_scores = torch.as_tensor(cache["pe_scores"]).float()
    sets = []
    if ensemble_mode in {"off", "compare"}:
        sets.append({
            "name": "pe_only", "scores": pe_scores, "constant": None,
            "pe": 1.0, "siglip2": 0.0, "dfn": 0.0,
        })
    if ensemble_mode in {"on", "compare"}:
        if not siglip2_path or not dfn_path:
            raise ValueError("Ensemble mode requires SigLIP2 and DFN feature files")
        sig_scores = auxiliary_scores(load_payload(siglip2_path), cache)
        dfn_scores = auxiliary_scores(load_payload(dfn_path), cache)
        pe_ranks = rank_matrix(pe_scores).float()
        sig_ranks = rank_matrix(sig_scores).float()
        dfn_ranks = rank_matrix(dfn_scores).float()
        for config in RETRIEVAL_CONFIGS:
            constant = float(config["constant"])
            scores = (
                config["pe"] / (constant + pe_ranks)
                + config["siglip2"] / (constant + sig_ranks)
                + config["dfn"] / (constant + dfn_ranks)
            )
            sets.append({**config, "scores": scores})
    return sets


def metric_row(result: dict, retrieval: dict, family: str, weight: float,
               postprocess: str, runtime_seconds: float, baseline_order=None) -> dict:
    metrics = result.get(postprocess, {})
    diagnostics = result.get("diagnostics", {})
    prefix = postprocess
    transition = diagnostics.get(
        f"{prefix}_transitions_vs_reference",
        diagnostics.get("transitions_vs_reference", {}),
    )
    calibration = diagnostics.get("score_calibration", {})
    return {
        "experiment_id": experiment_id(retrieval["name"], family, weight, postprocess),
        "retrieval": retrieval["name"],
        "rrf_constant": retrieval.get("constant"),
        "pe_weight": retrieval.get("pe"),
        "siglip2_weight": retrieval.get("siglip2"),
        "dfn_weight": retrieval.get("dfn"),
        "fusion_family": family,
        "fusion_weight": float(weight),
        "fusion_rank_constant": result.get("rank_constant"),
        "postprocess": postprocess,
        "mAP": metrics.get("mAP"),
        "R@1": metrics.get("R@1"),
        "R@5": metrics.get("R@5"),
        "R@10": metrics.get("R@10"),
        "R@50": metrics.get("R@50"),
        "gt_at_rank2": diagnostics.get(f"{prefix}_gt_at_rank2"),
        "top1_conflicts": diagnostics.get(f"{prefix}_top1_conflicts"),
        "reciprocal_swap_pairs": diagnostics.get(f"{prefix}_reciprocal_swap_pairs"),
        "helped_vs_baseline": transition.get("helped"),
        "harmed_vs_baseline": transition.get("harmed"),
        "top1_changed_by_retrieval": calibration.get("top1_changed_by_stage1"),
        "itm_top2_gap": calibration.get("itm_median_top2_gap"),
        "retrieval_top2_gap": calibration.get("stage1_median_top2_gap"),
        "final_top2_gap": calibration.get("final_median_top2_gap"),
        "itm_min": calibration.get("itm_min"),
        "itm_max": calibration.get("itm_max"),
        "itm_mean": calibration.get("itm_mean"),
        "itm_std": calibration.get("itm_std"),
        "retrieval_min": calibration.get("stage1_min"),
        "retrieval_max": calibration.get("stage1_max"),
        "retrieval_mean": calibration.get("stage1_mean"),
        "retrieval_std": calibration.get("stage1_std"),
        "runtime_seconds": round(float(runtime_seconds), 4),
    }


def add_resource_context(row: dict, args, cache: dict) -> dict:
    row["cache_reused"] = bool(args.cache_reused)
    row["itm_cache_preparation_seconds"] = cache["metadata"].get(
        "preparation_seconds"
    )
    row["peak_vram_mib"] = json.dumps(
        cache["metadata"].get("peak_vram_mib", {}), sort_keys=True
    )
    row["max_rss_mib"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
    )
    return row


def experiment_id(retrieval: str, family: str, weight: float, postprocess: str) -> str:
    value = str(float(weight)).replace(".", "p")
    return f"{retrieval}__{family}_{value}__{postprocess}"


def safe_sort(rows: list[dict], r10_floor: float) -> list[dict]:
    safe = [row for row in rows if row.get("R@10") is not None and row["R@10"] >= r10_floor]
    pool = safe or rows
    return sorted(
        pool,
        key=lambda row: (float(row.get("R@1") or -1), float(row.get("mAP") or -1),
                         float(row.get("R@10") or -1)),
        reverse=True,
    )


def write_answer(path: Path, rows: list[list[str]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(" ".join(str(value) for value in row[:10]) + "\n")


def query_diagnostic_rows(cache: dict, result: dict, postprocess: str) -> list[dict]:
    order = result["orders"][postprocess]
    scores = result["scores"].get(postprocess, result["scores"]["rerank"])
    gallery = cache["gallery_ids"]
    ranks = result.get("ranks", {}).get(postprocess)
    output = []
    for i, row in enumerate(order):
        score_row = scores[i]
        margin = float(score_row[0] - score_row[1]) if score_row.numel() > 1 else math.nan
        output.append({
            "query_index": i,
            "query_id": cache.get("query_labels", cache["query_image_ids"])[i],
            "gt_image_id": cache["query_image_ids"][i],
            "predicted_top1": gallery[int(row[0])],
            "gt_rank": int(ranks[i]) if ranks is not None else None,
            "top1_margin": margin,
        })
    return output


def make_charts(output_dir: Path, sweep_rows: list[dict], finalist_rows: list[dict],
                best_result: dict, best_stage: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; charts skipped")
        return

    legacy = [row for row in sweep_rows if row["fusion_family"] == "legacy"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for retrieval in sorted({row["retrieval"] for row in legacy}):
        rows = sorted((row for row in legacy if row["retrieval"] == retrieval),
                      key=lambda row: row["fusion_weight"])
        axes[0].plot([row["fusion_weight"] for row in rows],
                     [row["R@1"] for row in rows], marker="o", label=retrieval)
        axes[1].plot([row["fusion_weight"] for row in rows],
                     [row["mAP"] for row in rows], marker="o", label=retrieval)
    axes[0].set_title("R@1 vs legacy retrieval weight")
    axes[1].set_title("mAP vs legacy retrieval weight")
    for ax in axes:
        ax.set_xlabel("fusion weight")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fusion_weight_curves.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for family in sorted({row["fusion_family"] for row in sweep_rows}):
        rows = [row for row in sweep_rows if row["fusion_family"] == family]
        ax.scatter([row["R@1"] for row in rows], [row["mAP"] for row in rows],
                   label=family, alpha=0.75)
    ax.set_xlabel("R@1")
    ax.set_ylabel("mAP")
    ax.set_title("Fusion sweep Pareto view")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fusion_pareto.png", dpi=170)
    plt.close(fig)

    ranks = best_result.get("ranks", {}).get(best_stage)
    if ranks is not None:
        values = torch.as_tensor(ranks).long()
        labels = [str(i) for i in range(2, 11)] + ["11+"]
        counts = [int(values.eq(i).sum()) for i in range(2, 11)]
        counts.append(int(values.gt(10).sum()))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(labels, counts)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, count + 0.3, str(count),
                    ha="center", va="bottom")
        ax.set_title(f"GT rank among misses: {best_stage}")
        ax.set_xlabel("GT rank")
        ax.set_ylabel("queries")
        fig.tight_layout()
        fig.savefig(output_dir / "best_gt_rank_distribution.png", dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(finalist_rows)), 4.8))
    names = [row["experiment_id"] for row in finalist_rows]
    helped = [row.get("helped_vs_baseline") or 0 for row in finalist_rows]
    harmed = [row.get("harmed_vs_baseline") or 0 for row in finalist_rows]
    x = torch.arange(len(names)).numpy()
    ax.bar(x - 0.2, helped, 0.4, label="helped")
    ax.bar(x + 0.2, harmed, 0.4, label="harmed")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("Top-1 transitions versus PE-only ITM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "helped_harmed.png", dpi=170)
    plt.close(fig)

    metric_names = ("mAP", "R@1", "R@5", "R@10")
    fig, ax = plt.subplots(figsize=(max(11, 1.6 * len(finalist_rows)), 5.5))
    x = torch.arange(len(finalist_rows)).numpy()
    width = 0.2
    for metric_index, metric in enumerate(metric_names):
        offset = (metric_index - 1.5) * width
        values = [row[metric] for row in finalist_rows]
        ax.bar(x + offset, values, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row["experiment_id"] for row in finalist_rows],
        rotation=25, ha="right", fontsize=8,
    )
    ax.set_ylim(0, 1.05)
    ax.set_title("Finalist metrics by postprocessor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(output_dir / "finalist_stage_metrics.png", dpi=170)
    plt.close(fig)


def run_oldtest(args, cache: dict, retrieval_sets: list[dict]):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_retrieval = next(
        (item for item in retrieval_sets if item["name"] == "pe_only"),
        {"name": "pe_only", "scores": cache["pe_scores"], "constant": None,
         "pe": 1.0, "siglip2": 0.0, "dfn": 0.0},
    )
    baseline_result = evaluate_cached_rerank(
        cache, baseline_retrieval["scores"], "legacy", 0.0,
        postprocesses=("rerank",), include_top10=False,
    )
    baseline_order = baseline_result["orders"]["rerank"]
    baseline_row = add_resource_context(metric_row(
        baseline_result, baseline_retrieval, "legacy", 0.0, "rerank", 0.0
    ), args, cache)
    r10_floor = float(baseline_row["R@10"]) - float(args.r10_tolerance)

    retrieval_rows = []
    for retrieval in retrieval_sets:
        result = evaluate_cached_rerank(
            cache, retrieval["scores"], "legacy", 0.0,
            postprocesses=("rerank",), reference_order=baseline_order,
            include_top10=False,
        )
        row = add_resource_context(
            metric_row(result, retrieval, "legacy", 0.0, "rerank", 0.0),
            args, cache,
        )
        retrieval_metrics = result["retrieval"]
        for key, value in retrieval_metrics.items():
            row[f"retrieval_{key}"] = value
        retrieval_rows.append(row)
    write_csv(output_dir / "retrieval_screen.csv", retrieval_rows)

    sweep_rows = [baseline_row]
    for retrieval in retrieval_sets:
        for family, weight in FUSION_CONFIGS:
            started = time.perf_counter()
            result = evaluate_cached_rerank(
                cache, retrieval["scores"], family, weight,
                postprocesses=("rerank",), reference_order=baseline_order,
                include_top10=False,
            )
            row = add_resource_context(metric_row(
                result, retrieval, family, weight, "rerank",
                time.perf_counter() - started,
            ), args, cache)
            sweep_rows.append(row)
    write_csv(output_dir / "fusion_sweep.csv", sweep_rows)

    finalists = safe_sort(sweep_rows, r10_floor)[:int(args.finalists)]
    retrieval_by_name = {item["name"]: item for item in retrieval_sets}
    finalist_rows, finalist_results = [], {}
    for finalist in finalists:
        retrieval = retrieval_by_name.get(finalist["retrieval"], baseline_retrieval)
        started = time.perf_counter()
        result = evaluate_cached_rerank(
            cache, retrieval["scores"], finalist["fusion_family"],
            float(finalist["fusion_weight"]),
            postprocesses=("rerank", "greedy_sca", "gale_shapley"),
            reference_order=baseline_order,
        )
        elapsed = time.perf_counter() - started
        for stage in ("rerank", "greedy_sca", "gale_shapley"):
            row = add_resource_context(metric_row(
                result, retrieval, finalist["fusion_family"],
                float(finalist["fusion_weight"]), stage, elapsed,
            ), args, cache)
            finalist_rows.append(row)
            finalist_results[row["experiment_id"]] = (result, stage)
            write_answer(
                output_dir / f"answer_{row['experiment_id']}.txt",
                result["top10_by_stage"][stage],
            )
            write_csv(
                output_dir / f"queries_{row['experiment_id']}.csv",
                query_diagnostic_rows(cache, result, stage),
            )
    write_csv(output_dir / "finalist_results.csv", finalist_rows)

    best_row = safe_sort(finalist_rows, r10_floor)[0]
    best_result, best_stage = finalist_results[best_row["experiment_id"]]
    best_retrieval = retrieval_by_name.get(best_row["retrieval"], baseline_retrieval)
    best_config = {
        "selection_policy": "R@1, then mAP, then R@10",
        "r10_floor": r10_floor,
        "topk": int(cache["metadata"]["topk"]),
        "candidate_hash": cache["metadata"]["candidate_hash"],
        "retrieval": {key: best_retrieval.get(key) for key in
                      ("name", "constant", "pe", "siglip2", "dfn")},
        "fusion_family": best_row["fusion_family"],
        "fusion_weight": float(best_row["fusion_weight"]),
        "postprocess": best_stage,
        "metrics": {key: best_row[key] for key in ("mAP", "R@1", "R@5", "R@10", "R@50")},
    }
    save_json(output_dir / "best_inference_config.json", best_config)
    torch.save({"result": best_result, "stage": best_stage, "config": best_config},
               output_dir / "best_result.pt")
    write_answer(output_dir / "answer.txt", best_result["top10_by_stage"][best_stage])
    make_charts(output_dir, sweep_rows, finalist_rows, best_result, best_stage)

    metrics = {
        "mode": "oldtest_eval",
        "ensemble_mode": args.ensemble_mode,
        "cache_reused": bool(args.cache_reused),
        "baseline": baseline_row,
        "r10_floor": r10_floor,
        "best": best_row,
        "best_delta_vs_pe_itm": {
            key: float(best_row[key]) - float(baseline_row[key])
            for key in ("mAP", "R@1", "R@5", "R@10", "R@50")
        },
        "top_rerank_settings": safe_sort(sweep_rows, r10_floor)[:10],
        "finalists": finalist_rows,
        "retrieval_screen": retrieval_rows,
        "resource": {
            "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "itm_cache_preparation_seconds": cache["metadata"].get("preparation_seconds"),
            "itm_cache_peak_vram_mib": cache["metadata"].get("peak_vram_mib", {}),
            "sweep_rows": len(sweep_rows),
            "finalist_rows": len(finalist_rows),
        },
    }
    save_json(output_dir / "metrics.json", metrics)
    make_report_archive(output_dir)
    print("best configuration:")
    print(json.dumps(best_config, indent=2))


def run_official(args, cache: dict, retrieval_sets: list[dict]):
    if not args.best_config:
        raise ValueError("official_submit requires --best-config")
    config = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    if int(config["topk"]) != int(cache["metadata"]["topk"]):
        raise ValueError(
            f"Frozen configuration K={config['topk']} does not match "
            f"official cache K={cache['metadata']['topk']}"
        )
    retrieval_name = config["retrieval"]["name"]
    retrieval = next((item for item in retrieval_sets if item["name"] == retrieval_name), None)
    if retrieval is None and retrieval_name == "pe_only":
        retrieval = {
            "name": "pe_only", "scores": cache["pe_scores"], "constant": None,
            "pe": 1.0, "siglip2": 0.0, "dfn": 0.0,
        }
    if retrieval is None:
        raise ValueError(f"Best configuration requires unavailable retrieval {retrieval_name}")
    for key in ("constant", "pe", "siglip2", "dfn"):
        frozen = config["retrieval"].get(key)
        available = retrieval.get(key)
        if frozen is None and available is None:
            continue
        if float(frozen) != float(available):
            raise ValueError(
                f"Frozen retrieval {retrieval_name} has {key}={frozen}, "
                f"but the current implementation provides {available}"
            )
    result = evaluate_cached_rerank(
        cache, retrieval["scores"], config["fusion_family"],
        float(config["fusion_weight"]),
        postprocesses=(config["postprocess"],),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "best_inference_config.json", config)
    write_answer(output_dir / "answer.txt",
                 result["top10_by_stage"][config["postprocess"]])
    torch.save({"result": result, "stage": config["postprocess"], "config": config},
               output_dir / "best_result.pt")
    save_json(output_dir / "metrics.json", {
        "mode": "official_submit", "configuration": config,
        "num_queries": len(result["top10_by_stage"][config["postprocess"]]),
    })
    make_report_archive(output_dir)


def make_report_archive(output_dir: Path):
    temporary_base = output_dir.parent / f".{output_dir.name}_inference_report"
    target = output_dir / "inference_report.zip"
    if target.exists():
        target.unlink()
    archive = Path(shutil.make_archive(str(temporary_base), "zip", root_dir=output_dir))
    archive.replace(target)
    print(f"report archive: {target}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("oldtest_eval", "official_submit"), required=True)
    parser.add_argument("--ensemble-mode", choices=("off", "on", "compare"), default="compare")
    parser.add_argument("--siglip2")
    parser.add_argument("--dfn")
    parser.add_argument("--best-config")
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--r10-tolerance", type=float, default=0.001)
    parser.add_argument("--cache-reused", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cache = load_payload(args.cache)
    if args.mode == "oldtest_eval" and not cache["metadata"].get("has_ground_truth"):
        raise ValueError("oldtest_eval requires a cache with ground truth")
    retrieval_sets = retrieval_score_sets(
        cache, args.ensemble_mode, args.siglip2, args.dfn
    )
    if args.mode == "oldtest_eval":
        run_oldtest(args, cache, retrieval_sets)
    else:
        run_official(args, cache, retrieval_sets)


if __name__ == "__main__":
    main()
