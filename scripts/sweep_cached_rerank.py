#!/usr/bin/env python3
"""CPU-only fusion sweep over a cached X-VLM Top-K ITM matrix."""
from __future__ import annotations

import argparse
import csv
import json
import math
import resource
import shutil
import statistics
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
    {"name": "rrf_map_c5", "constant": 5, "pe": 2.0, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_map_c7p5", "constant": 7.5, "pe": 2.0, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_map_c15", "constant": 15, "pe": 2.0, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_map_c20", "constant": 20, "pe": 2.0, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_pe_2p25", "constant": 10, "pe": 2.25, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_pe_2p5", "constant": 10, "pe": 2.5, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_pe_2p75", "constant": 10, "pe": 2.75, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_pe_3", "constant": 10, "pe": 3.0, "siglip2": 0.5, "dfn": 0.75},
    {"name": "rrf_sig_light", "constant": 10, "pe": 2.0, "siglip2": 0.4, "dfn": 0.75},
    {"name": "rrf_dfn_light", "constant": 10, "pe": 2.0, "siglip2": 0.5, "dfn": 0.6},
    {"name": "rrf_aux_light", "constant": 10, "pe": 3.0, "siglip2": 0.25, "dfn": 0.5},
]

FUSION_CONFIGS = (
    [("legacy", value) for value in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)]
    + [("calibrated", value) for value in
       (0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
        1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0)]
    + [("adaptive", value) for value in (0.5, 0.75, 1.0, 1.25, 1.5)]
    + [("rank", value) for value in (0.5, 1.0, 2.0)]
)

POSTPROCESS_CONFIGS = [
    {"label": "rerank", "stage": "rerank", "params": {}},
    {"label": "greedy_sca", "stage": "greedy_sca", "params": {}},
    {"label": "gale_shapley", "stage": "gale_shapley", "params": {}},
    *[
        {
            "label": f"locked_gs_m{str(margin).replace('.', 'p')}",
            "stage": "locked_gale_shapley",
            "params": {"lock_margin": margin},
        }
        for margin in (
            0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25,
            0.3, 0.35, 0.4, 0.5, 0.75, 1.0,
        )
    ],
    *[
        {
            "label": f"consensus_locked_gs_m{str(margin).replace('.', 'p')}",
            "stage": "locked_gale_shapley",
            "params": {
                "lock_margin": margin,
                "require_component_agreement": True,
            },
        }
        for margin in (
            0.05, 0.1, 0.15, 0.2, 0.25, 0.275,
            0.3, 0.325, 0.35, 0.4, 0.5,
        )
    ],
    {
        "label": "gs_cycle_strict",
        "stage": "gale_shapley_cycle_rescue",
        "params": {
            "text_sim_threshold": 0.96, "candidate_depth": 2,
            "max_final_penalty": 0.20, "min_retrieval_gain": 0.05,
        },
    },
    {
        "label": "gs_cycle_balanced",
        "stage": "gale_shapley_cycle_rescue",
        "params": {
            "text_sim_threshold": 0.94, "candidate_depth": 3,
            "max_final_penalty": 0.35, "min_retrieval_gain": 0.05,
        },
    },
    {
        "label": "gs_cycle_wide",
        "stage": "gale_shapley_cycle_rescue",
        "params": {
            "text_sim_threshold": 0.90, "candidate_depth": 3,
            "max_final_penalty": 0.50, "min_retrieval_gain": 0.025,
        },
    },
    {
        "label": "gs_cycle_retrieval_strong",
        "stage": "gale_shapley_cycle_rescue",
        "params": {
            "text_sim_threshold": 0.94, "candidate_depth": 3,
            "max_final_penalty": 0.50, "min_retrieval_gain": 0.10,
        },
    },
]

# Always postprocess the strongest K=50 discoveries even if K=100 shifts them just
# outside the rerank-only Top-N. This prevents the two-stage search from pruning the
# configuration whose matching method is known to work best.
SEEDED_FINALISTS = [
    {"retrieval": "rrf_map_candidate", "family": "calibrated", "weight": 0.7},
    {"retrieval": "rrf_pe_2p5", "family": "legacy", "weight": 4.0},
    {"retrieval": "rrf_map_c5", "family": "adaptive", "weight": 0.5},
]


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


def contiguous_fold_metrics(ranks, folds: int = 5) -> dict[str, float | int | None]:
    """Robustness proxy that keeps neighboring paired queries in the same fold."""
    if ranks is None:
        return {
            "cv_folds": int(folds), "cv_R1_mean": None, "cv_R1_std": None,
            "cv_R1_worst": None, "cv_mAP_mean": None, "cv_mAP_std": None,
        }
    values = torch.as_tensor(ranks).long()
    chunks = [chunk for chunk in torch.tensor_split(values, int(folds)) if chunk.numel()]
    r1 = torch.tensor([float(chunk.eq(1).float().mean()) for chunk in chunks])
    maps = torch.tensor([float((1.0 / chunk.float()).mean()) for chunk in chunks])
    return {
        "cv_folds": len(chunks),
        "cv_R1_mean": float(r1.mean()),
        "cv_R1_std": float(r1.std(unbiased=False)),
        "cv_R1_worst": float(r1.min()),
        "cv_mAP_mean": float(maps.mean()),
        "cv_mAP_std": float(maps.std(unbiased=False)),
    }


def metric_row(result: dict, retrieval: dict, family: str, weight: float,
               postprocess: str, runtime_seconds: float, baseline_order=None,
               postprocess_label: str | None = None,
               postprocess_params: dict | None = None) -> dict:
    label = postprocess_label or postprocess
    metrics = result.get(postprocess, {})
    diagnostics = result.get("diagnostics", {})
    prefix = postprocess
    transition = diagnostics.get(
        f"{prefix}_transitions_vs_reference",
        diagnostics.get("transitions_vs_reference", {}),
    )
    calibration = diagnostics.get("score_calibration", {})
    robustness = contiguous_fold_metrics(
        result.get("ranks", {}).get(postprocess), folds=5
    )
    return {
        "experiment_id": experiment_id(retrieval["name"], family, weight, label),
        "retrieval": retrieval["name"],
        "rrf_constant": retrieval.get("constant"),
        "pe_weight": retrieval.get("pe"),
        "siglip2_weight": retrieval.get("siglip2"),
        "dfn_weight": retrieval.get("dfn"),
        "fusion_family": family,
        "fusion_weight": float(weight),
        "fusion_rank_constant": result.get("rank_constant"),
        "postprocess": label,
        "postprocess_stage": postprocess,
        "postprocess_params": json.dumps(postprocess_params or {}, sort_keys=True),
        "mAP": metrics.get("mAP"),
        "R@1": metrics.get("R@1"),
        "R@5": metrics.get("R@5"),
        "R@10": metrics.get("R@10"),
        "R@50": metrics.get("R@50"),
        "gt_at_rank2": diagnostics.get(f"{prefix}_gt_at_rank2"),
        "top1_conflicts": diagnostics.get(f"{prefix}_top1_conflicts"),
        "reciprocal_swap_pairs": diagnostics.get(f"{prefix}_reciprocal_swap_pairs"),
        "cycle_candidates": diagnostics.get(f"{prefix}_cycle_candidates"),
        "cycle_swaps": diagnostics.get(f"{prefix}_cycle_swaps"),
        "consensus_queries": diagnostics.get(f"{prefix}_consensus_queries"),
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
        "adaptive_weight_min": calibration.get("stage1_weight_min"),
        "adaptive_weight_mean": calibration.get("stage1_weight_mean"),
        "adaptive_weight_max": calibration.get("stage1_weight_max"),
        **robustness,
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


def pareto_frontier(rows: list[dict], metrics=("R@1", "mAP", "R@10")) -> list[dict]:
    """Return non-dominated settings for the requested maximize-only metrics."""
    valid = [row for row in rows if all(row.get(key) is not None for key in metrics)]
    frontier = []
    for row in valid:
        dominated = False
        for other in valid:
            if other is row:
                continue
            at_least = all(float(other[key]) >= float(row[key]) for key in metrics)
            strict = any(float(other[key]) > float(row[key]) for key in metrics)
            if at_least and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return safe_sort(frontier, r10_floor=-float("inf"))


def write_answer(path: Path, rows: list[list[str]]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(" ".join(str(value) for value in row[:10]) + "\n")


def query_diagnostic_rows(cache: dict, result: dict, postprocess: str) -> list[dict]:
    order = result["orders"][postprocess]
    scores = result["scores"].get(postprocess, result["scores"]["rerank"])
    retrieval_order = result["orders"]["retrieval"]
    itm_order = result["orders"]["itm"]
    rerank_order = result["orders"]["rerank"]
    retrieval_scores = result["scores"]["retrieval"]
    itm_scores = result["scores"]["itm"]
    rerank_scores = result["scores"]["rerank"]
    gallery = cache["gallery_ids"]
    ranks = result.get("ranks", {}).get(postprocess)
    retrieval_ranks = result.get("ranks", {}).get("retrieval")
    itm_ranks = result.get("ranks", {}).get("itm")
    rerank_ranks = result.get("ranks", {}).get("rerank")
    output = []
    for i, row in enumerate(order):
        score_row = scores[i]
        margin = float(score_row[0] - score_row[1]) if score_row.numel() > 1 else math.nan
        retrieval_margin = float(retrieval_scores[i, 0] - retrieval_scores[i, 1])
        itm_margin = float(itm_scores[i, 0] - itm_scores[i, 1])
        rerank_margin = float(rerank_scores[i, 0] - rerank_scores[i, 1])
        output.append({
            "query_index": i,
            "query_id": cache.get("query_labels", cache["query_image_ids"])[i],
            "gt_image_id": cache["query_image_ids"][i],
            "predicted_top1": gallery[int(row[0])],
            "gt_rank": int(ranks[i]) if ranks is not None else None,
            "top1_margin": margin,
            "retrieval_top1": gallery[int(retrieval_order[i, 0])],
            "itm_top1": gallery[int(itm_order[i, 0])],
            "rerank_top1": gallery[int(rerank_order[i, 0])],
            "retrieval_gt_rank": (
                int(retrieval_ranks[i]) if retrieval_ranks is not None else None
            ),
            "itm_gt_rank": int(itm_ranks[i]) if itm_ranks is not None else None,
            "rerank_gt_rank": (
                int(rerank_ranks[i]) if rerank_ranks is not None else None
            ),
            "retrieval_top2_margin": retrieval_margin,
            "itm_top2_margin": itm_margin,
            "rerank_top2_margin": rerank_margin,
            "retrieval_itm_agree": bool(
                int(retrieval_order[i, 0]) == int(itm_order[i, 0])
            ),
            "retrieval_final_agree": bool(
                int(retrieval_order[i, 0]) == int(row[0])
            ),
            "gt_in_candidates": bool(
                int(cache["gt_pos"][i]) in set(
                    int(value) for value in cache["candidate_indices"][i]
                )
            ) if cache.get("metadata", {}).get("has_ground_truth") else None,
        })
    return output


def summarize_failure_cases(rows: list[dict], result: dict, stage: str) -> dict:
    ranks = [int(row["gt_rank"]) for row in rows if row.get("gt_rank") is not None]
    misses = [row for row in rows if row.get("gt_rank") is not None and int(row["gt_rank"]) > 1]
    rank2 = [row for row in rows if row.get("gt_rank") is not None and int(row["gt_rank"]) == 2]

    def count(predicate, values):
        return sum(bool(predicate(row)) for row in values)

    def margin_summary(values):
        margins = [float(row["top1_margin"]) for row in values if math.isfinite(float(row["top1_margin"]))]
        return {
            "count": len(margins),
            "median": statistics.median(margins) if margins else None,
            "mean": statistics.mean(margins) if margins else None,
        }

    bins = {
        "rank1": sum(rank == 1 for rank in ranks),
        "rank2": sum(rank == 2 for rank in ranks),
        "rank3_5": sum(3 <= rank <= 5 for rank in ranks),
        "rank6_10": sum(6 <= rank <= 10 for rank in ranks),
        "rank11_50": sum(11 <= rank <= 50 for rank in ranks),
        "above50": sum(rank > 50 for rank in ranks),
    }
    diagnostics = result.get("diagnostics", {})
    output = {
        "stage": stage,
        "queries": len(rows),
        "misses": len(misses),
        "rank_bins": bins,
        "rank2_share_of_misses": len(rank2) / len(misses) if misses else 0.0,
        "candidate_misses": count(lambda row: not row.get("gt_in_candidates", False), rows),
        "component_disagreement": {
            "all_retrieval_vs_itm": count(lambda row: not row["retrieval_itm_agree"], rows),
            "miss_retrieval_vs_itm": count(lambda row: not row["retrieval_itm_agree"], misses),
            "rank2_retrieval_vs_itm": count(lambda row: not row["retrieval_itm_agree"], rank2),
        },
        "rank2_rescue_opportunities": {
            "retrieval_already_correct": count(
                lambda row: int(row["retrieval_gt_rank"]) == 1, rank2
            ),
            "itm_already_correct": count(lambda row: int(row["itm_gt_rank"]) == 1, rank2),
            "fused_rerank_already_correct": count(
                lambda row: int(row["rerank_gt_rank"]) == 1, rank2
            ),
        },
        "margin": {
            "correct": margin_summary([row for row in rows if int(row["gt_rank"]) == 1]),
            "miss": margin_summary(misses),
            "rank2": margin_summary(rank2),
        },
        "uncertainty_gates": {},
        "top1_conflicts": diagnostics.get(f"{stage}_top1_conflicts"),
        "reciprocal_swap_pairs": diagnostics.get(f"{stage}_reciprocal_swap_pairs"),
        "cycle_candidates": diagnostics.get(f"{stage}_cycle_candidates"),
        "cycle_swaps": diagnostics.get(f"{stage}_cycle_swaps"),
    }
    for threshold in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        flagged = [row for row in rows if float(row["top1_margin"]) <= threshold]
        caught = count(lambda row: int(row["gt_rank"]) > 1, flagged)
        output["uncertainty_gates"][str(threshold)] = {
            "flagged": len(flagged),
            "errors_caught": caught,
            "error_recall": caught / len(misses) if misses else 0.0,
            "flag_precision": caught / len(flagged) if flagged else 0.0,
        }
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

    locked_rows = [
        row for row in finalist_rows
        if row.get("postprocess_stage") == "locked_gale_shapley"
    ]
    if locked_rows:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        groups = {}
        for row in locked_rows:
            params = json.loads(row["postprocess_params"])
            mode = "consensus" if params.get("require_component_agreement") else "all"
            key = (
                f"{row['retrieval']} | {row['fusion_family']} "
                f"{row['fusion_weight']} | {mode}"
            )
            groups.setdefault(key, []).append(row)
        for label, rows in groups.items():
            rows = sorted(
                rows,
                key=lambda row: json.loads(row["postprocess_params"])["lock_margin"],
            )
            x_values = [json.loads(row["postprocess_params"])["lock_margin"] for row in rows]
            ax.plot(x_values, [row["R@1"] for row in rows], marker="o", label=label)
        ax.set_xlabel("confidence lock margin")
        ax.set_ylabel("R@1")
        ax.set_title("Confidence-locked Gale-Shapley sweep")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "lock_margin_curves.png", dpi=170)
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

    chart_finalists = safe_sort(finalist_rows, r10_floor=-float("inf"))[:15]
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(chart_finalists)), 5.2))
    names = [row["experiment_id"] for row in chart_finalists]
    helped = [row.get("helped_vs_baseline") or 0 for row in chart_finalists]
    harmed = [row.get("harmed_vs_baseline") or 0 for row in chart_finalists]
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
    fig, ax = plt.subplots(figsize=(max(11, 0.9 * len(chart_finalists)), 5.8))
    x = torch.arange(len(chart_finalists)).numpy()
    width = 0.2
    for metric_index, metric in enumerate(metric_names):
        offset = (metric_index - 1.5) * width
        values = [row[metric] for row in chart_finalists]
        ax.bar(x + offset, values, width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [row["experiment_id"] for row in chart_finalists],
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
    finalist_ids = {row["experiment_id"] for row in finalists}
    for seed in SEEDED_FINALISTS:
        match = next((
            row for row in sweep_rows
            if row["retrieval"] == seed["retrieval"]
            and row["fusion_family"] == seed["family"]
            and float(row["fusion_weight"]) == float(seed["weight"])
        ), None)
        if match is not None and match["experiment_id"] not in finalist_ids:
            finalists.append(match)
            finalist_ids.add(match["experiment_id"])
    retrieval_by_name = {item["name"]: item for item in retrieval_sets}
    finalist_rows, finalist_specs = [], {}
    for finalist in finalists:
        retrieval = retrieval_by_name.get(finalist["retrieval"], baseline_retrieval)
        for post_cfg in POSTPROCESS_CONFIGS:
            stage = post_cfg["stage"]
            label = post_cfg["label"]
            params = dict(post_cfg["params"])
            started = time.perf_counter()
            result = evaluate_cached_rerank(
                cache, retrieval["scores"], finalist["fusion_family"],
                float(finalist["fusion_weight"]),
                postprocesses=(stage,), postprocess_options=params,
                reference_order=baseline_order,
                include_top10=False,
            )
            elapsed = time.perf_counter() - started
            row = add_resource_context(metric_row(
                result, retrieval, finalist["fusion_family"],
                float(finalist["fusion_weight"]), stage, elapsed,
                postprocess_label=label, postprocess_params=params,
            ), args, cache)
            finalist_rows.append(row)
            finalist_specs[row["experiment_id"]] = (
                retrieval, finalist["fusion_family"],
                float(finalist["fusion_weight"]), stage, params, label,
            )
    write_csv(output_dir / "finalist_results.csv", finalist_rows)
    write_csv(output_dir / "pareto_settings.csv", pareto_frontier(finalist_rows))

    best_row = safe_sort(finalist_rows, r10_floor)[0]
    def evaluate_spec(row):
        retrieval, family, weight, stage, params, label = finalist_specs[
            row["experiment_id"]
        ]
        result = evaluate_cached_rerank(
            cache, retrieval["scores"], family, weight,
            postprocesses=(stage,), postprocess_options=params,
            reference_order=baseline_order,
        )
        return result, stage, params, label

    best_result, best_stage, best_postprocess_params, best_postprocess_label = (
        evaluate_spec(best_row)
    )
    def make_config(row, stage, params, label, floor, policy):
        retrieval = retrieval_by_name.get(row["retrieval"], baseline_retrieval)
        return {
            "selection_policy": policy,
            "r10_floor": floor,
            "topk": int(cache["metadata"]["topk"]),
            "candidate_hash": cache["metadata"]["candidate_hash"],
            "retrieval": {key: retrieval.get(key) for key in
                          ("name", "constant", "pe", "siglip2", "dfn")},
            "fusion_family": row["fusion_family"],
            "fusion_weight": float(row["fusion_weight"]),
            "postprocess": stage,
            "postprocess_label": label,
            "postprocess_params": params,
            "metrics": {key: row[key] for key in ("mAP", "R@1", "R@5", "R@10", "R@50")},
        }

    best_config = make_config(
        best_row, best_stage, best_postprocess_params, best_postprocess_label,
        r10_floor, "R@1, then mAP, then R@10",
    )
    save_json(output_dir / "best_inference_config.json", best_config)
    torch.save({"result": best_result, "stage": best_stage, "config": best_config},
               output_dir / "best_result.pt")
    write_answer(output_dir / "answer.txt", best_result["top10_by_stage"][best_stage])

    peak_r10 = max(float(row["R@10"]) for row in finalist_rows)
    conservative_floor = peak_r10 - float(args.r10_tolerance)
    conservative_row = safe_sort(finalist_rows, conservative_floor)[0]
    if conservative_row["experiment_id"] == best_row["experiment_id"]:
        conservative_result = best_result
        conservative_stage = best_stage
        conservative_params = best_postprocess_params
        conservative_label = best_postprocess_label
    else:
        conservative_result, conservative_stage, conservative_params, conservative_label = (
            evaluate_spec(conservative_row)
        )
    conservative_config = make_config(
        conservative_row, conservative_stage, conservative_params, conservative_label,
        conservative_floor,
        "R@1 with R@10 within tolerance of the best finalist R@10",
    )
    save_json(output_dir / "best_conservative_config.json", conservative_config)
    write_answer(
        output_dir / "answer_conservative.txt",
        conservative_result["top10_by_stage"][conservative_stage],
    )
    write_csv(
        output_dir / "best_conservative_queries.csv",
        query_diagnostic_rows(cache, conservative_result, conservative_stage),
    )
    best_query_rows = query_diagnostic_rows(cache, best_result, best_stage)
    write_csv(output_dir / "best_queries.csv", best_query_rows)
    write_csv(
        output_dir / "best_rank2_cases.csv",
        [row for row in best_query_rows if int(row["gt_rank"]) == 2],
    )
    failure_analysis = summarize_failure_cases(
        best_query_rows, best_result, best_stage
    )
    save_json(output_dir / "failure_analysis.json", failure_analysis)
    make_charts(output_dir, sweep_rows, finalist_rows, best_result, best_stage)

    metrics = {
        "mode": "oldtest_eval",
        "ensemble_mode": args.ensemble_mode,
        "cache_reused": bool(args.cache_reused),
        "baseline": baseline_row,
        "r10_floor": r10_floor,
        "best": best_row,
        "best_conservative": conservative_row,
        "conservative_r10_floor": conservative_floor,
        "best_delta_vs_pe_itm": {
            key: float(best_row[key]) - float(baseline_row[key])
            for key in ("mAP", "R@1", "R@5", "R@10", "R@50")
        },
        "top_rerank_settings": safe_sort(sweep_rows, r10_floor)[:10],
        "finalists": finalist_rows,
        "retrieval_screen": retrieval_rows,
        "failure_analysis": failure_analysis,
        "search_space": {
            "retrieval_configs": [
                {key: value for key, value in item.items() if key != "scores"}
                for item in retrieval_sets
            ],
            "fusion_configs": [
                {"family": family, "weight": weight}
                for family, weight in FUSION_CONFIGS
            ],
            "postprocess_configs": POSTPROCESS_CONFIGS,
        },
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
        postprocess_options=config.get("postprocess_params") or {},
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
