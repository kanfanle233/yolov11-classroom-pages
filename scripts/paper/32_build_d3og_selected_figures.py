#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


RUN_ORDER = ["behavior_aug_v1", "mainline_v3", "random6_baseline", "yolo11x_object"]
RUN_DISPLAY = {
    "behavior_aug_v1": "BehaviorAug-v1",
    "mainline_v3": "Mainline-v3",
    "random6_baseline": "Random6-Baseline",
    "yolo11x_object": "YOLO11x-Object",
}
RUN_COLORS = {
    "behavior_aug_v1": "#0b7285",
    "mainline_v3": "#2f9e44",
    "random6_baseline": "#e67700",
    "yolo11x_object": "#5f3dc4",
}


def _resolve_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (root / p).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        if required:
            raise
        return pd.DataFrame()


def _choose_device(requested: str) -> Tuple[str, Dict[str, object]]:
    requested = (requested or "auto").lower()
    info: Dict[str, object] = {
        "requested_device": requested,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
    }
    if requested == "cpu":
        info["resolved_device"] = "cpu"
        return "cpu", info
    if requested == "cuda":
        if torch.cuda.is_available():
            idx = int(torch.cuda.current_device())
            info["resolved_device"] = "cuda"
            info["cuda_device_name"] = torch.cuda.get_device_name(idx)
            return "cuda", info
        info["resolved_device"] = "cpu"
        info["fallback_reason"] = "cuda_requested_but_unavailable"
        return "cpu", info
    if torch.cuda.is_available():
        idx = int(torch.cuda.current_device())
        info["resolved_device"] = "cuda"
        info["cuda_device_name"] = torch.cuda.get_device_name(idx)
        return "cuda", info
    info["resolved_device"] = "cpu"
    info["fallback_reason"] = "auto_no_cuda"
    return "cpu", info


def _bootstrap_ci(values: Sequence[float], seed: int = 20260422, n_boot: int = 2000) -> Tuple[float, float, float]:
    arr = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = arr[idx].mean(axis=1)
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return (mean, lo, hi)


def _gpu_histogram(values: np.ndarray, bins: np.ndarray, device: str, gpu_chunk_size: int) -> Tuple[np.ndarray, Dict[str, object]]:
    values = np.asarray(values, dtype=np.float32)
    bins = np.asarray(bins, dtype=np.float32)
    log: Dict[str, object] = {"used_cuda": False, "fallback_reason": None}
    if values.size == 0:
        return np.zeros(max(0, len(bins) - 1), dtype=np.float32), log
    if device == "cuda":
        try:
            edges = torch.tensor(bins, device="cuda", dtype=torch.float32)
            hist = torch.zeros(len(bins) - 1, device="cuda", dtype=torch.float32)
            chunk = max(1, int(gpu_chunk_size))
            for s in range(0, values.size, chunk):
                e = min(values.size, s + chunk)
                part = torch.tensor(values[s:e], device="cuda", dtype=torch.float32)
                part = part.clamp(min=float(bins[0]), max=float(np.nextafter(bins[-1], bins[0])))
                idx = torch.bucketize(part, edges, right=False) - 1
                idx = idx.clamp(0, len(bins) - 2)
                hist += torch.bincount(idx, minlength=len(bins) - 1).to(torch.float32)
            denom = float(values.size) * np.maximum(np.diff(bins), 1e-12)
            density = hist.detach().cpu().numpy() / denom
            log["used_cuda"] = True
            return density.astype(np.float32), log
        except Exception as exc:
            log["fallback_reason"] = f"cuda_hist_failed:{type(exc).__name__}"
    counts, _ = np.histogram(values, bins=bins, density=True)
    return counts.astype(np.float32), log


def _gpu_center_scale(mat: np.ndarray, device: str, gpu_chunk_size: int) -> Tuple[np.ndarray, Dict[str, object]]:
    arr = np.asarray(mat, dtype=np.float32)
    log: Dict[str, object] = {"used_cuda": False, "fallback_reason": None}
    if arr.size == 0:
        return arr, log
    if device == "cuda":
        try:
            t = torch.tensor(arr, device="cuda", dtype=torch.float32)
            absmax = torch.nan_to_num(torch.abs(t), nan=0.0).max()
            if float(absmax) <= 1e-8:
                return np.zeros_like(arr), log
            scaled = (t / absmax).detach().cpu().numpy().astype(np.float32)
            log["used_cuda"] = True
            return scaled, log
        except Exception as exc:
            log["fallback_reason"] = f"cuda_scale_failed:{type(exc).__name__}"
    absmax = float(np.nanmax(np.abs(arr)))
    if absmax <= 1e-8:
        return np.zeros_like(arr), log
    return (arr / absmax).astype(np.float32), log


def _gpu_colwise_center_scale(mat: np.ndarray, device: str) -> Tuple[np.ndarray, Dict[str, object]]:
    arr = np.asarray(mat, dtype=np.float32)
    log: Dict[str, object] = {"used_cuda": False, "fallback_reason": None}
    if arr.size == 0:
        return arr, log
    if device == "cuda":
        try:
            t = torch.tensor(arr, device="cuda", dtype=torch.float32)
            denom = torch.nan_to_num(torch.abs(t), nan=0.0).max(dim=0).values
            denom = torch.where(denom <= 1e-8, torch.ones_like(denom), denom)
            out = (t / denom).detach().cpu().numpy().astype(np.float32)
            log["used_cuda"] = True
            return out, log
        except Exception as exc:
            log["fallback_reason"] = f"cuda_colscale_failed:{type(exc).__name__}"
    denom = np.nanmax(np.abs(arr), axis=0)
    denom = np.where(denom <= 1e-8, 1.0, denom)
    return (arr / denom).astype(np.float32), log


def _clean_run_frame(df: pd.DataFrame, run_name: str) -> pd.DataFrame:
    out = df.copy()
    out["run_name"] = run_name
    for c in [
        "duration_sec",
        "elapsed_sec",
        "align_avg_candidates",
        "verified_p_match_mean",
        "event_queries_count",
        "verified_count",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _family_from_title(title: str) -> str:
    t = (title or "").lower()
    table = [
        ("bar", ["bar", "bullet", "column", "histogram"]),
        ("line", ["line", "slope", "stream"]),
        ("area", ["area"]),
        ("scatter", ["scatter", "bubble", "dot", "voronoi", "hexbin"]),
        ("heatmap", ["heatmap", "calendar", "matrix"]),
        ("network", ["network", "graph", "force", "chord", "sankey", "bundle"]),
        ("hierarchy", ["tree", "treemap", "sunburst", "pack", "hierarchical"]),
        ("map", ["map", "projection", "geo", "cartogram"]),
        ("pie", ["pie", "donut"]),
    ]
    for fam, keys in table:
        if any(k in t for k in keys):
            return fam
    return "other"


def _fetch_html(url: str, timeout: int = 25, retries: int = 4, backoff: float = 1.2) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; YOLOv11-paper-audit/1.0)"}
    last_exc: Optional[Exception] = None
    for i in range(max(1, retries)):
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            return r.text
        except Exception as exc:
            last_exc = exc
            sleep_s = backoff * (1.0 + i)
            time.sleep(sleep_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"failed to fetch: {url}")


def _extract_examples(html: str, page_url: str) -> List[Dict[str, str]]:
    if BeautifulSoup is None:
        return []
    soup = BeautifulSoup(html, "html.parser")
    rows: List[Dict[str, str]] = []
    for a in soup.select("div.examples-container a"):
        href = (a.get("href") or "").strip()
        title = (a.get_text(" ", strip=True) or "").strip()
        if not href or not title:
            continue
        full = urljoin(page_url, href)
        rows.append(
            {
                "title": title,
                "example_url": full,
                "example_path": urlparse(full).path,
                "source_page": page_url,
            }
        )
    return rows


@dataclass
class D3CrawlResult:
    catalog: pd.DataFrame
    categories: pd.DataFrame
    family_counts: pd.DataFrame
    crawl_log: Dict[str, object]


def _crawl_d3og_all(out_dir: Path) -> D3CrawlResult:
    failed_urls: List[Dict[str, str]] = []
    index_urls: List[str] = []
    category_seed: Dict[str, Dict[str, str]] = {}
    category_example_map: Dict[str, set] = defaultdict(set)

    base = "https://d3og.com/"
    home_html = _fetch_html(base)
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup (bs4) is required for d3og parsing")
    home = BeautifulSoup(home_html, "html.parser")

    index_urls.append(base)
    for a in home.select("div.pagination-container.top a"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        full = urljoin(base, href)
        if "index-" in full and full not in index_urls:
            index_urls.append(full)
    index_urls = sorted(index_urls, key=lambda u: (0 if u == base else int(re.search(r"index-(\d+)\.html", u).group(1))))

    examples_raw: List[Dict[str, str]] = []
    for url in index_urls:
        try:
            html = home_html if url == base else _fetch_html(url)
            examples_raw.extend(_extract_examples(html, url))
        except Exception as exc:
            failed_urls.append({"url": url, "reason": f"{type(exc).__name__}:{exc}"})

    categories_url = "https://d3og.com/categories.html"
    try:
        categories_html = _fetch_html(categories_url)
        cat_soup = BeautifulSoup(categories_html, "html.parser")
        for a in cat_soup.select("div.categories-container a"):
            href = (a.get("href") or "").strip()
            name = (a.get_text(" ", strip=True) or "").strip()
            if not href or not name:
                continue
            full = urljoin(categories_url, href)
            m = re.match(r".*category-([a-z0-9_-]+)-1\.html$", full)
            if not m:
                continue
            slug = m.group(1)
            category_seed[slug] = {"name": name, "seed_url": full}
    except Exception as exc:
        failed_urls.append({"url": categories_url, "reason": f"{type(exc).__name__}:{exc}"})

    cat_page_count: Dict[str, int] = {}
    for slug, meta in category_seed.items():
        visited = set()
        queue = [meta["seed_url"]]
        cat_page_count[slug] = 0
        while queue:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            cat_page_count[slug] += 1
            try:
                html = _fetch_html(url)
                rows = _extract_examples(html, url)
                for r in rows:
                    examples_raw.append(r)
                    category_example_map[slug].add(r["example_path"])
                s = BeautifulSoup(html, "html.parser")
                for a in s.select("a"):
                    href = (a.get("href") or "").strip()
                    if not href:
                        continue
                    full = urljoin(url, href)
                    if re.search(rf"category-{re.escape(slug)}-\d+\.html$", full) and full not in visited:
                        queue.append(full)
            except Exception as exc:
                failed_urls.append({"url": url, "reason": f"{type(exc).__name__}:{exc}"})

    df = pd.DataFrame(examples_raw)
    if df.empty:
        df = pd.DataFrame(columns=["title", "example_url", "example_path", "source_page", "categories", "chart_family"])
    else:
        agg = (
            df.groupby(["example_path", "example_url", "title"], as_index=False)
            .agg(source_pages=("source_page", lambda s: "|".join(sorted(set(map(str, s))))))
        )
        cat_lookup: Dict[str, List[str]] = {}
        for slug, paths in category_example_map.items():
            for p in paths:
                cat_lookup.setdefault(p, []).append(category_seed[slug]["name"])
        agg["categories"] = agg["example_path"].map(lambda p: "|".join(sorted(cat_lookup.get(p, []))))
        agg["chart_family"] = agg["title"].map(_family_from_title)
        df = agg.sort_values(["chart_family", "title"]).reset_index(drop=True)

    family_counts = (
        df.groupby("chart_family", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    categories = []
    for slug, meta in category_seed.items():
        categories.append(
            {
                "category_slug": slug,
                "category_name": meta["name"],
                "seed_url": meta["seed_url"],
                "page_count": int(cat_page_count.get(slug, 0)),
                "example_count": int(len(category_example_map.get(slug, set()))),
            }
        )
    categories_df = pd.DataFrame(categories).sort_values("example_count", ascending=False).reset_index(drop=True)

    crawl_log = {
        "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "index_page_count": len(index_urls),
        "category_count": len(category_seed),
        "catalog_rows": int(len(df)),
        "failed_url_count": int(len(failed_urls)),
        "failed_urls": failed_urls,
    }

    _ensure_dir(out_dir)
    df.to_csv(out_dir / "d3og_catalog.csv", index=False, encoding="utf-8-sig")
    df.to_json(out_dir / "d3og_catalog.json", orient="records", force_ascii=False, indent=2)
    categories_df.to_csv(out_dir / "d3og_categories.csv", index=False, encoding="utf-8-sig")
    family_counts.to_csv(out_dir / "d3og_family_counts.csv", index=False, encoding="utf-8-sig")
    (out_dir / "d3og_crawl_log.json").write_text(json.dumps(crawl_log, ensure_ascii=False, indent=2), encoding="utf-8")

    return D3CrawlResult(catalog=df, categories=categories_df, family_counts=family_counts, crawl_log=crawl_log)


def _pick_d3_example(catalog: pd.DataFrame, family: str, include_words: Sequence[str]) -> Dict[str, str]:
    if catalog.empty:
        return {"d3_title": "", "d3_url": "", "d3_family": family}
    df = catalog.copy()
    if family:
        scoped = df[df["chart_family"] == family]
        if not scoped.empty:
            df = scoped
    words = [w.lower() for w in include_words if w]
    score = np.zeros(len(df), dtype=float)
    titles = df["title"].fillna("").astype(str).str.lower().tolist()
    for i, t in enumerate(titles):
        s = 0.0
        for w in words:
            if w in t:
                s += 1.0
        score[i] = s
    df = df.assign(_score=score).sort_values(["_score", "title"], ascending=[False, True])
    row = df.iloc[0]
    return {
        "d3_title": str(row["title"]),
        "d3_url": str(row["example_url"]),
        "d3_family": str(row["chart_family"]),
    }


def _build_chart_selection_matrix(catalog: pd.DataFrame) -> pd.DataFrame:
    specs = [
        {
            "chart_id": "paper_fig01",
            "analysis_task": "Batch-level metric comparison with confidence interval",
            "preferred_family": "bar",
            "keywords": ["grouped bar", "bullet", "bar"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "docs/assets/tables/paper_d3_selected/tbl01_run_metric_ci_enhanced.csv",
        },
        {
            "chart_id": "paper_fig02",
            "analysis_task": "Case-level paired gain/loss from mainline to behavior",
            "preferred_family": "line",
            "keywords": ["line transition", "line", "slope"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "docs/assets/tables/paper_d3_selected/tbl02_case_pairs_mainline_behavior.csv",
        },
        {
            "chart_id": "paper_fig03",
            "analysis_task": "Distribution of verified score by run",
            "preferred_family": "bar",
            "keywords": ["histogram", "density", "bar"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "docs/assets/tables/paper_d3_selected/tbl02_case_pairs_mainline_behavior.csv + case metrics csv",
        },
        {
            "chart_id": "paper_fig04",
            "analysis_task": "Top-case multi-metric delta heatmap",
            "preferred_family": "heatmap",
            "keywords": ["calendar", "matrix", "heatmap"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "docs/assets/tables/paper_d3_selected/tbl02_case_pairs_mainline_behavior.csv",
        },
        {
            "chart_id": "paper_fig05",
            "analysis_task": "Epoch three-line training curves",
            "preferred_family": "line",
            "keywords": ["multi-series line", "line"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "runs/detect/*/results.csv",
        },
        {
            "chart_id": "paper_fig06",
            "analysis_task": "Robustness to temporal offset (fixed vs adaptive)",
            "preferred_family": "line",
            "keywords": ["line", "multi-series"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "output/paper_experiments/exp_a_uq_align/alignment_noise_curve.csv",
        },
        {
            "chart_id": "paper_fig07",
            "analysis_task": "Calibration quality by UQ bin",
            "preferred_family": "bar",
            "keywords": ["grouped bar", "bar"],
            "paper_use": "appendix",
            "frontend_use": "yes",
            "input_data": "output/paper_experiments/exp_b_reliability_calibration/reliability_bins.csv",
        },
        {
            "chart_id": "paper_fig08",
            "analysis_task": "Latency-quality tradeoff scatter",
            "preferred_family": "scatter",
            "keywords": ["scatter", "dot"],
            "paper_use": "main_text",
            "frontend_use": "yes",
            "input_data": "paper_experiments/*_case_metrics.csv",
        },
    ]
    rows = []
    for spec in specs:
        pick = _pick_d3_example(catalog, spec["preferred_family"], spec["keywords"])
        rows.append(
            {
                **spec,
                **pick,
                "selection_reason": "matched family + title keyword against d3og catalog",
            }
        )
    return pd.DataFrame(rows)


def _style_matplotlib(force_cpu: bool) -> None:
    if force_cpu:
        matplotlib.use("Agg")
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#d0d7de",
            "axes.labelcolor": "#0f172a",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#e2e8f0",
            "grid.linestyle": "-",
            "font.size": 12,
            "axes.grid": True,
        }
    )


def _compute_run_metric_ci(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metrics = ["verified_p_match_mean", "align_avg_candidates", "elapsed_sec"]
    for run in RUN_ORDER:
        df = frames.get(run)
        if df is None or df.empty:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
            if vals.size == 0:
                continue
            mean, lo, hi = _bootstrap_ci(vals, seed=20260422 + len(rows), n_boot=2000)
            rows.append(
                {
                    "run_name": run,
                    "metric": metric,
                    "n": int(vals.size),
                    "mean": float(mean),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "median": float(np.median(vals)),
                    "std": float(np.std(vals, ddof=1) if vals.size > 1 else 0.0),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["run_display"] = out["run_name"].map(lambda x: RUN_DISPLAY.get(x, x))
    return out


def _build_case_pairs(mainline_df: pd.DataFrame, behavior_df: pd.DataFrame) -> pd.DataFrame:
    if mainline_df.empty or behavior_df.empty:
        return pd.DataFrame()
    left = mainline_df.rename(
        columns={
            "verified_p_match_mean": "mainline_verified_p_match_mean",
            "align_avg_candidates": "mainline_align_avg_candidates",
            "elapsed_sec": "mainline_elapsed_sec",
        }
    )
    right = behavior_df.rename(
        columns={
            "verified_p_match_mean": "behavior_verified_p_match_mean",
            "align_avg_candidates": "behavior_align_avg_candidates",
            "elapsed_sec": "behavior_elapsed_sec",
        }
    )
    keep_cols = [
        "case_id",
        "view_code",
        "duration_sec",
        "mainline_verified_p_match_mean",
        "mainline_align_avg_candidates",
        "mainline_elapsed_sec",
        "behavior_verified_p_match_mean",
        "behavior_align_avg_candidates",
        "behavior_elapsed_sec",
    ]
    merged = pd.merge(
        left[["case_id", "view_code", "duration_sec", "mainline_verified_p_match_mean", "mainline_align_avg_candidates", "mainline_elapsed_sec"]],
        right[["case_id", "view_code", "duration_sec", "behavior_verified_p_match_mean", "behavior_align_avg_candidates", "behavior_elapsed_sec"]],
        on=["case_id", "view_code", "duration_sec"],
        how="inner",
    )
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged["d_verified_p_match_mean"] = merged["behavior_verified_p_match_mean"] - merged["mainline_verified_p_match_mean"]
    merged["d_align_avg_candidates"] = merged["behavior_align_avg_candidates"] - merged["mainline_align_avg_candidates"]
    merged["d_elapsed_sec"] = merged["behavior_elapsed_sec"] - merged["mainline_elapsed_sec"]
    merged["abs_d_verified_p_match_mean"] = merged["d_verified_p_match_mean"].abs()
    merged = merged.sort_values("d_verified_p_match_mean", ascending=False).reset_index(drop=True)
    return merged


def _plot_fig01_batch_ci(ci_df: pd.DataFrame, out_path: Path) -> None:
    metrics = ["verified_p_match_mean", "align_avg_candidates", "elapsed_sec"]
    metric_title = {
        "verified_p_match_mean": "Verified Match Score",
        "align_avg_candidates": "Avg Alignment Candidates",
        "elapsed_sec": "Elapsed Time (sec)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = ci_df[ci_df["metric"] == metric].copy()
        if sub.empty:
            ax.set_axis_off()
            continue
        sub["order"] = sub["run_name"].map(lambda x: RUN_ORDER.index(x) if x in RUN_ORDER else 999)
        sub = sub.sort_values("order")
        x = np.arange(len(sub))
        means = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci95_low"].to_numpy(dtype=float)
        hi = sub["ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([means - lo, hi - means])
        colors = [RUN_COLORS.get(r, "#94a3b8") for r in sub["run_name"]]
        ax.bar(x, means, color=colors, alpha=0.9, edgecolor="#0f172a", linewidth=0.5)
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#111827", elinewidth=1.2, capsize=4)
        for k, v in enumerate(means):
            ax.text(k, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color="#0f172a")
        ax.set_title(metric_title.get(metric, metric))
        ax.set_xticks(x)
        ax.set_xticklabels([RUN_DISPLAY.get(r, r) for r in sub["run_name"]], rotation=18, ha="right")
    fig.suptitle("Batch Metrics with 95% Bootstrap CI", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_fig02_case_slope(case_df: pd.DataFrame, out_path: Path, top_n: int = 22) -> None:
    if case_df.empty:
        return
    sub = case_df.sort_values("abs_d_verified_p_match_mean", ascending=False).head(top_n).copy()
    fig, ax = plt.subplots(figsize=(9, 10))
    x0, x1 = 0.0, 1.0
    for _, r in sub.iterrows():
        y0 = float(r["mainline_verified_p_match_mean"])
        y1 = float(r["behavior_verified_p_match_mean"])
        d = y1 - y0
        color = "#0b7285" if d >= 0 else "#b02a37"
        alpha = 0.85 if abs(d) >= 0.01 else 0.4
        ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=1.8)
    ax.scatter(np.full(len(sub), x0), sub["mainline_verified_p_match_mean"], color="#2f9e44", s=32, zorder=3, label="Mainline-v3")
    ax.scatter(np.full(len(sub), x1), sub["behavior_verified_p_match_mean"], color="#0b7285", s=32, zorder=3, label="BehaviorAug-v1")
    top = sub.iloc[0]
    ax.text(
        x1 + 0.02,
        float(top["behavior_verified_p_match_mean"]),
        f"best gain: {top['case_id']} ({top['d_verified_p_match_mean']:+.3f})",
        fontsize=10,
        color="#0f172a",
        va="center",
    )
    ax.set_xlim(-0.2, 1.45)
    ax.set_xticks([x0, x1])
    ax.set_xticklabels(["Mainline-v3", "BehaviorAug-v1"])
    ax.set_ylabel("verified_p_match_mean")
    ax.set_title("Case-wise Paired Change (Top |Delta| Cases)", fontsize=16, weight="bold")
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_fig03_hist(frames: Dict[str, pd.DataFrame], out_path: Path, device: str, gpu_chunk_size: int) -> Dict[str, Dict[str, object]]:
    bins = np.linspace(0.0, 1.0, 21)
    mids = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    op_logs: Dict[str, Dict[str, object]] = {}
    for run in RUN_ORDER:
        df = frames.get(run)
        if df is None or df.empty or "verified_p_match_mean" not in df.columns:
            continue
        vals = pd.to_numeric(df["verified_p_match_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        density, op = _gpu_histogram(vals, bins=bins, device=device, gpu_chunk_size=gpu_chunk_size)
        op_logs[f"hist_{run}"] = op
        ax.plot(mids, density, color=RUN_COLORS.get(run, "#64748b"), linewidth=2.0, label=RUN_DISPLAY.get(run, run))
        ax.fill_between(mids, density, color=RUN_COLORS.get(run, "#64748b"), alpha=0.12)
    ax.set_xlabel("verified_p_match_mean")
    ax.set_ylabel("density")
    ax.set_title("Score Distribution Across Runs", fontsize=16, weight="bold")
    ax.set_xlim(0, 1)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return op_logs


def _plot_fig04_heatmap(case_df: pd.DataFrame, out_path: Path, device: str, gpu_chunk_size: int) -> Dict[str, object]:
    cols = ["d_verified_p_match_mean", "d_align_avg_candidates", "d_elapsed_sec"]
    if case_df.empty:
        return {"used_cuda": False, "fallback_reason": "empty_case_df"}
    sub = case_df.sort_values("abs_d_verified_p_match_mean", ascending=False).head(20).copy()
    mat = sub[cols].fillna(0.0).to_numpy(dtype=np.float32)
    scaled, op = _gpu_colwise_center_scale(mat, device=device)

    fig, ax = plt.subplots(figsize=(8.5, 9.3))
    im = ax.imshow(scaled, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["case_id"].tolist())
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=18, ha="right")
    ax.set_title("Top-20 Case Delta Heatmap (Behavior - Mainline)", fontsize=16, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("scaled delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return op


def _plot_fig05_epoch_three_lines(root: Path, out_path: Path, csv_list: Sequence[str]) -> None:
    csv_paths = [_resolve_path(root, p) for p in csv_list if str(p).strip()]
    frames = []
    for p in csv_paths:
        if not p.exists():
            continue
        df = _safe_read_csv(p)
        if df.empty:
            continue
        df["__run__"] = p.parent.name
        frames.append(df)
    if not frames:
        return
    m1 = "metrics/precision(B)"
    m2 = "metrics/recall(B)"
    m3 = "metrics/mAP50(B)"
    valid = [d for d in frames if all(c in d.columns for c in (m1, m2, m3))]
    if not valid:
        return
    fig, axes = plt.subplots(len(valid), 1, figsize=(14, 3.5 * len(valid)), squeeze=False)
    for i, df in enumerate(valid):
        ax = axes[i, 0]
        x = np.arange(1, len(df) + 1)
        ax.plot(x, df[m1], color="#0b7285", linewidth=2, label="Precision")
        ax.plot(x, df[m2], color="#e67700", linewidth=2, label="Recall")
        ax.plot(x, df[m3], color="#2f9e44", linewidth=2, label="mAP50")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(1, max(2, len(df)))
        ax.set_title(f"Epoch Curves ({df['__run__'].iloc[0]})", weight="bold")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
    axes[-1, 0].set_xlabel("epoch")
    fig.suptitle("Epoch Three-Line Comparison", fontsize=18, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_fig06_noise_curve(noise_df: pd.DataFrame, out_path: Path) -> None:
    if noise_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for mode, color in [("fixed", "#b02a37"), ("adaptive_uq", "#0b7285")]:
        sub = noise_df[noise_df["mode"] == mode].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("offset_sec")
        axes[0].plot(sub["offset_sec"], sub["alignment_recall_at_1"], marker="o", color=color, linewidth=2.2, label=mode)
        axes[1].plot(sub["offset_sec"], sub["mean_temporal_overlap"], marker="o", color=color, linewidth=2.2, label=mode)
    axes[0].set_title("Recall@1 vs Temporal Offset")
    axes[0].set_xlabel("offset_sec")
    axes[0].set_ylabel("alignment_recall_at_1")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_title("Overlap vs Temporal Offset")
    axes[1].set_xlabel("offset_sec")
    axes[1].set_ylabel("mean_temporal_overlap")
    axes[1].set_ylim(-0.03, 1.03)
    for ax in axes:
        ax.legend()
    fig.suptitle("Robustness: Fixed Window vs Adaptive UQ Window", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_fig07_reliability(rel_df: pd.DataFrame, out_path: Path) -> None:
    if rel_df.empty:
        return
    show = rel_df.copy()
    show["method"] = show["method"].astype(str)
    show["uq_bin"] = show["uq_bin"].astype(str)
    pivot = show.pivot_table(index="uq_bin", columns="method", values="ECE", aggfunc="mean")
    pivot = pivot.reindex(["low_uq", "mid_uq", "high_uq"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if not pivot.empty:
        x = np.arange(len(pivot.index))
        width = 0.22
        for i, method in enumerate(pivot.columns):
            vals = pivot[method].to_numpy(dtype=float)
            ax.bar(x + (i - (len(pivot.columns) - 1) / 2) * width, vals, width=width, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.tolist())
    ax.set_ylabel("ECE")
    ax.set_title("Calibration Error by UQ Bin and Method", fontsize=16, weight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_fig08_latency_scatter(all_cases_df: pd.DataFrame, out_path: Path) -> None:
    if all_cases_df.empty:
        return
    keep = all_cases_df.dropna(subset=["duration_sec", "elapsed_sec", "verified_p_match_mean"]).copy()
    if keep.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    for run in RUN_ORDER:
        sub = keep[keep["run_name"] == run]
        if sub.empty:
            continue
        s = np.clip(pd.to_numeric(sub.get("align_avg_candidates", 0), errors="coerce").fillna(0).to_numpy(dtype=float), 0, 20)
        s = 24 + s * 7.0
        ax.scatter(
            sub["duration_sec"],
            sub["elapsed_sec"],
            s=s,
            c=RUN_COLORS.get(run, "#64748b"),
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            label=RUN_DISPLAY.get(run, run),
        )
    ax.set_xlabel("video duration_sec")
    ax.set_ylabel("pipeline elapsed_sec")
    ax.set_title("Latency vs Video Duration (bubble size: align_avg_candidates)", fontsize=15, weight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _to_json_records(df: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []
    src = df if columns is None else df[list(columns)].copy()
    out = []
    for row in src.to_dict(orient="records"):
        clean: Dict[str, object] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v) if isinstance(v, np.floating) else int(v)
            elif isinstance(v, (list, tuple, dict)):
                clean[k] = v
            elif pd.isna(v):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


def _write_showcase_json(
    out_path: Path,
    ci_df: pd.DataFrame,
    case_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    rel_df: pd.DataFrame,
    selection_df: pd.DataFrame,
    family_counts_df: pd.DataFrame,
) -> None:
    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "paper/frontend chart data generated from real run outputs",
        },
        "run_metric_ci": _to_json_records(ci_df),
        "case_pairs": _to_json_records(case_df),
        "noise_curve": _to_json_records(noise_df),
        "reliability_bins": _to_json_records(rel_df),
        "chart_selection_matrix": _to_json_records(selection_df),
        "d3og_family_counts": _to_json_records(family_counts_df),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser("Build d3og-selected paper figures and frontend data")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--gpu_chunk_size", type=int, default=200000)
    parser.add_argument("--force_cpu_for_matplotlib", action="store_true")
    parser.add_argument("--crawl_d3og", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mainline_csv", type=str, default="paper_experiments/real_cases/random6_20260420_mainline_v3_case_metrics.csv")
    parser.add_argument("--behavior_csv", type=str, default="paper_experiments/real_cases/random6_20260420_behavior_aug_v1_case_metrics.csv")
    parser.add_argument("--yolo_csv", type=str, default="paper_experiments/run_logs/random6_20260420_yolo11x_object_case_metrics.csv")
    parser.add_argument("--baseline_csv", type=str, default="paper_experiments/run_logs/random6_20260420_case_metrics.csv")
    parser.add_argument("--noise_csv", type=str, default="output/paper_experiments/exp_a_uq_align/alignment_noise_curve.csv")
    parser.add_argument("--reliability_csv", type=str, default="output/paper_experiments/exp_b_reliability_calibration/reliability_bins.csv")
    parser.add_argument(
        "--epoch_csvs",
        type=str,
        default="runs/detect/case_yolo_train/results.csv,runs/detect/train3/results.csv,runs/detect/train/results.csv",
    )
    args = parser.parse_args()

    root = _resolve_path(Path.cwd(), args.root)
    chart_dir = root / "docs" / "assets" / "charts" / "paper_d3_selected"
    table_dir = root / "docs" / "assets" / "tables" / "paper_d3_selected"
    _ensure_dir(chart_dir)
    _ensure_dir(table_dir)

    _style_matplotlib(force_cpu=bool(args.force_cpu_for_matplotlib))
    device, device_info = _choose_device(args.device)

    source_paths = {
        "mainline_v3": _resolve_path(root, args.mainline_csv),
        "behavior_aug_v1": _resolve_path(root, args.behavior_csv),
        "yolo11x_object": _resolve_path(root, args.yolo_csv),
        "random6_baseline": _resolve_path(root, args.baseline_csv),
    }
    run_frames: Dict[str, pd.DataFrame] = {}
    missing_inputs: List[str] = []
    for run, path in source_paths.items():
        if not path.exists():
            missing_inputs.append(str(path))
            run_frames[run] = pd.DataFrame()
            continue
        run_frames[run] = _clean_run_frame(_safe_read_csv(path, required=False), run)

    noise_path = _resolve_path(root, args.noise_csv)
    reliability_path = _resolve_path(root, args.reliability_csv)
    noise_df = _safe_read_csv(noise_path, required=False)
    reliability_df = _safe_read_csv(reliability_path, required=False)
    if not noise_path.exists():
        missing_inputs.append(str(noise_path))
    if not reliability_path.exists():
        missing_inputs.append(str(reliability_path))

    d3_result = D3CrawlResult(
        catalog=pd.DataFrame(),
        categories=pd.DataFrame(),
        family_counts=pd.DataFrame(),
        crawl_log={"skipped": True},
    )
    if args.crawl_d3og == 1:
        try:
            d3_result = _crawl_d3og_all(table_dir)
        except Exception as exc:
            fallback_catalog = _safe_read_csv(table_dir / "d3og_catalog.csv", required=False)
            fallback_family = _safe_read_csv(table_dir / "d3og_family_counts.csv", required=False)
            fallback_cats = _safe_read_csv(table_dir / "d3og_categories.csv", required=False)
            d3_result = D3CrawlResult(
                catalog=fallback_catalog,
                categories=fallback_cats,
                family_counts=fallback_family,
                crawl_log={
                    "skipped": False,
                    "crawl_failed": True,
                    "reason": f"{type(exc).__name__}:{exc}",
                    "used_cached_catalog": bool(not fallback_catalog.empty),
                },
            )

    selection_df = _build_chart_selection_matrix(d3_result.catalog)
    if not selection_df.empty:
        selection_df.to_csv(table_dir / "chart_selection_matrix.csv", index=False, encoding="utf-8-sig")

    ci_df = _compute_run_metric_ci(run_frames)
    if not ci_df.empty:
        ci_df.to_csv(table_dir / "tbl01_run_metric_ci_enhanced.csv", index=False, encoding="utf-8-sig")

    case_df = _build_case_pairs(run_frames.get("mainline_v3", pd.DataFrame()), run_frames.get("behavior_aug_v1", pd.DataFrame()))
    if not case_df.empty:
        case_df.to_csv(table_dir / "tbl02_case_pairs_mainline_behavior.csv", index=False, encoding="utf-8-sig")

    all_cases = pd.concat([df for df in run_frames.values() if df is not None and not df.empty], ignore_index=True)
    if not all_cases.empty:
        all_cases.to_csv(table_dir / "tbl03_all_case_metrics_long.csv", index=False, encoding="utf-8-sig")

    fig_manifest: List[Dict[str, str]] = []
    cuda_ops: Dict[str, Dict[str, object]] = {}

    if not ci_df.empty:
        p = chart_dir / "fig01_batch_metric_ci.png"
        _plot_fig01_batch_ci(ci_df, p)
        fig_manifest.append({"figure_id": "fig01", "path": str(p)})
    if not case_df.empty:
        p = chart_dir / "fig02_case_slope_mainline_vs_behavior.png"
        _plot_fig02_case_slope(case_df, p)
        fig_manifest.append({"figure_id": "fig02", "path": str(p)})

        p = chart_dir / "fig04_case_delta_heatmap_top20.png"
        op = _plot_fig04_heatmap(case_df, p, device=device, gpu_chunk_size=args.gpu_chunk_size)
        cuda_ops["heatmap_delta_scale"] = op
        fig_manifest.append({"figure_id": "fig04", "path": str(p)})

    p = chart_dir / "fig03_score_hist_density_runs.png"
    op_hist = _plot_fig03_hist(run_frames, p, device=device, gpu_chunk_size=args.gpu_chunk_size)
    cuda_ops.update(op_hist)
    fig_manifest.append({"figure_id": "fig03", "path": str(p)})

    epoch_csvs = [x.strip() for x in str(args.epoch_csvs).split(",") if x.strip()]
    p = chart_dir / "fig05_epoch_three_lines.png"
    _plot_fig05_epoch_three_lines(root, p, epoch_csvs)
    fig_manifest.append({"figure_id": "fig05", "path": str(p)})

    if not noise_df.empty:
        p = chart_dir / "fig06_noise_robustness_curve.png"
        _plot_fig06_noise_curve(noise_df, p)
        fig_manifest.append({"figure_id": "fig06", "path": str(p)})
    if not reliability_df.empty:
        p = chart_dir / "fig07_reliability_bins_ece.png"
        _plot_fig07_reliability(reliability_df, p)
        fig_manifest.append({"figure_id": "fig07", "path": str(p)})
    if not all_cases.empty:
        p = chart_dir / "fig08_latency_vs_duration_scatter.png"
        _plot_fig08_latency_scatter(all_cases, p)
        fig_manifest.append({"figure_id": "fig08", "path": str(p)})

    if fig_manifest:
        pd.DataFrame(fig_manifest).to_csv(table_dir / "tbl05_figure_manifest.csv", index=False, encoding="utf-8-sig")

    showcase_json = table_dir / "showcase_data.json"
    _write_showcase_json(
        showcase_json,
        ci_df=ci_df,
        case_df=case_df,
        noise_df=noise_df,
        rel_df=reliability_df,
        selection_df=selection_df,
        family_counts_df=d3_result.family_counts,
    )

    build_log = {
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "device": device_info,
        "cuda_ops": cuda_ops,
        "missing_inputs": missing_inputs,
        "d3og_crawl_log": d3_result.crawl_log,
        "generated_chart_count": len(fig_manifest),
        "generated_charts": fig_manifest,
        "generated_tables": [
            str(table_dir / "tbl01_run_metric_ci_enhanced.csv"),
            str(table_dir / "tbl02_case_pairs_mainline_behavior.csv"),
            str(table_dir / "tbl03_all_case_metrics_long.csv"),
            str(table_dir / "chart_selection_matrix.csv"),
            str(table_dir / "d3og_catalog.csv"),
            str(showcase_json),
        ],
    }
    (table_dir / "plot_build_log_d3_selected.json").write_text(json.dumps(build_log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "chart_count": len(fig_manifest), "table_dir": str(table_dir), "chart_dir": str(chart_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
