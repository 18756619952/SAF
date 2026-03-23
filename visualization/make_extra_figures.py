#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate *additional* figures (raincloud + scenario-marker variants) without altering existing outputs.

What it does (high level):
- Fig F (baseline + policy): original boxplot+points + raincloud; each with and without scenario markers.
- Fig 3.3 (baseline only): min–max rectangle + pass/fail split + jitter cloud; plus raincloud pass/fail;
  each with and without scenario markers; percent pass labels per year.

Usage:
python make_extra_figures.py --policy-outdir <.../policy_layer> \
  --regions "GCAM 6.0 NGFS|EU-15" "GCAM 6.0 NGFS|USA" "GCAM 6.0 NGFS|China" "GCAM 6.0 NGFS|Japan" \
  --scenarios "Current Policies" "Delayed transition" "Fragmented World" "Net Zero 2050"
"""

import argparse
from pathlib import Path
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import json

MUTED_10 = [
    '#4C72B0',  # blue
    '#DD8452',  # orange
    '#55A868',  # green
    '#C44E52',  # red
    '#8172B3',  # purple
    '#937860',  # brown
    '#DA8BC3',  # pink
    '#8C8C8C',  # gray
    '#CCB974',  # olive
    '#64B5CD',  # cyan
]

def make_path_color(pathways):
    """Deterministic mapping from pathway name -> muted colors.
    PtL_DAC is always blue.
    """
    pathways = list(pathways)
    mapping = {}
    if 'PtL_DAC' in pathways:
        mapping['PtL_DAC'] = MUTED_10[0]
        pathways = [p for p in pathways if p != 'PtL_DAC']
    palette = MUTED_10[1:]
    for i, p in enumerate(pathways):
        mapping[p] = palette[i % len(palette)]
    return mapping

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------- Defaults (LCFS-style CI standard points; kgCO2e/MJ) ----------
DEFAULT_LCFS_POINTS = [
    {"year": 2025, "value": 0.0890},
    {"year": 2030, "value": 0.0623},
    {"year": 2045, "value": 0.0089},
]

SCEN_MARKERS = {
    "Current Policies": "o",
    "Delayed transition": "s",
    "Fragmented World": "^",
    "Net Zero 2050": "D",
}

# --- aesthetic controls (journal-style) ---
THR_LINE_COLOR = "0.30"   # medium gray; visible but not dominant
THR_LINE_LW    = 1.9      # thinner than before
THR_LINE_DASH  = (0, (3, 2))  # shorter/denser dashes (more uniform look)
THR_ZORDER     = 2.2

LEGEND_FONTSIZE = 9
LEGEND_TITLE_FONTSIZE = 9


PATHWAY_ORDER_FALLBACK = [
    "ATJ_FR", "ATJ_Switchgrass",
    "FT_Miscanthus", "FT_Switchgrass",
    "HEFA_Soybean", "HEFA_UCO",
    "PtL_DAC",
]

def _norm_cols(df: pd.DataFrame):
    """Return mapping from lowercase stripped colname -> original colname."""
    return {c.strip().lower(): c for c in df.columns}

def _pick_col(df: pd.DataFrame, candidates):
    m = _norm_cols(df)
    for cand in candidates:
        if cand in m:
            return m[cand]
    # fuzzy contains
    for cand in candidates:
        for k, orig in m.items():
            if cand in k:
                return orig
    return None

def infer_columns(df: pd.DataFrame):
    """Infer standard columns (year, region, scenario, pathway)."""
    year = _pick_col(df, ["year"])
    region = _pick_col(df, ["region"])
    scenario = _pick_col(df, ["scenario"])
    pathway = _pick_col(df, ["pathway", "path"])
    if year is None:
        raise ValueError("Could not find a 'year' column in CSV.")
    if region is None:
        raise ValueError("Could not find a 'region' column in CSV.")
    if scenario is None:
        raise ValueError("Could not find a 'scenario' column in CSV.")
    if pathway is None:
        raise ValueError("Could not find a 'pathway' column in CSV.")
    return year, region, scenario, pathway

def infer_ci_col(df: pd.DataFrame):
    """Infer CI column; return (colname, factor_to_g_per_MJ)."""
    m = _norm_cols(df)
    # priority exact-ish
    for key in [
        "ci_gco2e_per_mj",
        "ci_gco2_per_mj",
        "ci_gco2eq_per_mj",
        "ci_kgco2e_per_mj",
        "ci_kgco2_per_mj",
        "ci_kgco2eq_per_mj",
        "ci",
        "carbon_intensity",
    ]:
        for k, orig in m.items():
            if k == key:
                col = orig
                break
        else:
            col = None
        if col is not None:
            break
    if col is None:
        # contains "ci" and "per_mj"
        for k, orig in m.items():
            if ("ci" in k) and ("per_mj" in k):
                col = orig
                break
    if col is None:
        raise ValueError("Could not infer a CI column (expected something like ci_*_per_MJ).")
    lk = col.strip().lower()
    # units
    if "kg" in lk:
        return col, 1000.0  # kg/MJ -> g/MJ
    return col, 1.0        # already g/MJ

def infer_cost_col(df: pd.DataFrame):
    """Infer net cost column; return (colname, factor_to_USD_per_GJ)."""
    m = _norm_cols(df)
    # prefer existing used in policy layer
    for key in [
        "net_cost_lcfs_usd2022_per_mj",
        "net_cost_usd2022_per_mj",
        "net_cost_per_mj",
        "net_cost",
    ]:
        for k, orig in m.items():
            if k == key:
                col = orig
                break
        else:
            col = None
        if col is not None:
            break
    if col is None:
        # contains net_cost
        for k, orig in m.items():
            if "net_cost" in k:
                col = orig
                break
    if col is None:
        raise ValueError("Could not infer net cost column (expected net_cost_*).")
    lk = col.strip().lower()
    if "per_mj" in lk:
        return col, 1000.0  # USD/MJ -> USD/GJ
    return col, 1.0

def infer_credit_balance_col(df: pd.DataFrame):
    """Infer LCFS credit-balance column (signed). Returns (colname, factor_to_USD_per_GJ).

    Convention used here:
      - We want y<0 to mean **Credit** (revenue, lowers net cost),
        and y>0 to mean **Deficit** (payment, raises net cost).
      - In policy_layer outputs, `lcfs_credit_value_*` is typically:
           + for credits, - for deficits.
        Therefore we plot:  y = -lcfs_credit_value
    """
    m = _norm_cols(df)
    for key in [
        "lcfs_credit_value_usd2022_per_mj",
        "lcfs_credit_value_per_mj",
        "lcfs_credit_value",
        "credit_value_usd2022_per_mj",
        "credit_value",
    ]:
        for k, orig in m.items():
            if k == key:
                col = orig
                break
        else:
            col = None
        if col is not None:
            break
    if col is None:
        # fallback: contains both 'lcfs' and 'credit'
        for k, orig in m.items():
            if ("lcfs" in k and "credit" in k) or ("credit_value" in k):
                col = orig
                break
        else:
            col = None
    if col is None:
        raise ValueError("Could not infer LCFS credit value column (expected lcfs_credit_value_*).")
    lk = col.strip().lower()
    if "per_mj" in lk:
        return col, 1000.0  # USD/MJ -> USD/GJ
    return col, 1.0


def load_policy_config_points(policy_outdir: Path):
    """Try to load policy_config.json; else return DEFAULT_LCFS_POINTS."""
    # priority: policy_outdir/policy_config.json then repo root next to script
    for p in [policy_outdir / "policy_config.json", Path(__file__).resolve().parent / "policy_config.json"]:
        if p.exists():
            try:
                cfg = json.loads(p.read_text(encoding="utf-8"))
                pts = cfg.get("ci_standard_points_year_kgCO2_per_MJ")
                if isinstance(pts, list) and all(("year" in x and "value" in x) for x in pts):
                    return pts
            except Exception:
                pass
    return DEFAULT_LCFS_POINTS

def threshold_series(years, points):
    """Return dict year->threshold_kg_per_MJ for given years (piecewise linear; hold after last point)."""
    pts = sorted([(int(p["year"]), float(p["value"])) for p in points], key=lambda x: x[0])
    y0, v0 = pts[0]
    xs = np.array([p[0] for p in pts], dtype=float)
    vs = np.array([p[1] for p in pts], dtype=float)
    out = {}
    for y in years:
        y = int(y)
        if y <= xs[0]:
            out[y] = float(vs[0])
        elif y >= xs[-1]:
            out[y] = float(vs[-1])
        else:
            out[y] = float(np.interp(y, xs, vs))
    return out

def deterministic_rng(*items):
    """Stable RNG seed from strings/ints."""
    s = "|".join(map(str, items))
    seed = (abs(hash(s)) % (2**32 - 1))
    return np.random.default_rng(seed)

def jitter_x(center_x, n, half_width, rng):
    """Random jitter within [-half_width, +half_width], reproducible."""
    return center_x + rng.uniform(-half_width, half_width, size=n)

def draw_threshold_segment(ax, x_center, y, half_width, *,
                          color=THR_LINE_COLOR, lw=THR_LINE_LW, dash=THR_LINE_DASH,
                          zorder=THR_ZORDER):
    """Draw a per-year horizontal threshold segment with publication-friendly styling."""
    x0 = x_center - half_width
    x1 = x_center + half_width
    ax.plot([x0, x1], [y, y], color=color, lw=lw, linestyle=dash,
            solid_capstyle="butt", dash_capstyle="butt", zorder=zorder)

# ---------- plotting helpers ----------
def get_pathway_order(df, pathway_col):
    vals = list(pd.unique(df[pathway_col]))
    # keep fallback order first
    order = [p for p in PATHWAY_ORDER_FALLBACK if p in vals]
    # append remaining sorted
    for p in sorted([x for x in vals if x not in order]):
        order.append(p)
    return order

def make_color_map(pathways):
    """Match make_paper_figures.py muted palette."""
    return make_path_color(pathways)


def half_violin(ax, dists, positions, width=0.55, side="right", alpha=0.25, edge_lw=0.8):
    parts = ax.violinplot(dists, positions=positions, widths=width,
                          showmeans=False, showextrema=False, showmedians=False)
    for body in parts["bodies"]:
        verts = body.get_paths()[0].vertices
        x = verts[:, 0]
        mid = np.mean(x)
        if side == "right":
            verts[:, 0] = np.clip(x, mid, np.inf)
        else:
            verts[:, 0] = np.clip(x, -np.inf, mid)
        body.set_alpha(alpha)
        body.set_linewidth(edge_lw)
        body.set_zorder(0.6)
    return parts

def add_box(ax, dists, positions, width=0.18, alpha=0.18, showfliers=False):
    bp = ax.boxplot(dists, positions=positions, widths=width,
                    patch_artist=True, showfliers=showfliers)
    for b in bp["boxes"]:
        b.set_alpha(alpha)
        b.set_zorder(0.9)
    for k in ["whiskers", "caps", "medians"]:
        for line in bp[k]:
            line.set_linewidth(1.0)
    return bp

def scatter_cloud(ax, x_center, y, colors, marker="o", s=18, alpha=0.85,
                  half_width=0.28, rng=None, edgecolor="none", facecolor=None, zorder=3):
    y = np.asarray(y)
    if rng is None:
        rng = np.random.default_rng(0)
    xs = jitter_x(x_center, len(y), half_width, rng)
    if facecolor is None:
        ax.scatter(xs, y, s=s, c=colors, marker=marker, alpha=alpha,
                   edgecolors=edgecolor, zorder=zorder)
    else:
        ax.scatter(xs, y, s=s, facecolors=facecolor, edgecolors=colors, marker=marker,
                   alpha=alpha, zorder=zorder)
    return xs

def beeswarm_1d_x(ax, x_center, y, color_list, marker="o", s=18, alpha=0.8,
                  max_half_width=0.28, rng=None, zorder=3):
    """
    Simple 1D beeswarm in x: avoid overlap in display coordinates.
    Keeps y unchanged, adjusts x within [x_center-max_half_width, x_center+max_half_width].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    y = np.asarray(y)
    n = len(y)
    # Sort by y so close values pack
    order = np.argsort(y)
    xs = np.full(n, x_center, dtype=float)
    # Convert a nominal marker radius from points to pixels
    fig = ax.figure
    r_pts = math.sqrt(s) * 0.9  # heuristic
    r_px = r_pts * fig.dpi / 72.0
    # spacing in pixels
    min_dist_px = 2.0 * r_px * 0.92

    placed = []  # list of (x_disp, y_disp)
    for idx in order:
        yi = y[idx]
        # start at center; if collides, try offsets
        base_x = x_center
        # candidate offsets in data units (will test in display units)
        # generate symmetric offsets increasing
        cand = [0.0]
        step = max_half_width / 18.0
        for k in range(1, 19):
            cand += [k*step, -k*step]
        rng.shuffle(cand[1:])  # keep 0 first, shuffle the rest slightly
        chosen = None
        for dx in cand:
            x_try = base_x + dx
            if abs(dx) > max_half_width:
                continue
            # display coords
            x_disp, y_disp = ax.transData.transform((x_try, yi))
            ok = True
            for xp, yp in placed:
                if abs(y_disp - yp) <= min_dist_px:
                    if abs(x_disp - xp) <= min_dist_px:
                        ok = False
                        break
            if ok:
                chosen = x_try
                placed.append((x_disp, y_disp))
                break
        if chosen is None:
            chosen = base_x + rng.uniform(-max_half_width, max_half_width)
        xs[idx] = chosen

    ax.scatter(xs, y, s=s, c=color_list, marker=marker, alpha=alpha,
               edgecolors="none", zorder=zorder)
    return xs

# ---------- Figure generators ----------
def figF_boxplot_points(df, outpath: Path, regions, years, scenarios, scenario_markers=False):
    year_col, region_col, scen_col, path_col = infer_columns(df)
    cost_col, cost_factor = infer_cost_col(df)
    df = df.copy()
    df["_cost_USD_per_GJ"] = df[cost_col].astype(float) * cost_factor

    pathways = get_pathway_order(df, path_col)
    colors = make_color_map(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax_i, (ax, rg) in enumerate(zip(axes, regions)):
        sub = df[df[region_col] == rg]
        dists = []
        for y in years:
            yy = sub[sub[year_col] == y]["_cost_USD_per_GJ"].values
            dists.append(yy if len(yy) else np.array([np.nan]))

        pos = np.arange(len(years)) + 1
        bp = ax.boxplot(dists, positions=pos, widths=0.60, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_alpha(0.18)
        for k in ["whiskers", "caps", "medians"]:
            for line in bp[k]:
                line.set_linewidth(1.0)

        # points (avoid overlap): use beeswarm if many identical
        for i, y in enumerate(years):
            dyy = sub[sub[year_col] == y]
            if dyy.empty:
                continue
            rng = deterministic_rng("FigF", rg, y)
            # use scenario markers or unified marker
            if scenario_markers:
                for sc in scenarios:
                    dsc = dyy[dyy[scen_col] == sc]
                    if dsc.empty:
                        continue
                    color_list = [colors[p] for p in dsc[path_col].tolist()]
                    marker = SCEN_MARKERS.get(sc, "o")
                    beeswarm_1d_x(ax, pos[i], dsc["_cost_USD_per_GJ"].values, color_list,
                                 marker=marker, s=22, alpha=0.8, max_half_width=0.30, rng=rng)
            else:
                color_list = [colors[p] for p in dyy[path_col].tolist()]
                beeswarm_1d_x(ax, pos[i], dyy["_cost_USD_per_GJ"].values, color_list,
                             marker="o", s=22, alpha=0.8, max_half_width=0.30, rng=rng)

        ax.set_title(rg.replace("GCAM 6.0 NGFS|", ""), loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in axes[::2]:
        ax.set_ylabel("Net cost (USD/GJ)")

    axes[0].set_xticks(np.arange(len(years)) + 1)
    axes[0].set_xticklabels(years)

    # --- legends (bottom, inside reserved margin) ---
    # --- legends (compact, separated blocks) ---
    bottom_space = 0.16 if scenario_markers else 0.10
    fig.tight_layout(rect=(0, bottom_space, 1, 1))

    path_ncol = len(pathways) if len(pathways) <= 7 else 4
    path_handles = [Line2D([0],[0], marker="o", linestyle="",
                           markerfacecolor=colors[p], markeredgecolor="none",
                           markersize=7, label=p) for p in pathways]
    fig.legend(path_handles, pathways, loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=path_ncol, frameon=False, fontsize=LEGEND_FONTSIZE,
               handletextpad=0.4, columnspacing=1.2)

    if scenario_markers:
        scen_handles = [Line2D([0],[0], marker=SCEN_MARKERS.get(sc, "o"), linestyle="none",
                               markerfacecolor="white", markeredgecolor="black", color="black",
                               markersize=7, label=sc) for sc in scenarios]
        fig.legend(scen_handles, scenarios, loc="lower center", bbox_to_anchor=(0.5, 0.070),
                   ncol=4, frameon=False, title="Scenario",
                   fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
                   handletextpad=0.5, columnspacing=1.8)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def figF_raincloud(df, outpath: Path, regions, years, scenarios, scenario_markers=False):
    year_col, region_col, scen_col, path_col = infer_columns(df)
    cost_col, cost_factor = infer_cost_col(df)
    df = df.copy()
    df["_cost_USD_per_GJ"] = df[cost_col].astype(float) * cost_factor

    pathways = get_pathway_order(df, path_col)
    colors = make_color_map(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax_i, (ax, rg) in enumerate(zip(axes, regions)):
        sub = df[df[region_col] == rg]
        dists = []
        for y in years:
            yy = sub[sub[year_col] == y]["_cost_USD_per_GJ"].values
            dists.append(yy if len(yy) else np.array([np.nan]))

        pos = np.arange(len(years)) + 1
        half_violin(ax, dists, pos, width=0.70, side="right", alpha=0.18, edge_lw=0.7)
        add_box(ax, dists, pos, width=0.20, alpha=0.14, showfliers=False)

        # dot cloud on left side with jitter; optional scenario markers
        for i, y in enumerate(years):
            dyy = sub[sub[year_col] == y]
            if dyy.empty:
                continue
            rng = deterministic_rng("FigF_rain", rg, y)
            if scenario_markers:
                for sc in scenarios:
                    dsc = dyy[dyy[scen_col] == sc]
                    if dsc.empty:
                        continue
                    cols = [colors[p] for p in dsc[path_col].tolist()]
                    scatter_cloud(ax, pos[i]-0.12, dsc["_cost_USD_per_GJ"].values,
                                  colors=cols, marker=SCEN_MARKERS.get(sc, "o"),
                                  s=18, alpha=0.70, half_width=0.22, rng=rng, edgecolor="none")
            else:
                cols = [colors[p] for p in dyy[path_col].tolist()]
                scatter_cloud(ax, pos[i]-0.12, dyy["_cost_USD_per_GJ"].values,
                              colors=cols, marker="o", s=18, alpha=0.70,
                              half_width=0.22, rng=rng, edgecolor="none")

        ax.set_title(rg.replace("GCAM 6.0 NGFS|", ""), loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in axes[::2]:
        ax.set_ylabel("Net cost (USD/GJ)")

    axes[0].set_xticks(np.arange(len(years)) + 1)
    axes[0].set_xticklabels(years)

    # --- legends (bottom, inside reserved margin) ---
    # --- legends (compact, separated blocks) ---
    bottom_space = 0.17 if scenario_markers else 0.11
    fig.tight_layout(rect=(0, bottom_space, 1, 1))

    path_ncol = len(pathways) if len(pathways) <= 7 else 4
    path_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=colors[p], markeredgecolor="none",
               markersize=7, label=p)
        for p in pathways
    ]
    fig.legend(path_handles, pathways, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), ncol=path_ncol, frameon=False,
               fontsize=LEGEND_FONTSIZE, handletextpad=0.4, columnspacing=1.2)

    if scenario_markers:
        scen_handles = [
            Line2D([0], [0], marker=SCEN_MARKERS.get(sc, "o"),
                   linestyle="", markerfacecolor="white",
                   markeredgecolor="black", color="black",
                   markersize=7, label=sc)
            for sc in scenarios
        ]
        fig.legend(scen_handles, scenarios, loc="lower center",
                   bbox_to_anchor=(0.5, 0.072), ncol=4, frameon=False, title="Scenario",
                   fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
                   handletextpad=0.5, columnspacing=1.8)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def figF_credit_balance_raincloud(df, outpath: Path, regions, years, scenarios, scenario_markers=False):
    """Raincloud of LCFS credit balance (Credit vs Deficit), Fig 3.4-style.

    y-axis convention:
      - above 0: Credit (revenue; positive balance)
      - below 0: Deficit (payment; negative balance)
      - tick labels are signed (show +/-), with a 0 midline.
    """
    year_col, region_col, scen_col, path_col = infer_columns(df)
    cred_col, factor = infer_credit_balance_col(df)
    df = df.copy()

    # Plot convention: credit above 0, deficit below 0
    # In policy_layer outputs, `lcfs_credit_value_*` is typically:
    #   + for credits, - for deficits.
    df["_bal_USD_per_GJ"] = df[cred_col].astype(float) * factor

    pathways = get_pathway_order(df, path_col)
    colors = make_color_map(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax_i, (ax, rg) in enumerate(zip(axes, regions)):
        sub = df[df[region_col] == rg]
        dists = []
        for y in years:
            yy = sub[sub[year_col] == y]["_bal_USD_per_GJ"].values
            dists.append(yy if len(yy) else np.array([np.nan]))

        pos = np.arange(len(years)) + 1
        # half violin (right side), dot cloud (left side) — no boxplot for this balance view
        half_violin(ax, dists, pos, width=0.70, side="right", alpha=0.18, edge_lw=0.7)

        # dot cloud on left side with jitter; optional scenario markers
        for i, y in enumerate(years):
            dyy = sub[sub[year_col] == y]
            if dyy.empty:
                continue
            rng = deterministic_rng("FigF_credit_bal", rg, y)
            if scenario_markers:
                for sc in scenarios:
                    dsc = dyy[dyy[scen_col] == sc]
                    if dsc.empty:
                        continue
                    cols = [colors[p] for p in dsc[path_col].tolist()]
                    scatter_cloud(
                        ax, pos[i] - 0.12, dsc["_bal_USD_per_GJ"].values,
                        colors=cols, marker=SCEN_MARKERS.get(sc, "o"),
                        s=18, alpha=0.70, half_width=0.22, rng=rng, edgecolor="none"
                    )
            else:
                cols = [colors[p] for p in dyy[path_col].tolist()]
                scatter_cloud(
                    ax, pos[i] - 0.12, dyy["_bal_USD_per_GJ"].values,
                    colors=cols, marker="o", s=18, alpha=0.70,
                    half_width=0.22, rng=rng, edgecolor="none"
                )

        ax.axhline(0, color="0.25", lw=1.0, alpha=0.75, zorder=0)
        ax.set_title(rg.replace("GCAM 6.0 NGFS|", ""), loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

        # Split labels (Credit above 0; Deficit below 0)
        # Place them INSIDE each subplot (top-left / bottom-left) to avoid overlap
        # across panels and to keep them aligned regardless of figure layout.
        label_kw = dict(fontsize=10, color="0.25",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))
        ax.text(0.02, 0.92, "Credit", transform=ax.transAxes,
                ha="left", va="top", **label_kw)
        ax.text(0.02, 0.08, "Deficit", transform=ax.transAxes,
                ha="left", va="bottom", **label_kw)

    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in axes[::2]:
        ax.set_ylabel("Credit balance (USD/GJ)")

    # x ticks as years (shared across panels)
    axes[0].set_xticks(np.arange(len(years)) + 1)
    axes[0].set_xticklabels(years)

    # --- legends (match Fig 3.4 raincloud style) ---
    bottom_space = 0.17 if scenario_markers else 0.11
    fig.tight_layout(rect=(0, bottom_space, 1, 1))

    path_ncol = len(pathways) if len(pathways) <= 7 else 4
    path_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=colors[p], markeredgecolor="none",
               markersize=7, label=p)
        for p in pathways
    ]
    fig.legend(path_handles, pathways, loc="lower center",
               bbox_to_anchor=(0.5, 0.012), ncol=path_ncol, frameon=False,
               fontsize=LEGEND_FONTSIZE, handletextpad=0.4, columnspacing=1.2)

    if scenario_markers:
        scen_handles = [
            Line2D([0], [0], marker=SCEN_MARKERS.get(sc, "o"),
                   linestyle="", markerfacecolor="white",
                   markeredgecolor="black", color="black",
                   markersize=7, label=sc)
            for sc in scenarios
        ]
        fig.legend(scen_handles, scenarios, loc="lower center",
                   bbox_to_anchor=(0.5, 0.072), ncol=4, frameon=False, title="Scenario",
                   fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
                   handletextpad=0.5, columnspacing=1.8)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def fig33_minmax_passfail(df, outpath: Path, regions, years, scenarios, policy_points,
                          scenario_markers=False):
    year_col, region_col, scen_col, path_col = infer_columns(df)
    ci_col, ci_factor = infer_ci_col(df)
    df = df.copy()
    df["_ci_g_per_MJ"] = df[ci_col].astype(float) * ci_factor
    # threshold in g/MJ
    thr_kg = threshold_series(years, policy_points)
    thr_g = {y: thr_kg[y] * 1000.0 for y in thr_kg}

    pathways = get_pathway_order(df, path_col)
    colors = make_color_map(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    pos = np.arange(len(years)) + 1
    rect_width = 0.62
    rect_edge = (0.45, 0.45, 0.45, 1.0)  # medium gray
    rect_lw = 1.0

    for ax, rg in zip(axes, regions):
        sub = df[df[region_col] == rg]

        for i, y in enumerate(years):
            dyy = sub[sub[year_col] == y]
            if dyy.empty:
                continue
            vals = dyy["_ci_g_per_MJ"].values
            ymin, ymax = np.nanmin(vals), np.nanmax(vals)
            ax.add_patch(Rectangle((pos[i] - rect_width/2, ymin),
                                   rect_width, ymax - ymin,
                                   fill=False, edgecolor=rect_edge, lw=rect_lw, zorder=1))

            # threshold split line inside the min–max rectangle (per-year)
            thr = thr_g[int(y)]
            draw_threshold_segment(ax, pos[i], thr, rect_width/2)

            # pass/fail split
            is_pass = vals <= thr

            rng = deterministic_rng("Fig33_minmax", rg, y)
            # scatter: pass filled, fail hollow; colors by pathway
            if scenario_markers:
                for sc in scenarios:
                    dsc = dyy[dyy[scen_col] == sc]
                    if dsc.empty:
                        continue
                    vals_sc = dsc["_ci_g_per_MJ"].values
                    thr_sc = thr
                    pass_sc = vals_sc <= thr_sc
                    cols = [colors[p] for p in dsc[path_col].tolist()]
                    # fail hollow
                    if np.any(~pass_sc):
                        scatter_cloud(ax, pos[i], vals_sc[~pass_sc], [cols[j] for j in np.where(~pass_sc)[0]],
                                      marker=SCEN_MARKERS.get(sc, "o"), s=20, alpha=0.9,
                                      half_width=0.30, rng=rng, facecolor="none", zorder=4)
                    if np.any(pass_sc):
                        ax.scatter(jitter_x(pos[i], pass_sc.sum(), 0.30, rng),
                                   vals_sc[pass_sc], s=20, c=[cols[j] for j in np.where(pass_sc)[0]],
                                   marker=SCEN_MARKERS.get(sc, "o"), alpha=0.85, edgecolors="none", zorder=3)
            else:
                cols = [colors[p] for p in dyy[path_col].tolist()]
                # fail hollow
                if np.any(~is_pass):
                    fail_cols = [cols[j] for j in np.where(~is_pass)[0]]
                    scatter_cloud(ax, pos[i], vals[~is_pass], fail_cols,
                                  marker="o", s=20, alpha=0.9,
                                  half_width=0.30, rng=rng, facecolor="none", zorder=4)
                if np.any(is_pass):
                    pass_cols = [cols[j] for j in np.where(is_pass)[0]]
                    ax.scatter(jitter_x(pos[i], is_pass.sum(), 0.30, rng), vals[is_pass],
                               s=20, c=pass_cols, marker="o", alpha=0.85, edgecolors="none", zorder=3)            # percent label: outside the rectangle (above ymax)
            pass_rate = float(np.mean(is_pass)) * 100.0
            # Place label just outside the rectangle, to the lower-right of the threshold segment
            x_text = pos[i] + rect_width/2 + 0.06
            y_text = thr - 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0])
            ax.text(x_text, y_text, f"{pass_rate:.0f}%", ha="left", va="top",
                    fontsize=8.5, color="0.25", zorder=6, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))


        ax.set_title(rg.replace("GCAM 6.0 NGFS|", ""), loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in axes[::2]:
        ax.set_ylabel("CI (gCO$_2$e/MJ)")

    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(years)

    # --- legends (bottom, inside reserved margin) ---
    # --- legends (compact, 3 clearly separated rows) ---
    bottom_space = 0.26 if scenario_markers else 0.16
    fig.tight_layout(rect=(0, bottom_space, 1, 1))

    # legend row positions (figure fraction)
    y_path = 0.020
    y_scen = 0.085
    y_thr  = 0.145
    y_thr_noscen = 0.080


    path_ncol = len(pathways) if len(pathways) <= 7 else 4
    path_handles = [Line2D([0],[0], marker="o", linestyle="",
                           markerfacecolor=colors[p], markeredgecolor="none",
                           markersize=7, label=p) for p in pathways]
    fig.legend(path_handles, pathways, loc="lower center", bbox_to_anchor=(0.5, y_path),
               ncol=path_ncol, frameon=False, fontsize=LEGEND_FONTSIZE,
               handletextpad=0.4, columnspacing=1.2)

    ph = Line2D([0],[0], marker="o", color="black", linestyle="none",
                markerfacecolor="black", markersize=6)
    fh = Line2D([0],[0], marker="o", color="black", linestyle="none",
                markerfacecolor="none", markersize=6)
    fig.legend([ph, fh], ["Pass (≤ threshold)", "Fail (> threshold)"], loc="lower center",
               bbox_to_anchor=(0.5, y_thr if scenario_markers else y_thr_noscen), ncol=2, frameon=False,
               title="Threshold", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
               handletextpad=0.6, columnspacing=2.0)

    if scenario_markers:
        scen_handles = [Line2D([0],[0], marker=SCEN_MARKERS.get(sc, "o"), linestyle="none",
                               markerfacecolor="white", markeredgecolor="black", color="black",
                               markersize=7, label=sc) for sc in scenarios]
        fig.legend(scen_handles, scenarios, loc="lower center", bbox_to_anchor=(0.5, y_scen),
                   ncol=4, frameon=False, title="Scenario",
                   fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
                   handletextpad=0.5, columnspacing=1.8)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def fig33_raincloud_passfail(df, outpath: Path, regions, years, scenarios, policy_points,
                             scenario_markers=False):
    year_col, region_col, scen_col, path_col = infer_columns(df)
    ci_col, ci_factor = infer_ci_col(df)
    df = df.copy()
    df["_ci_g_per_MJ"] = df[ci_col].astype(float) * ci_factor

    thr_kg = threshold_series(years, policy_points)
    thr_g = {y: thr_kg[y] * 1000.0 for y in thr_kg}

    pathways = get_pathway_order(df, path_col)
    colors = make_color_map(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    pos = np.arange(len(years)) + 1

    for ax, rg in zip(axes, regions):
        sub = df[df[region_col] == rg]

        # prepare dists per year for half violin
        dists = []
        for y in years:
            yy = sub[sub[year_col] == y]["_ci_g_per_MJ"].values
            dists.append(yy if len(yy) else np.array([np.nan]))

        half_violin(ax, dists, pos, width=0.70, side="right", alpha=0.14, edge_lw=0.7)

        for i, y in enumerate(years):
            dyy = sub[sub[year_col] == y]
            if dyy.empty:
                continue
            vals = dyy["_ci_g_per_MJ"].values
            thr = thr_g[int(y)]
            is_pass = vals <= thr
            rng = deterministic_rng("Fig33_rain", rg, y)

            # pass/fail dots: left cloud
            if scenario_markers:
                for sc in scenarios:
                    dsc = dyy[dyy[scen_col] == sc]
                    if dsc.empty:
                        continue
                    vals_sc = dsc["_ci_g_per_MJ"].values
                    pass_sc = vals_sc <= thr
                    cols = [colors[p] for p in dsc[path_col].tolist()]
                    # fail hollow
                    if np.any(~pass_sc):
                        idx = np.where(~pass_sc)[0]
                        scatter_cloud(ax, pos[i]-0.12, vals_sc[~pass_sc], [cols[j] for j in idx],
                                      marker=SCEN_MARKERS.get(sc, "o"), s=18, alpha=0.9,
                                      half_width=0.22, rng=rng, facecolor="none", zorder=4)
                    if np.any(pass_sc):
                        idx = np.where(pass_sc)[0]
                        ax.scatter(jitter_x(pos[i]-0.12, len(idx), 0.22, rng), vals_sc[pass_sc],
                                   s=18, c=[cols[j] for j in idx], marker=SCEN_MARKERS.get(sc, "o"),
                                   alpha=0.80, edgecolors="none", zorder=3)
            else:
                cols = [colors[p] for p in dyy[path_col].tolist()]
                if np.any(~is_pass):
                    idx = np.where(~is_pass)[0]
                    scatter_cloud(ax, pos[i]-0.12, vals[~is_pass], [cols[j] for j in idx],
                                  marker="o", s=18, alpha=0.9, half_width=0.22, rng=rng,
                                  facecolor="none", zorder=4)
                if np.any(is_pass):
                    idx = np.where(is_pass)[0]
                    ax.scatter(jitter_x(pos[i]-0.12, len(idx), 0.22, rng), vals[is_pass],
                               s=18, c=[cols[j] for j in idx], marker="o",
                               alpha=0.80, edgecolors="none", zorder=3)

            # percent label: outside threshold segment, lower-right
            pass_rate = float(np.mean(is_pass)) * 100.0
            x_text = pos[i] + 0.62/2 + 0.06
            y_text = thr - 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0])
            ax.text(x_text, y_text, f"{pass_rate:.0f}%", ha="left", va="top",
                    fontsize=8.5, color="0.25", zorder=6, clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.65))

            # Threshold segment (per-year; avoid polyline)
            draw_threshold_segment(ax, pos[i], thr, 0.62/2)


        ax.set_title(rg.replace("GCAM 6.0 NGFS|", ""), loc="left", fontweight="bold")
        ax.grid(True, axis="y", alpha=0.2)

    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in axes[::2]:
        ax.set_ylabel("CI (gCO$_2$e/MJ)")

    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(years)

    # --- legends (bottom, inside reserved margin) ---
    # --- legends (compact, 3 clearly separated rows) ---
    bottom_space = 0.26 if scenario_markers else 0.16
    fig.tight_layout(rect=(0, bottom_space, 1, 1))

    # legend row positions (figure fraction)
    y_path = 0.020
    y_scen = 0.085
    y_thr  = 0.145
    y_thr_noscen = 0.080


    path_ncol = len(pathways) if len(pathways) <= 7 else 4
    path_handles = [Line2D([0],[0], marker="o", linestyle="",
                           markerfacecolor=colors[p], markeredgecolor="none",
                           markersize=7, label=p) for p in pathways]
    fig.legend(path_handles, pathways, loc="lower center", bbox_to_anchor=(0.5, y_path),
               ncol=path_ncol, frameon=False, fontsize=LEGEND_FONTSIZE,
               handletextpad=0.4, columnspacing=1.2)

    ph = Line2D([0],[0], marker="o", color="black", linestyle="none",
                markerfacecolor="black", markersize=6)
    fh = Line2D([0],[0], marker="o", color="black", linestyle="none",
                markerfacecolor="none", markersize=6)
    fig.legend([ph, fh], ["Pass (≤ threshold)", "Fail (> threshold)"], loc="lower center",
               bbox_to_anchor=(0.5, y_thr if scenario_markers else y_thr_noscen), ncol=2, frameon=False,
               title="Threshold", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
               handletextpad=0.6, columnspacing=2.0)

    if scenario_markers:
        scen_handles = [Line2D([0],[0], marker=SCEN_MARKERS.get(sc, "o"), linestyle="none",
                               markerfacecolor="white", markeredgecolor="black", color="black",
                               markersize=7, label=sc) for sc in scenarios]
        fig.legend(scen_handles, scenarios, loc="lower center", bbox_to_anchor=(0.5, y_scen),
                   ncol=4, frameon=False, title="Scenario",
                   fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
                   handletextpad=0.5, columnspacing=1.8)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-outdir", required=True, type=str, help=".../policy_layer")
    ap.add_argument("--regions", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2025, 2030, 2035, 2040, 2045, 2050])
    args = ap.parse_args()

    policy_outdir = Path(args.policy_outdir)
    years = [int(y) for y in args.years]
    regions = list(args.regions)
    scenarios = list(args.scenarios)

    merged_csv = policy_outdir / "policy_merged_metrics.csv"
    filtered_csv = policy_outdir / "policy_filtered_metrics.csv"
    if not merged_csv.exists():
        raise FileNotFoundError(f"Missing baseline merged metrics: {merged_csv}")
    if not filtered_csv.exists():
        raise FileNotFoundError(f"Missing policy filtered metrics: {filtered_csv}")

    df_base = pd.read_csv(merged_csv)
    df_pol = pd.read_csv(filtered_csv)

    policy_points = load_policy_config_points(policy_outdir)

    # Output directories (match paper_figures_chapters structure, but *extra* filenames)
    out_ch34 = policy_outdir / "paper_figures_chapters" / "ch3_4_compliance_cost"
    out_ch33 = policy_outdir / "paper_figures_chapters" / "ch3_3_lcfs_feasibility"

    # ---- Fig F extra figures: policy only; raincloud; with/without scenario markers
    figF_raincloud(df_pol, out_ch34 / "Fig_F_net_cost_raincloud_policy.png",
                   regions, years, scenarios, scenario_markers=False)
    figF_raincloud(df_pol, out_ch34 / "Fig_F_net_cost_raincloud_policy_scenmarkers.png",
                   regions, years, scenarios, scenario_markers=True)

    # ---- Fig F extra figures: LCFS credit balance (Credit vs Deficit); raincloud; with/without scenario markers
    figF_credit_balance_raincloud(df_pol, out_ch34 / "Fig_F_credit_balance_raincloud_policy.png",
                                 regions, years, scenarios, scenario_markers=False)
    figF_credit_balance_raincloud(df_pol, out_ch34 / "Fig_F_credit_balance_raincloud_policy_scenmarkers.png",
                                 regions, years, scenarios, scenario_markers=True)

    # ---- Fig 3.3 extra figures: baseline only; min–max rectangle + pass/fail; raincloud pass/fail; with/without scenario markers
    fig33_minmax_passfail(df_base, out_ch33 / "Fig_3_3_minmax_passfail_baseline.png",
                          regions, years, scenarios, policy_points, scenario_markers=False)
    fig33_minmax_passfail(df_base, out_ch33 / "Fig_3_3_minmax_passfail_baseline_scenmarkers.png",
                          regions, years, scenarios, policy_points, scenario_markers=True)

    fig33_raincloud_passfail(df_base, out_ch33 / "Fig_3_3_raincloud_passfail_baseline.png",
                             regions, years, scenarios, policy_points, scenario_markers=False)
    fig33_raincloud_passfail(df_base, out_ch33 / "Fig_3_3_raincloud_passfail_baseline_scenmarkers.png",
                             regions, years, scenarios, policy_points, scenario_markers=True)

    print("✅ Extra figures written to:")
    print(f"  - {out_ch34}")
    print(f"  - {out_ch33}")

if __name__ == "__main__":
    main()
