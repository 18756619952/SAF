"""
make_paper_figures.py (clean rewrite)
-------------------------------------
Generates ONLY:
  1) Line plots (baseline=merged; policy-feasible=filtered)
  2) Fig F: Net cost boxplot + jittered points (4 regions)
  3) Fig I: Break-even credit price vs UCO (4 regions, fixed scenario)

CLI compatible with run_all.py (accepts but does not use):
  --pareto-year, --space-year, --heatmap-source

Inputs (from policy_layer):
  - policy_merged_metrics.csv
  - policy_filtered_metrics.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import hashlib

# Color palette: Muted (Tableau/Seaborn-style)
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



DEFAULT_LCFS_CI_STANDARD_POINTS_YEAR_KGCO2_PER_MJ: list[tuple[int, float]] = [
    (2025, 0.0890),
    (2030, 0.0623),
    (2045, 0.0089),
]
def make_path_color(pathways):
    """Deterministic mapping from pathway name -> muted colors.
    PtL_DAC is always blue.
    """
    pathways = list(pathways)
    mapping = {}
    # Force PtL_DAC to blue
    if 'PtL_DAC' in pathways:
        mapping['PtL_DAC'] = MUTED_10[0]
        pathways = [p for p in pathways if p != 'PtL_DAC']
    # Assign remaining pathways in order (skip blue)
    palette = MUTED_10[1:]
    for i, p in enumerate(pathways):
        mapping[p] = palette[i % len(palette)]
    return mapping
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle




def apply_rcparams() -> None:
    """Compact, journal-like defaults (no constrained_layout)."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "grid.linewidth": 0.35,
        "grid.alpha": 0.25,
    })


def beeswarm_x_offsets_1d(
    ax: plt.Axes,
    x_base: float,
    y_vals: np.ndarray,
    max_jitter: float,
    s_points2: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Compute x-offsets (data units) to reduce point overlap.

    Uses a 1D beeswarm in *pixel space* (marker-size aware), then maps offsets back to data units.
    Only x is moved; y is unchanged. Deterministic given `rng`.
    """
    y_vals = np.asarray(y_vals, dtype=float)
    n = len(y_vals)
    if n <= 1:
        return np.zeros(n, dtype=float)

    radius_pt = (float(s_points2) ** 0.5) / 2.0
    radius_px = radius_pt * ax.figure.dpi / 72.0
    min_dist = 2.05 * radius_px

    order = np.argsort(y_vals + rng.uniform(-1e-9, 1e-9, size=n))
    offsets = np.zeros(n, dtype=float)
    placed: list[tuple[float, float]] = []

    step_px = 0.92 * min_dist

    for idx in order:
        y = float(y_vals[idx])

        x0_pix, _ = ax.transData.transform((x_base, y))
        x1_pix, _ = ax.transData.transform((x_base + 1.0, y))
        px_per_data = float(x1_pix - x0_pix)
        dx_data = 0.01 if px_per_data == 0 else (step_px / px_per_data)

        kmax = int(max(1, np.floor(max_jitter / max(dx_data, 1e-12))))
        candidates = [0.0]
        for k in range(1, kmax + 1):
            candidates.append(+k * dx_data)
            candidates.append(-k * dx_data)

        chosen = 0.0
        for dx in candidates:
            dx = float(np.clip(dx, -max_jitter, +max_jitter))
            x_pix, y_pix = ax.transData.transform((x_base + dx, y))

            ok = True
            for xp, yp in placed:
                if (x_pix - xp) ** 2 + (y_pix - yp) ** 2 < (min_dist ** 2):
                    ok = False
                    break
            if ok:
                chosen = dx
                placed.append((x_pix, y_pix))
                break

        offsets[idx] = chosen

    return offsets



def deterministic_spread_offsets(n: int, half_width: float, margin: float = 0.06) -> np.ndarray:
    """Deterministically spread n points within [-half_width+margin, +half_width-margin].
    No randomness; does not change y-values (only x positions).
    """
    if n <= 1:
        return np.zeros(n, dtype=float)
    usable = max(0.0, float(half_width) - float(margin))
    if usable <= 0.0:
        return np.zeros(n, dtype=float)
    return np.linspace(-usable, usable, n, dtype=float)


def beeswarm_x_positions(
    ax: plt.Axes,
    x_center: float,
    y_vals: np.ndarray,
    marker_size: float,
    max_span: float,
    pad_px: float = 0.8,
) -> np.ndarray:
    """Compute x-positions for a beeswarm (swarm) plot around x_center.

    - Keeps y values unchanged (only moves x for visibility).
    - Deterministic: no randomness.
    - Uses pixel-space collision checks so it is stable across axis scales.

    Parameters
    ----------
    ax : matplotlib Axes
    x_center : float
        Central x position in data coordinates.
    y_vals : array-like
        Y values in data coordinates.
    marker_size : float
        Scatter size passed to ax.scatter (points^2).
    max_span : float
        Maximum absolute x offset in data units (beeswarm bandwidth).
    pad_px : float
        Extra padding between markers in pixels.
    """
    y = np.asarray(y_vals, dtype=float)
    n = int(y.size)
    if n <= 1 or max_span <= 0:
        return np.full(n, float(x_center), dtype=float)

    # marker radius in pixels (approx): size is area in points^2
    fig = ax.figure
    dpi = float(fig.dpi) if fig is not None else 100.0
    r_pt = (np.sqrt(float(marker_size)) / 2.0)
    r_px = r_pt * dpi / 72.0
    min_sep = 2.0 * r_px + float(pad_px)

    # Candidate offsets: 0, +d, -d, +2d, -2d, ... within [-max_span, +max_span]
    step = max_span / 12.0
    step = max(step, 1e-6)
    ks = np.arange(0, int(np.ceil(max_span / step)) + 1)
    cand = np.empty(2 * len(ks) - 1, dtype=float)
    cand[0] = 0.0
    j = 1
    for k in ks[1:]:
        cand[j] = +k * step
        cand[j + 1] = -k * step
        j += 2
    cand = cand[np.abs(cand) <= max_span + 1e-12]

    # Place points from low to high y (stable). If ties, keep original order.
    order = np.lexsort((np.arange(n), y))
    placed_px = []  # list of (px, py)
    xs = np.full(n, float(x_center), dtype=float)

    for idx in order:
        yi = float(y[idx])
        # try candidates until no collision
        chosen = 0.0
        for off in cand:
            xi = float(x_center + off)
            px, py = ax.transData.transform((xi, yi))
            ok = True
            for (pxj, pyj) in placed_px:
                if abs(px - pxj) < min_sep and abs(py - pyj) < min_sep:
                    # quick reject; for more exactness use euclidean distance
                    if (px - pxj) ** 2 + (py - pyj) ** 2 < (min_sep ** 2):
                        ok = False
                        break
            if ok:
                chosen = off
                placed_px.append((px, py))
                break
        xs[idx] = float(x_center + chosen)

    return xs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def short_region(r: str) -> str:
    """Drop model prefix like 'GCAM 6.0 NGFS|' for cleaner panel titles."""
    return r.split("|")[-1] if "|" in r else r


def clean_region_name(r: str) -> str:
    """Backward-compatible alias used by Fig 3.2/3.3 plotting functions."""
    return short_region(r)


def save_png_pdf(fig: plt.Figure, outpath: Path) -> None:
    """Save both .png and .pdf with tight bounding box."""
    outpath = Path(outpath)
    ensure_dir(outpath.parent)
    fig.savefig(outpath, bbox_inches="tight")
    if outpath.suffix.lower() != ".pdf":
        fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")


def pick_col(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not find {label}. Tried: {candidates}. "
        f"Available (first 40): {list(df.columns)[:40]}"
    )


def available_years(df: pd.DataFrame, y_min: int, y_max: int) -> List[int]:
    ys = sorted({int(y) for y in df["Year"].dropna().unique() if y_min <= int(y) <= y_max})
    return ys




def lineplots_for_all_scenarios(
    df: pd.DataFrame,
    scenarios: List[str],
    regions: List[str],
    years: List[int],
    outdir: Path,
    cost_col: str,
    ci_col: str,
    tag: str,
) -> None:
    """
    For each scenario: 2×2 region facets, lines by pathway.
      - cost plotted as USD/GJ (from USD/MJ × 1000)
      - CI plotted as gCO2e/MJ (from kg/MJ × 1000)
    """
    apply_rcparams()
    ensure_dir(outdir)

    dd = df[df["Year"].isin(years) & df["Scenario"].isin(scenarios) & df["Region"].isin(regions)].copy()
    if dd.empty:
        return

    pathways = sorted(dd["pathway"].unique())
    
    path_color = make_path_color(pathways)

    series = [
        ("cost", cost_col, "Cost (USD/GJ)", 1000.0),
        ("ci",   ci_col,   "Carbon intensity (gCO₂e/MJ)", 1000.0),
    ]

    for metric, col, ylabel, conv in series:
        for sc in scenarios:
            sub_sc = dd[dd["Scenario"] == sc].copy()
            if sub_sc.empty:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True, sharey=True)
            axes = np.ravel(axes)

           
            yvals = (sub_sc[col].astype(float).to_numpy() * conv)
            yvals = yvals[np.isfinite(yvals)]
            if yvals.size > 0:
                ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
                pad = 0.05 * (ymax - ymin) if ymax > ymin else (0.05 * abs(ymax) + 1.0)
                ylim = (ymin - pad, ymax + pad)
            else:
                ylim = None

            for idx, (ax, rg) in enumerate(zip(axes, regions)):
                row = idx // 2
                sub = sub_sc[sub_sc["Region"] == rg].copy()
                if sub.empty:
                    ax.set_axis_off()
                    continue

                for p in pathways:
                    s = sub[sub["pathway"] == p].sort_values("Year")
                    if s.empty:
                        continue
                    ax.plot(
                        s["Year"].astype(int).to_numpy(),
                        (s[col].astype(float).to_numpy() * conv),
                        color=path_color[p],
                        marker="o",
                        markersize=3.0,
                    )

                ax.set_title(short_region(rg), loc="left", fontweight="bold")
                ax.grid(True, axis="y")
                if ylim is not None:
                    ax.set_ylim(*ylim)
                if row == 0:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel("Year")
            fig.supylabel(ylabel)

            
            handles = [Line2D([0], [0], color=path_color[p], label=p) for p in pathways]
            fig.legend(
                handles=handles,
                loc="lower center",
                ncol=min(4, max(1, len(handles))),
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
                title=None,
            )

            fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.16, wspace=0.10, hspace=0.22)

            fname = f"{metric}_lineplots_{tag}_{sc.replace(' ', '_')}.png"
            save_png_pdf(fig, outdir / fname)
            plt.close(fig)




def figF_net_cost_boxplot_with_points(
    df: pd.DataFrame,
    scenarios: List[str],
    regions: List[str],
    years: List[int],
    net_cost_col: str,
    outpath: Path,
    scenario_markers: bool = False,
    pathways_order: Optional[List[str]] = None,
) -> None:
    """
    2×2 region facets.
    Per year: boxplot of net cost pooled over all pathways×scenarios, with a non-overlapping scatter overlay.

    The overlay uses a deterministic jitter cloud (x-only) within each year's box width.
    net_cost_col expected in USD/MJ; plotted as USD/GJ.
    """
    apply_rcparams()
    ensure_dir(outpath.parent)

    dd = df[df["Year"].isin(years) & df["Scenario"].isin(scenarios) & df["Region"].isin(regions)].copy()
    if dd.empty:
        return

    if pathways_order is None:
        pathways = sorted(dd["pathway"].unique())
    else:
        pathways = [p for p in pathways_order if p in set(dd["pathway"].unique())]
        pathways += sorted(set(dd["pathway"].unique()) - set(pathways))

    path_color = make_path_color(pathways)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), sharey=True)
    axes = np.ravel(axes)

    x_positions = {y: i + 1 for i, y in enumerate(years)}
    xticks = [x_positions[y] for y in years]
    xticklabels = [str(y) for y in years]

    box_width = 0.60
    box_half = box_width / 2.0

    # Scatter styling
    point_s = 16
    point_alpha = 0.65

    
    scen_marker = {
        "Current Policies": "o",
        "Delayed transition": "s",
        "Fragmented World": "^",
        "Net Zero 2050": "D",
    } if scenario_markers else {}

    
    max_jitter = 0.90 * box_half

    for idx, (ax, rg) in enumerate(zip(axes, regions)):
        row = idx // 2
        sub = dd[dd["Region"] == rg].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        
        vals = []
        for y in years:
            v = (sub[sub["Year"] == y][net_cost_col].astype(float) * 1000.0).to_numpy()  # USD/GJ
            vals.append(v)

        ax.boxplot(
            vals,
            positions=xticks,
            widths=box_width,
            showfliers=False,
            showcaps=True,
            boxprops=dict(color="0.2", linewidth=1.0),
            whiskerprops=dict(color="0.2", linewidth=1.0),
            capprops=dict(color="0.2", linewidth=1.0),
            medianprops=dict(color="0.4", linestyle="--", linewidth=1.2),
        )
        
        for y in years:
            s_y = sub[sub["Year"] == y].copy()
            if s_y.empty:
                continue

            s_y["_net"] = s_y[net_cost_col].astype(float) * 1000.0  # USD/GJ

            
            s_y = s_y.sort_values(["pathway", "Scenario", "_net"]).reset_index(drop=True)

            seed_str = f"{rg}|{int(y)}|net|beeswarm"
            seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)

            x_base = float(x_positions[y])
            y_vals = s_y["_net"].to_numpy(dtype=float)
                        # Deterministic jitter so repeated runs are stable
            x_offsets = rng.uniform(-max_jitter, max_jitter, size=len(y_vals))

            for i in range(len(s_y)):
                rowp = s_y.iloc[i]
                x = x_base + float(x_offsets[i])
                yv = float(rowp["_net"])
                pth = str(rowp["pathway"])
                mk = scen_marker.get(str(rowp["Scenario"]), "o") if scenario_markers else "o"
                col = path_color.get(pth, "0.4")
                if scenario_markers:
                    
                    ax.scatter(
                        x, yv,
                        s=point_s,
                        marker=mk,
                        facecolors="none",
                        edgecolors=col,
                        alpha=point_alpha,
                        linewidths=1.0,
                        zorder=3,
                    )
                else:
                    ax.scatter(
                        x, yv,
                        s=point_s,
                        marker=mk,
                        c=[col],
                        alpha=point_alpha,
                        linewidths=0.0,
                        zorder=3,
                    )

            s_y.drop(columns=["_net"], inplace=True, errors="ignore")

        ax.set_title(short_region(rg), loc="left", fontweight="bold")
        ax.set_xticks(xticks)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("Year")
        ax.grid(True, axis="y", alpha=0.25)

    fig.supylabel("Net cost (USD/GJ)")
    # Legends
    path_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=path_color[p], markeredgecolor="none",
               markersize=6, label=p)
        for p in pathways
    ]

    if scenario_markers:
        scen_handles = [
            Line2D([0], [0], marker=scen_marker.get(s, "o"), color="none",
                   markerfacecolor="none", markeredgecolor="0.25", markeredgewidth=1.0,
                   markersize=6, label=s)
            for s in scenarios
        ]
        leg_thr = fig.legend(
            handles=scen_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.15),
            ncol=min(4, max(1, len(scen_handles))),
            frameon=False,
            title="Scenario",
        )
        fig.add_artist(leg_thr)

        fig.legend(
            handles=path_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=min(4, max(1, len(path_handles))),
            frameon=False,
            title="Pathway",
        )
        bottom_space = 0.30
    else:
        fig.legend(
            handles=path_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.08),
            ncol=min(4, max(1, len(path_handles))),
            frameon=False,
            title=None,
        )
        bottom_space = 0.22

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=bottom_space, wspace=0.10, hspace=0.22)


    save_png_pdf(fig, outpath)
    plt.close(fig)



def figI_breakevenP_allpaths_vs_uco(
    df: pd.DataFrame,
    regions: List[str],
    years: List[int],
    cost_col: str,
    ci_col: str,
    scenario_fixed: str,
    uco_name: str,
    cap: float,
    outpath: Path,
    pathways_order: Optional[List[str]] = None,
) -> None:
    """
    2×2 region facets. Scenario fixed (e.g., 'Net Zero 2050').
    Grouped bars by year; each bar is break-even credit price P (USD/tCO2) for pathway to beat UCO.

    Break-even:
      (cost_p + P * (CI_p - CI_std)) = (cost_uco + P * (CI_uco - CI_std))
      => P = -(cost_p - cost_uco) / ((CI_p - CI_uco) in tCO2/MJ)

    cost: USD/MJ; CI: kg/MJ. We compute P in USD/tCO2.
    """
    apply_rcparams()
    ensure_dir(outpath.parent)

    dd = df[(df["Year"].isin(years)) & (df["Region"].isin(regions)) & (df["Scenario"] == scenario_fixed)].copy()
    if dd.empty:
        return

    if pathways_order is None:
        pathways = sorted(dd["pathway"].unique())
    else:
        pathways = [p for p in pathways_order if p in set(dd["pathway"].unique())]
        pathways += sorted(set(dd["pathway"].unique()) - set(pathways))

    
    path_color = make_path_color(pathways)

    order = [p for p in pathways if p != uco_name]
    if not order:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), sharey=True)
    axes = np.ravel(axes)

    x = np.arange(len(years))
    width = 0.82 / max(len(order), 1)

    for idx, (ax, rg) in enumerate(zip(axes, regions)):
        row = idx // 2
        sub = dd[dd["Region"] == rg]
        if sub.empty:
            ax.set_axis_off()
            continue

        uco = sub[sub["pathway"] == uco_name].set_index("Year")[[cost_col, ci_col]]

        for i, p in enumerate(order):
            s = sub[sub["pathway"] == p].set_index("Year")[[cost_col, ci_col]]
            mm = s.join(uco, lsuffix="_p", rsuffix="_uco", how="inner")
            if mm.empty:
                continue

            P_be = []
            for y in years:
                if y not in mm.index:
                    P_be.append(np.nan)
                    continue
                d_cost = float(mm.loc[y, f"{cost_col}_p"] - mm.loc[y, f"{cost_col}_uco"])  # USD/MJ
                d_ci_t = float((mm.loc[y, f"{ci_col}_p"] - mm.loc[y, f"{ci_col}_uco"]) / 1000.0)  # tCO2/MJ
                if abs(d_ci_t) < 1e-12:
                    P_be.append(np.nan)
                else:
                    P_be.append((-d_cost) / d_ci_t)

            P_plot = np.clip(np.array(P_be, dtype=float), -cap, cap)
            ax.bar(
                x + (i - (len(order) - 1) / 2) * width,
                P_plot,
                width=width,
                label=p,
                color=path_color.get(p, "0.4"),
            )

        ax.set_title(short_region(rg), loc="left", fontweight="bold")
        ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels([str(y) for y in years])
            ax.set_xlabel("Year")
        ax.grid(True, axis="y")
    fig.supylabel("Break-even price (USD/tCO₂)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        title=None,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.16, wspace=0.10, hspace=0.22)

    save_png_pdf(fig, outpath)
    plt.close(fig)



def figI_breakevenP_allpaths_vs_winner(
    df: pd.DataFrame,
    regions: List[str],
    years: List[int],
    cost_col: str,
    ci_col: str,
    scenario_fixed: str,
    cap: float,
    outpath: Path,
    pathways_order: Optional[List[str]] = None,
) -> None:
    """
    2×2 region facets. Scenario fixed (e.g., 'Net Zero 2050').
    Grouped bars by year; each bar is break-even credit price P (USD/tCO2) for pathway to reach COST PARITY
    with the *compliance winner* of that same (Region, Year, Scenario).

    Winner definition (per Region-Year within the fixed Scenario): pathway with minimum `cost_col`.

    Break-even (parity with winner):
      (cost_p + P * (CI_p - CI_std)) = (cost_w + P * (CI_w - CI_std))
      => P = -(cost_p - cost_w) / ((CI_p - CI_w) in tCO2/MJ)

    cost: USD/MJ; CI: kg/MJ. We compute P in USD/tCO2.

    Notes
    -----
    - Winner can change by year.
    - If a pathway is the winner in a given year, its parity threshold is 0 for that year.
    """
    apply_rcparams()
    ensure_dir(outpath.parent)

    dd = df[(df["Year"].isin(years)) & (df["Region"].isin(regions)) & (df["Scenario"] == scenario_fixed)].copy()
    if dd.empty:
        return

    # pathway ordering
    if pathways_order is None:
        pathways = sorted(dd["pathway"].unique())
    else:
        pathways = [p for p in pathways_order if p in set(dd["pathway"].unique())]
        pathways += sorted(set(dd["pathway"].unique()) - set(pathways))

    # Use the same pathway→color mapping as earlier figures
    path_color = make_path_color(pathways)

    if not pathways:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), sharey=True)
    axes = np.ravel(axes)

    x = np.arange(len(years))
    width = 0.82 / max(len(pathways), 1)

    for idx, (ax, rg) in enumerate(zip(axes, regions)):
        row = idx // 2
        sub = dd[dd["Region"] == rg].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        # Determine winner pathway per year (deterministic tie-break by pathway name)
        winners = {}
        for y in years:
            sy = sub[sub["Year"] == y]
            if sy.empty:
                continue
            # ensure numeric
            sy = sy.copy()
            sy[cost_col] = pd.to_numeric(sy[cost_col], errors='coerce')
            min_cost = sy[cost_col].min()
            if pd.isna(min_cost):
                continue
            cand = sy[sy[cost_col] == min_cost]
            winners[y] = sorted(cand["pathway"].astype(str).unique())[0]

        # Build year-indexed winner table
        w_rows = []
        for y in years:
            wp = winners.get(y, None)
            if wp is None:
                continue
            rr = sub[(sub["Year"] == y) & (sub["pathway"] == wp)][["Year", cost_col, ci_col]].copy()
            if rr.empty:
                continue
            rr = rr.iloc[[0]]
            rr["winner_pathway"] = wp
            w_rows.append(rr)
        if not w_rows:
            ax.set_axis_off()
            continue

        wtab = pd.concat(w_rows, ignore_index=True).set_index("Year")

        for i, p in enumerate(pathways):
            s = sub[sub["pathway"] == p].set_index("Year")[[cost_col, ci_col]].copy()
            if s.empty:
                continue

            # align with winner per year
            mm = s.join(wtab[[cost_col, ci_col]], lsuffix="_p", rsuffix="_w", how="inner")
            if mm.empty:
                continue

            P_be = []
            for y in years:
                if y not in mm.index:
                    P_be.append(np.nan)
                    continue
                # If this pathway is the winner that year, parity threshold is 0
                if winners.get(y, None) == p:
                    P_be.append(0.0)
                    continue
                d_cost = float(mm.loc[y, f"{cost_col}_p"] - mm.loc[y, f"{cost_col}_w"])  # USD/MJ
                d_ci_t = float((mm.loc[y, f"{ci_col}_p"] - mm.loc[y, f"{ci_col}_w"]) / 1000.0)  # tCO2/MJ
                if abs(d_ci_t) < 1e-12:
                    P_be.append(np.nan)
                else:
                    P_be.append((-d_cost) / d_ci_t)

            P_plot = np.clip(np.array(P_be, dtype=float), -cap, cap)
            ax.bar(
                x + (i - (len(pathways) - 1) / 2) * width,
                P_plot,
                width=width,
                label=p,
                color=path_color.get(p, "0.4"),
            )

        ax.set_title(short_region(rg), loc="left", fontweight="bold")
        ax.axhline(0, linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_xticks(x)
        if row == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xticklabels([str(y) for y in years])
            ax.set_xlabel("Year")
        ax.grid(True, axis="y")

    fig.supylabel("Break-even price (USD/tCO₂)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(4, max(1, len(labels))),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        title=None,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.16, wspace=0.10, hspace=0.22)

    save_png_pdf(fig, outpath)
    plt.close(fig)


def fig32_cost_ci_scatter_facets(
    df: pd.DataFrame,
    scenarios: list[str],
    regions: list[str],
    years: list[int],
    cost_col: str,
    ci_col: str,
    outpath: Path,
    pathways_order: list[str] | None = None,
    scenario_fixed: str = "Net Zero 2050",
) -> None:
    """Fig 3.2 — Cost–CI trade-off (Scheme B: single scenario + insets).

    Design:
      - Main axes: all non-FT, non-PtL pathways as **trajectories** across model years.
        Only a **single terminal arrowhead** per pathway (direction over time).
      - Insets (per region):
          (i) PtL-DAC (large swing): dedicated inset (bottom-right).
         (ii) FT routes (tiny swing): two split micro-insets (upper-left), one per FT route.
              Uses **absolute CI/Cost**, but formats ticks with **offset notation** to avoid
              ugly decimals (e.g., 68.38) while keeping the line visible.

    Notes:
      - No interpolation: only connects model years.
      - No scatter markers: lines only.
      - No connector lines / zoom rectangles (avoid clutter).
      - Inset titles/labels omitted (pathway colors explained in global legend).
    """
    apply_rcparams()
    ensure_dir(outpath.parent)

    dd = df[
        df["Year"].isin(years)
        & (df["Scenario"] == scenario_fixed)
        & df["Region"].isin(regions)
    ].copy()
    if dd.empty:
        return

    # Pathway ordering to keep colors consistent with other figures
    if pathways_order is None:
        pathways = sorted(dd["pathway"].astype(str).unique())
    else:
        present = set(dd["pathway"].astype(str).unique())
        pathways = [p for p in pathways_order if p in present]
        pathways += sorted(present - set(pathways))

    path_color = make_path_color(pathways)

    # ---- unit normalization ----
    ci_vals = pd.to_numeric(dd[ci_col], errors="coerce").dropna()
    ci_factor = 1.0 if (len(ci_vals) and float(ci_vals.median()) > 5.0) else 1000.0  # kg/MJ → g/MJ
    dd["_ci_g"] = pd.to_numeric(dd[ci_col], errors="coerce") * ci_factor
    dd["_cost_gj"] = pd.to_numeric(dd[cost_col], errors="coerce") * 1000.0  # USD/MJ → USD/GJ
    dd = dd.dropna(subset=["_ci_g", "_cost_gj"])
    if dd.empty:
        return

    # Pathway groups for Scheme B
    ptl_name = "PtL_DAC"
    ft_names = ["FT_Miscanthus", "FT_Switchgrass"]
    main_paths = [p for p in pathways if p not in ([ptl_name] + ft_names)]

    from matplotlib.ticker import ScalarFormatter, MaxNLocator
    from matplotlib.patches import FancyArrowPatch

    def _traj_line_with_terminal_arrow(
        ax: plt.Axes,
        x: np.ndarray,
        y: np.ndarray,
        *,
        color: str,
        lw: float,
        alpha: float,
        z: int,
        arrow_scale: float = 9,
        shrinkA_pt: float = 0.0,
        shrinkB_pt: float = 2.5,
    ) -> None:
        """Polyline + ONE terminal arrowhead (robust against overdraw/clutter).

        - The last segment is drawn ONLY via FancyArrowPatch.
        - `shrinkB` trims the arrow so the head doesn't spill into nearby trajectories.
        - Slightly smaller default `mutation_scale` to reduce overlap.
        """
        if len(x) < 2:
            return

        # draw all segments except last
        if len(x) > 2:
            ax.plot(
                x[:-1],
                y[:-1],
                linestyle="-",
                linewidth=lw,
                alpha=alpha,
                color=color,
                zorder=z,
                solid_capstyle="butt",
                solid_joinstyle="miter",
            )

        # terminal arrow (last segment only)
        x0, y0 = float(x[-2]), float(y[-2])
        x1, y1 = float(x[-1]), float(y[-1])

        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=float(arrow_scale),
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=z + 1,
            shrinkA=float(shrinkA_pt),
            shrinkB=float(shrinkB_pt),
            capstyle="butt",
            joinstyle="miter",
        )
        ax.add_patch(arr)

    def _format_micro_ticks(axz: plt.Axes) -> None:
        """Clean, consistent ticks for micro-insets (no scientific, no auto-offset)."""
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter

        # fewer ticks to avoid label collisions in small inset boxes
        axz.xaxis.set_major_locator(MaxNLocator(2))
        axz.yaxis.set_major_locator(MaxNLocator(2))
        axz.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        axz.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        axz.tick_params(labelsize=6, length=2, pad=1)
        for sp in axz.spines.values():
            sp.set_linewidth(0.7)

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), sharex=False, sharey=False)
    axes = axes.flatten()

    for idx, (ax, rg) in enumerate(zip(axes, regions)):
        row = idx // 2
        subR = dd[dd["Region"] == rg].copy()
        if subR.empty:
            ax.set_axis_off()
            continue

        sub_main = subR[subR["pathway"].isin(main_paths)].copy()
        sub_ptl = subR[subR["pathway"] == ptl_name].copy()
        sub_ft = subR[subR["pathway"].isin(ft_names)].copy()

        # ---- main-panel limits (based on moderate-range pathways) ----
        base_for_lim = sub_main if len(sub_main) else subR
        x_min = float(base_for_lim["_ci_g"].min()); x_max = float(base_for_lim["_ci_g"].max())
        y_min = float(base_for_lim["_cost_gj"].min()); y_max = float(base_for_lim["_cost_gj"].max())
        x_rng = max(1e-6, (x_max - x_min))
        y_rng = max(1e-6, (y_max - y_min))
        ax.set_xlim(x_min - 0.12 * x_rng, x_max + 0.12 * x_rng)
        ax.set_ylim(y_min - 0.12 * y_rng, y_max + 0.12 * y_rng)

        ax.grid(True, axis="both", alpha=0.18)
        ax.set_title(short_region(rg), loc="left", fontweight="bold")

        # ---- main trajectories ----
        for pth in main_paths:
            subP = sub_main[sub_main["pathway"] == pth].sort_values("Year")
            if len(subP) < 2:
                continue
            x = subP["_ci_g"].to_numpy(dtype=float)
            y = subP["_cost_gj"].to_numpy(dtype=float)
            col = path_color.get(pth, "0.4")
            _traj_line_with_terminal_arrow(ax, x, y, color=col, lw=1.9, alpha=0.88, z=3, arrow_scale=9)

        # ---- faint context traces in main (so readers know what is inset) ----
        if len(sub_ft):
            for pth in ft_names:
                subP = sub_ft[sub_ft["pathway"] == pth].sort_values("Year")
                if len(subP) < 2:
                    continue
                col = path_color.get(pth, "0.4")
                ax.plot(
                    subP["_ci_g"].to_numpy(float),
                    subP["_cost_gj"].to_numpy(float),
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.12,
                    color=col,
                    zorder=1,
                    solid_capstyle="butt",
                )

        if len(sub_ptl) >= 2:
            subP = sub_ptl.sort_values("Year")
            colp = path_color.get(ptl_name, "0.4")
            ax.plot(
                subP["_ci_g"].to_numpy(float),
                subP["_cost_gj"].to_numpy(float),
                linestyle="-",
                linewidth=1.0,
                alpha=0.10,
                color=colp,
                zorder=1,
                solid_capstyle="butt",
            )

        # ---- inset geometry ----
        inset_ptl_box = [0.62, 0.08, 0.34, 0.26]  # bottom-right (smaller to reduce clutter)

        # ---- FT micro-insets (upper-left, split) ----
        if len(sub_ft) >= 2:
            inset_ft1_box = [0.04, 0.72, 0.16, 0.22]  # left (smaller + higher)
            inset_ft2_box = [0.22, 0.72, 0.16, 0.22]  # right (leave more gap)

            def _tight_limits_abs(axz: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
                x0, x1 = float(np.min(x)), float(np.max(x))
                y0, y1 = float(np.min(y)), float(np.max(y))
                xr = max(1e-9, (x1 - x0))
                yr = max(1e-9, (y1 - y0))
                # very small margins to keep the tiny trend visible
                mx = max(0.0005, 0.30 * xr)
                my = max(0.0005, 0.30 * yr)
                axz.set_xlim(x0 - mx, x1 + mx)
                axz.set_ylim(y0 - my, y1 + my)

            # FT_Miscanthus
            sp_m = sub_ft[sub_ft["pathway"] == "FT_Miscanthus"].sort_values("Year")
            if len(sp_m) >= 2:
                ax_ft1 = ax.inset_axes(inset_ft1_box, zorder=10)
                ax_ft1.set_facecolor("white")
                ax_ft1.patch.set_alpha(0.98)
                ax_ft1.grid(True, alpha=0.12)
                colm = path_color.get("FT_Miscanthus", "0.4")
                xft = sp_m["_ci_g"].to_numpy(dtype=float)
                yft = sp_m["_cost_gj"].to_numpy(dtype=float)
                _tight_limits_abs(ax_ft1, xft, yft)
                _traj_line_with_terminal_arrow(ax_ft1, xft, yft, color=colm, lw=2.2, alpha=0.98, z=12, arrow_scale=8)
                _format_micro_ticks(ax_ft1)
                ax_ft1.set_xlabel(""); ax_ft1.set_ylabel(""); ax_ft1.set_title("")

            # FT_Switchgrass
            sp_s = sub_ft[sub_ft["pathway"] == "FT_Switchgrass"].sort_values("Year")
            if len(sp_s) >= 2:
                ax_ft2 = ax.inset_axes(inset_ft2_box, zorder=10)
                ax_ft2.set_facecolor("white")
                ax_ft2.patch.set_alpha(0.98)
                ax_ft2.grid(True, alpha=0.12)
                cols = path_color.get("FT_Switchgrass", "0.4")
                xfs = sp_s["_ci_g"].to_numpy(dtype=float)
                yfs = sp_s["_cost_gj"].to_numpy(dtype=float)
                _tight_limits_abs(ax_ft2, xfs, yfs)
                _traj_line_with_terminal_arrow(ax_ft2, xfs, yfs, color=cols, lw=2.2, alpha=0.98, z=12, arrow_scale=8)
                _format_micro_ticks(ax_ft2)
                ax_ft2.set_xlabel(""); ax_ft2.set_ylabel(""); ax_ft2.set_title("")

        # ---- PtL-DAC inset ----
        if len(sub_ptl) >= 2:
            ax_in = ax.inset_axes(inset_ptl_box, zorder=10)
            ax_in.set_facecolor("white")
            ax_in.patch.set_alpha(0.98)
            ax_in.grid(True, alpha=0.15)

            subP = sub_ptl.sort_values("Year")
            x = subP["_ci_g"].to_numpy(dtype=float)
            y = subP["_cost_gj"].to_numpy(dtype=float)
            colp = path_color.get(ptl_name, "0.4")

            x0, x1 = float(np.min(x)), float(np.max(x))
            y0, y1 = float(np.min(y)), float(np.max(y))
            xr = max(1e-6, (x1 - x0))
            yr = max(1e-6, (y1 - y0))
            mx = max(0.8, 0.10 * xr)
            my = max(0.8, 0.10 * yr)
            ax_in.set_xlim(x0 - mx, x1 + mx)
            ax_in.set_ylim(y0 - my, y1 + my)

            _traj_line_with_terminal_arrow(ax_in, x, y, color=colp, lw=1.9, alpha=0.92, z=12, arrow_scale=9)
            ax_in.tick_params(labelsize=7, length=2)
            ax_in.set_xlabel(""); ax_in.set_ylabel(""); ax_in.set_title("")

        # Hide x tick labels on top row for compactness
        if row == 0:
            ax.tick_params(labelbottom=False)

    # Axis labels (only on left & bottom)
    axes[2].set_xlabel("CI (gCO$_2$e/MJ)")
    axes[3].set_xlabel("CI (gCO$_2$e/MJ)")
    axes[0].set_ylabel("Cost (USD/GJ)")
    axes[2].set_ylabel("Cost (USD/GJ)")

    # Legend: pathways (line proxy, no marker)
    path_handles = [
        Line2D([0], [0], color=path_color.get(p, "0.4"), lw=2.2, label=p)
        for p in pathways
    ]
    fig.legend(
        handles=path_handles,
        loc="lower center",
        ncol=min(7, len(path_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.03),
        handlelength=2.4,
        handletextpad=0.6,
        columnspacing=1.4,
    )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.18, wspace=0.08, hspace=0.14)
    save_png_pdf(fig, outpath)
    plt.close(fig)


def load_lcfs_ci_standard_g_per_MJ(policy_config_path: "Path|None", years: list[int]) -> dict[int, float]:
    """
    Return CI standard per year (gCO2e/MJ).
    - If policy_config_path is provided, reads policy_config.json.
    - If not found / None, falls back to DEFAULT_LCFS_CI_STANDARD_POINTS_YEAR_KGCO2_PER_MJ.
    Supports BOTH formats for anchor points:
      1) [[year, value_kg_per_MJ], ...]
      2) [{"year": year, "value": value_kg_per_MJ}, ...]
    Uses linear interpolation between anchor points and clamps outside the range.
    """
    if policy_config_path is not None:
        cfg = json.loads(Path(policy_config_path).read_text(encoding="utf-8"))
        pts_raw = cfg.get("lcfs_style", {}).get("ci_standard_points_year_kgCO2_per_MJ", [])
    else:
        pts_raw = []

    if not pts_raw:
        pts = list(DEFAULT_LCFS_CI_STANDARD_POINTS_YEAR_KGCO2_PER_MJ)
    else:
        first = pts_raw[0]
        if isinstance(first, dict):
            pts = [(int(p["year"]), float(p["value"])) for p in pts_raw]
        elif isinstance(first, (list, tuple)) and len(first) >= 2:
            pts = [(int(p[0]), float(p[1])) for p in pts_raw]
        else:
            raise ValueError("Unsupported format for lcfs_style.ci_standard_points_year_kgCO2_per_MJ")

    pts = sorted(pts, key=lambda t: t[0])  # kg/MJ
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float) * 1000.0  # g/MJ

    out = {}
    for y in years:
        if y <= xs.min():
            out[int(y)] = float(ys[xs.argmin()])
        elif y >= xs.max():
            out[int(y)] = float(ys[xs.argmax()])
        else:
            out[int(y)] = float(np.interp(y, xs, ys))
    return out



def fig32_cost_ci_grid_4x4(
    df: pd.DataFrame,
    regions: list[str],
    years: list[int],
    cost_col: str,
    ci_col: str,
    outpath: Path,
    *,
    pathways_order: list[str] | None = None,
    scenario_fixed: str = "Net Zero 2050",
    ptl_name: str = "PtL_DAC",
    ft_names: list[str] | None = None,
) -> None:
    """Fig 3.2 — 4×4 grid (no insets): 4 pathway groups × 4 regions.

    Rows (scales are independent per row; shared across columns within row):
      Row 1: FT_Miscanthus (absolute cost; zoomed)
      Row 2: FT_Switchgrass (absolute cost; zoomed)
      Row 3: ATJ/HEFA & other (medium swing; absolute scale; tighter padding)
      Row 4: PtL-DAC (large swing; absolute scale)

    Encoding (clean, journal style):
      - Connect model years only (no interpolation).
      - Direction by start/end markers (no along-track arrows):
          start (min year): open circle
          end   (max year): filled circle
    """
    apply_rcparams()
    ensure_dir(outpath.parent)

    # ----------------------------
    # Tick helpers (journal style)
    # ----------------------------
    def _choose_nice_step(span: float, max_n: int) -> float:
        """Choose a 'nice' step (1/2/2.5/5/10 × 10^k) so that <= max_n ticks cover the span."""
        if span <= 0 or not np.isfinite(span):
            return 1.0
        raw = span / max(max_n - 1, 1)
        mag = 10 ** int(np.floor(np.log10(raw)))
        for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
            step = mult * mag
            if step >= raw - 1e-12:
                return step
        return 10.0 * mag

    def _set_x_ticks_equal(ax, x0: float, x1: float, *,
                           n: int = 3,
                           inner_frac: float = 0.08,
                           pad_frac: float = 0.06,
                           force_decimals: int | None = None,
                           integer: bool = False,
                           snap: float = 10.0) -> None:
        """Set x ticks with (visually) equal spacing, symmetric around the inner-range center.

        Key goals:
          - 3–4 ticks only (n)
          - ticks do NOT sit on the plot boundary (inner_frac)
          - after formatting/rounding, displayed tick differences remain equal
          - avoid ugly '-0'
        """
        x0 = float(x0); x1 = float(x1)
        if not (np.isfinite(x0) and np.isfinite(x1)) or x0 == x1:
            return
        lo, hi = (x0, x1) if x0 < x1 else (x1, x0)
        span = hi - lo

        # xlim padding so edge labels don't collide with spines
        ax.set_xlim(lo - pad_frac * span, hi + pad_frac * span)

        # inner bounds for ticks (keeps ticks away from borders)
        il = lo + inner_frac * span
        ih = hi - inner_frac * span
        if ih <= il:
            il, ih = lo, hi

        center = 0.5 * (il + ih)

        if integer:
            # Use odd n for a clean symmetric integer grid
            if n % 2 == 0:
                n = n - 1
                if n < 3:
                    n = 3

            # choose a 'nice' integer-ish step and snap to multiples of `snap`
            raw_step = (ih - il) / max(n - 1, 1)
            step = _choose_nice_step(ih - il, max_n=n)
            # ensure step is not smaller than raw_step
            step = max(step, raw_step)

            # snap step and center
            if snap and snap > 0:
                step = round(step / snap) * snap if step >= snap else snap
                center = round(center / snap) * snap

            # symmetric ticks around center
            k0 = (n - 1) // 2
            ticks = [center + (i - k0) * step for i in range(n)]
            ticks = [float(round(t, 0)) for t in ticks]
            ticks = [0.0 if abs(t) < 1e-12 else t for t in ticks]  # kill -0

            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        else:
            # float ticks (FT rows): enforce equal displayed spacing by rounding step+start consistently
            n = 3 if n <= 3 else 4

            raw_step = (ih - il) / max(n - 1, 1)
            if force_decimals is None:
                # choose decimals from step size
                if raw_step <= 0:
                    decimals = 2
                else:
                    decimals = int(np.clip(np.ceil(-np.log10(raw_step)) + 1, 2, 4))
            else:
                decimals = int(force_decimals)

            step = round(raw_step, decimals)
            if step == 0:
                step = round(raw_step, 4) if raw_step > 0 else 0.01

            # compute symmetric start so ticks are centered
            if n % 2 == 1:
                k0 = (n - 1) // 2
                start = round(center - k0 * step, decimals)
            else:
                # even ticks: center lies midway between the two middle ticks
                start = round(center - (n - 1) / 2 * step, decimals)

            ticks = [round(start + i * step, decimals) for i in range(n)]
            # de-duplicate guard (rare but safe)
            uniq = []
            eps = 0.5 * 10 ** (-decimals)
            for t in ticks:
                if not uniq or abs(t - uniq[-1]) > eps:
                    uniq.append(float(t))
            ax.set_xticks(uniq)
            ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))

        ax.tick_params(axis="x", pad=3)

    if ft_names is None:
        ft_names = ["FT_Miscanthus", "FT_Switchgrass"]

    dd = df[
        df["Year"].isin(years)
        & (df["Scenario"] == scenario_fixed)
        & df["Region"].isin(regions)
    ].copy()
    if dd.empty:
        return

    # ---- Pathway ordering to keep colors consistent ----
    if pathways_order is None:
        pathways = sorted(dd["pathway"].astype(str).unique())
    else:
        present = set(dd["pathway"].astype(str).unique())
        pathways = [p for p in pathways_order if p in present]
        pathways += sorted(present - set(pathways))

    path_color = make_path_color(pathways)

    # ---- unit normalization ----
    ci_vals = pd.to_numeric(dd[ci_col], errors="coerce").dropna()
    ci_factor = 1.0 if (len(ci_vals) and float(ci_vals.median()) > 5.0) else 1000.0  # kg/MJ → g/MJ
    dd["_ci_g"] = pd.to_numeric(dd[ci_col], errors="coerce") * ci_factor
    dd["_cost_gj"] = pd.to_numeric(dd[cost_col], errors="coerce") * 1000.0  # USD/MJ → USD/GJ
    dd = dd.dropna(subset=["_ci_g", "_cost_gj"])
    if dd.empty:
        return

    # ---- Groups ----
    ft_mis = ["FT_Miscanthus"] if "FT_Miscanthus" in pathways else []
    ft_swg = ["FT_Switchgrass"] if "FT_Switchgrass" in pathways else []
    ptl_group = [ptl_name] if ptl_name in pathways else []
    mid_group = [p for p in pathways if p not in (set(ft_names) | {ptl_name})]

    groups = [
        ("FT_Miscanthus", ft_mis),
        ("FT_Switchgrass", ft_swg),
        ("ATJ/HEFA & other routes", mid_group),
        ("PtL-DAC", ptl_group),
    ]

    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    fig = plt.figure(figsize=(12.2, 9.2))
    gs = gridspec.GridSpec(nrows=4, ncols=4, figure=fig, wspace=0.22, hspace=0.30)

    # Pre-compute row-wise limits (shared within each row across regions)
    def _lims_for(paths: list[str], *, ycol: str, pad_x_frac: float, pad_y_frac: float):
        sub = dd[dd["pathway"].astype(str).isin(paths)]
        if sub.empty:
            return None
        xmin, xmax = float(sub["_ci_g"].min()), float(sub["_ci_g"].max())
        ymin, ymax = float(sub[ycol].min()), float(sub[ycol].max())
        # padding (avoid zero width)
        xr = max(xmax - xmin, 1e-6)
        yr = max(ymax - ymin, 1e-6)
        pad_x = float(pad_x_frac) * xr
        pad_y = float(pad_y_frac) * yr
        return (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

  
    row_specs = [
        {"ycol": "_cost_gj",  "pad_x": 0.03, "pad_y": 0.12},  # FT_Miscanthus
        {"ycol": "_cost_gj",  "pad_x": 0.03, "pad_y": 0.12},  # FT_Switchgrass
        {"ycol": "_cost_gj",  "pad_x": 0.03, "pad_y": 0.05},  # mid group (tighter)
        {"ycol": "_cost_gj",  "pad_x": 0.06, "pad_y": 0.08},  # PtL
    ]

    row_lims = [
        _lims_for(groups[r][1], ycol=row_specs[r]["ycol"], pad_x_frac=row_specs[r]["pad_x"], pad_y_frac=row_specs[r]["pad_y"])
        for r in range(4)

    ]

    
    if row_lims[2] is not None:
        x0, x1, y0, y1 = row_lims[2]
        # Prefer a clean 0–40 window when applicable (common for ATJ/HEFA CI range)
        x0_nice = 0.0 if x0 >= -1e-6 else np.floor(x0 / 10.0) * 10.0
        x1_nice = np.ceil(x1 / 10.0) * 10.0
        if x1_nice < 40.0:
            x1_nice = 40.0
        row_lims[2] = (x0_nice, x1_nice, y0, y1)

    # Plot helper

    def _plot_traj(ax: plt.Axes, sub: pd.DataFrame, color: str, *, ycol: str):
      
        sub = sub.sort_values("Year")
        x = sub["_ci_g"].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        if len(x) < 1:
            return

        line, = ax.plot(
            x, y,
            color=color,
            lw=2.0,
            alpha=0.95,
            solid_capstyle="round",
            solid_joinstyle="round",
            antialiased=True,
            zorder=2,
        )

        # Optional subtle outline (helps in dense panels; safe if patheffects unavailable)
        try:
            import matplotlib.patheffects as pe
            line.set_path_effects([pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
        except Exception:
            pass

        # Start/end markers (open circle = start; filled circle = end)
        ax.plot(
            x[0], y[0],
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle="None",
            zorder=3,
        )
        ax.plot(
            x[-1], y[-1],
            marker="o",
            markersize=4.4,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,
            linestyle="None",
            zorder=3,
        )

    # Build axes
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(4)]

    # Column headers (regions) — keep position, remove prefixes
    for c, reg in enumerate(regions):
        axes[0][c].set_title(clean_region_name(str(reg)), pad=6)

    # X label only on bottom row (keep), but show x tick labels for the first column in EACH row
    for c in range(4):
        axes[3][c].set_xlabel("CI (gCO$_2$e/MJ)")

    # Global y label (units only; no pathway labels).
    fig.text(0.008, 0.50, "Cost (USD/GJ)", rotation=90, va="center", ha="left", fontsize=10)
    # Formatting per axis
    for r in range(4):
        lims = row_lims[r]
        for c in range(4):
            ax = axes[r][c]
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)

            ax.tick_params(axis="both", which="major", length=3)

            # Default tick density
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))

            # FT rows: absolute cost, but keep x-axis ticks clean (avoid coordinate noise)
            if r in (0, 1):
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

            if lims is not None:
                ax.set_xlim(lims[0], lims[1])
                ax.set_ylim(lims[2], lims[3])


            # Custom x ticks (journal style): force 3–4 ticks, avoid edge-overlap.
            if lims is not None:
                if r in (0, 1):
                    # FT rows: narrow CI range; 3 ticks; 2 decimals; ticks not on edges
                    _set_x_ticks_equal(ax, lims[0], lims[1], n=3, inner_frac=0.12, pad_frac=0.10, force_decimals=2, integer=False)
                elif r == 2:
                    # Medium-swing row: 4 integer-like ticks
                    _set_x_ticks_equal(ax, lims[0], lims[1], n=3, inner_frac=0.08, pad_frac=0.06, integer=True, snap=5.0)
                elif r == 3:
                    # PtL row: 4 integer-like ticks
                    _set_x_ticks_equal(ax, lims[0], lims[1], n=3, inner_frac=0.08, pad_frac=0.06, integer=True, snap=10.0)

            # Hide y tick labels on non-first columns (reduce clutter)
            if c != 0:
                ax.set_yticklabels([])

            # X tick labels:
            # - Always show for bottom row (readability).
            # - Also show for first column of each upper row (so readers can quantify each row's CI scale).
            if r != 3 and c != 0:
                ax.set_xticklabels([])

    # Plot data
    for r, (_, paths) in enumerate(groups):
        ycol = row_specs[r]["ycol"]
        for c, reg in enumerate(regions):
            ax = axes[r][c]
            if not paths:
                ax.text(
                    0.5, 0.5, "Not available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#777777",
                )
                continue

            sub_rc = dd[
                dd["Region"].astype(str).eq(str(reg))
                & dd["pathway"].astype(str).isin(paths)
            ]
            if sub_rc.empty:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#777777",
                )
                continue

            for p in paths:
                sp = sub_rc[sub_rc["pathway"].astype(str).eq(p)]
                if sp.empty:
                    continue
                _plot_traj(ax, sp, color=path_color.get(p, "#333333"), ycol=ycol)

    # Global legend: pathways (colors) + marker meaning
    from matplotlib.lines import Line2D
    legend_elems: list[Line2D] = []
    legend_labels: list[str] = []

    # Only include pathways that appear in any group
    used_paths: list[str] = []
    for _, ps in groups:
        used_paths.extend(ps)
    used_paths = [p for p in pathways if p in set(used_paths)]

    for p in used_paths:
        legend_elems.append(Line2D([0], [0], color=path_color.get(p, "#333333"), lw=2.0))
        legend_labels.append(p)

    marker_elems = [
        Line2D([0], [0], marker="o", color="black", markersize=5,
               markerfacecolor="white", markeredgewidth=1.0, linestyle="None"),
        Line2D([0], [0], marker="o", color="black", markersize=5,
               markerfacecolor="black", markeredgewidth=0.8, linestyle="None"),
    ]
    marker_labels = [f"Start ({min(years)})", f"End ({max(years)})"]

    leg1 = fig.legend(
        legend_elems,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(6, max(1, len(legend_labels))),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.1,
    )
    fig.add_artist(leg1)
    fig.legend(
        marker_elems,
        marker_labels,
        loc="lower right",
        bbox_to_anchor=(0.99, -0.02),
        ncol=1,
        frameon=False,
        handletextpad=0.6,
    )

    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.98])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def fig33_ci_distribution_with_threshold(
    df_total: pd.DataFrame,
    df_feasible: pd.DataFrame | None,
    scenarios: list[str],
    regions: list[str],
    years: list[int],
    ci_col: str,
    policy_config_path: "Path|None",
    outpath: Path,
    scenario_markers: bool = False,
):
    """Fig 3.3 — LCFS-style CI threshold feasibility & distribution.

    2×2 facets by region. Points are colored by pathway.
      - Filled point: pass (CI ≤ threshold)
      - Hollow point: fail (CI > threshold)
    Threshold is a connected polyline across years.

    scenario_markers=True optionally uses marker shapes for scenarios (for a variant figure).
    """
    apply_rcparams()
    ensure_dir(outpath.parent)

    # ---- CI unit normalization (kgCO2e/MJ vs gCO2e/MJ) ----
    ci_vals = pd.to_numeric(df_total[ci_col], errors="coerce").dropna()
    ci_factor = 1.0 if (len(ci_vals) and float(ci_vals.median()) > 5.0) else 1000.0

    ci_std = load_lcfs_ci_standard_g_per_MJ(policy_config_path, years)
    path_color = make_path_color(df_total["pathway"].unique())

    scen_mark = {
        "Current Policies": "o",
        "Delayed transition": "s",
        "Fragmented World": "^",
        "Net Zero 2050": "D",
    }

    # Ordinal x positions (even spacing)
    pos = np.arange(len(years), dtype=float) + 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    # Shared y-lims
    df_sel = df_total[
        df_total["Region"].isin(regions)
        & df_total["Year"].isin(years)
        & df_total["Scenario"].isin(scenarios)
    ].copy()
    df_sel["_ci_g"] = pd.to_numeric(df_sel[ci_col], errors="coerce") * ci_factor
    y_min = float(np.nanmin(df_sel["_ci_g"].values)) if len(df_sel) else -10.0
    y_max = float(np.nanmax(df_sel["_ci_g"].values)) if len(df_sel) else 120.0
    y_rng = max(1.0, (y_max - y_min))
    y_min, y_max = y_min - 0.08 * y_rng, y_max + 0.12 * y_rng

    point_size = 20
    point_alpha = 0.88
    fail_lw = 0.95
    cloud_half_width = 0.28

    thr_color = "0.35"
    thr_lw = 2.0
    thr_dashes = (3.0, 2.2)

    for ax, region in zip(axes, regions):
        subR = df_total[
            (df_total["Region"] == region)
            & (df_total["Year"].isin(years))
            & (df_total["Scenario"].isin(scenarios))
        ].copy()

        ax.set_title(clean_region_name(region), loc="left", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.18)
        ax.set_ylim(y_min, y_max)

        if subR.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Scatter points year-by-year
        for i, y in enumerate(years):
            sY = subR[subR["Year"] == y].copy()
            if sY.empty:
                continue
            sY["_ci_g"] = pd.to_numeric(sY[ci_col], errors="coerce") * ci_factor
            sY = sY.dropna(subset=["_ci_g"])
            if sY.empty:
                continue

            sY = sY.sort_values(["Scenario", "pathway", "_ci_g"]).reset_index(drop=True)
            vals = sY["_ci_g"].to_numpy(dtype=float)
            thr = float(ci_std[int(y)])
            is_pass = vals <= thr

            seed_y = int(hashlib.md5(f"ci33|paper|{region}|{int(y)}".encode("utf-8")).hexdigest()[:8], 16)
            rng_y = np.random.RandomState(seed_y)
            x_off = rng_y.uniform(-cloud_half_width, cloud_half_width, size=len(vals)) if len(vals) > 1 else np.array([0.0])

            for j in range(len(sY)):
                r = sY.iloc[j]
                x = float(pos[i]) + float(x_off[j])
                yv = float(r["_ci_g"])
                pth = str(r["pathway"])
                col = path_color.get(pth, "0.4")
                mk = scen_mark.get(str(r["Scenario"]), "o") if scenario_markers else "o"

                if bool(is_pass[j]):
                    ax.scatter(x, yv, s=point_size, alpha=point_alpha, color=col, marker=mk,
                               edgecolor="none", linewidths=0.0, zorder=3)
                else:
                    ax.scatter(x, yv, s=point_size, alpha=point_alpha, facecolors="none",
                               edgecolors=col, marker=mk, linewidths=fail_lw, zorder=3)

        # Connected threshold polyline
        xs = pos
        ys = np.array([float(ci_std[int(y)]) for y in years], dtype=float)
        (ln,) = ax.plot(xs, ys, color=thr_color, lw=thr_lw, zorder=2)
        ln.set_dashes(thr_dashes)
        ln.set_solid_capstyle("butt")

        # Subtle shading below threshold
        ymin_ax, ymax_ax = ax.get_ylim()
        ax.fill_between(xs, ymin_ax, ys, color="0.93", alpha=0.25, zorder=0)
        ax.set_ylim(ymin_ax, ymax_ax)

        ax.set_xlim(0.5, float(len(years)) + 0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(y) for y in years])

    # Axis labels
    axes[2].set_xlabel("Year")
    axes[3].set_xlabel("Year")
    axes[0].set_ylabel("CI (gCO$_2$e/MJ)")
    axes[2].set_ylabel("CI (gCO$_2$e/MJ)")

    # Legends: pass/fail + (optional) scenarios + pathways
    path_order = sorted(path_color.keys())
    path_handles = [
        Line2D([0], [0], marker="o", linestyle="none",
               markerfacecolor=path_color[p], markeredgecolor="none",
               label=p, markersize=6)
        for p in path_order
    ]

    pass_handle = Line2D([0], [0], marker="o", linestyle="none",
                         color="0.15", markerfacecolor="0.15", label="Pass (≤ threshold)", markersize=6)
    fail_handle = Line2D([0], [0], marker="o", linestyle="none",
                         color="0.15", markerfacecolor="none", markeredgecolor="0.15",
                         label="Fail (> threshold)", markersize=6)

    scen_handles = []
    if scenario_markers:
        for sc in scenarios:
            scen_handles.append(
                Line2D([0], [0], marker=scen_mark.get(sc, "o"), linestyle="none",
                       markerfacecolor="none", markeredgecolor="0.25",
                       color="0.25", label=sc, markersize=6)
            )

    if scenario_markers:
        bottom = 0.30
        y_thr, y_scen, y_path = 0.165, 0.105, 0.035
    else:
        bottom = 0.22
        y_thr, y_path = 0.090, 0.030

    fig.legend(handles=[pass_handle, fail_handle], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, y_thr),
               handletextpad=0.6, columnspacing=1.6)

    if scenario_markers:
        fig.legend(handles=scen_handles, title="Scenario", loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, y_scen),
                   handletextpad=0.6, columnspacing=1.6)

    fig.legend(handles=path_handles, loc="lower center", ncol=min(4, len(path_handles)),
               frameon=False, bbox_to_anchor=(0.5, y_path),
               handletextpad=0.6, columnspacing=1.4)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=bottom, wspace=0.06, hspace=0.14)
    save_png_pdf(fig, outpath)
    plt.close(fig)


def find_policy_config(policy_outdir: Path) -> "Path|None":
    """Locate policy_config.json in common locations."""
    candidates = []
    try:
        candidates.append(Path(policy_outdir) / "policy_config.json")
        candidates.append(Path(policy_outdir).parent / "policy_config.json")
    except Exception:
        pass
    try:
        candidates.append(Path(__file__).resolve().parent / "policy_config.json")
    except Exception:
        pass
    candidates.append(Path.cwd() / "policy_config.json")
    # Walk up from policy_outdir (up to 6 levels) to find a repo-level config
    try:
        cur = Path(policy_outdir).resolve()
        for _ in range(6):
            candidates.append(cur / "policy_config.json")
            cur = cur.parent
    except Exception:
        pass
    for c in candidates:
        try:
            if c and c.exists():
                return c
        except Exception:
            continue
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-outdir", required=True, help="policy_layer output directory")
    ap.add_argument("--scenarios", nargs="+", required=True, help="Exactly 4 scenarios (space separated)")
    ap.add_argument("--regions", nargs="+", required=True, help="Exactly 4 regions (space separated)")
    ap.add_argument("--year-min", type=int, default=2025)
    ap.add_argument("--year-max", type=int, default=2050)

    # CLI compatibility with older pipeline (accepted but unused here)
    ap.add_argument("--pareto-year", type=int, default=2050)
    ap.add_argument("--space-year", type=int, default=2050)
    ap.add_argument("--heatmap-source", default="none", choices=["none", "merged", "filtered", "both"])

    ap.add_argument("--outdir", default=None, help="override output dir (default: <policy-outdir>/paper_figures)")
    ap.add_argument("--uco-name", default="HEFA_UCO", help="Pathway name used as UCO benchmark")
    ap.add_argument("--scenario-for-figI", default="Net Zero 2050", help="Fixed scenario used in Fig I")
    ap.add_argument("--scenario-for-fig32", default="Net Zero 2050", help="Fixed scenario used in Chapter 3.2 figs")
    ap.add_argument("--policy-config", default=None, help="Path to policy_config.json (default: <policy-outdir>/policy_config.json if exists)")
    ap.add_argument("--cap-figI", type=float, default=5000.0, help="Cap for Fig I y-axis values (±cap)")
    args = ap.parse_args()

    policy_outdir = Path(args.policy_outdir)
    merged_path = policy_outdir / "policy_merged_metrics.csv"
    filtered_path = policy_outdir / "policy_filtered_metrics.csv"
    if not merged_path.exists():
        raise SystemExit(f"Missing: {merged_path}")
    if not filtered_path.exists():
        raise SystemExit(f"Missing: {filtered_path}")

    merged = pd.read_csv(merged_path)
    filtered = pd.read_csv(filtered_path)

    # Basic schema checks
    for df_name, df in [("merged", merged), ("filtered", filtered)]:
        for c in ["Scenario", "Region", "Year", "pathway"]:
            if c not in df.columns:
                raise SystemExit(f"{df_name} missing column: {c}")

    # Pick columns (robust to variants)
    cost_col_filtered = pick_col(filtered, ["cost_USD2022_per_MJ", "cost_USD_per_MJ", "total_cost_usd_per_MJ", "Total", "cost"], "cost (filtered)")
    ci_col_filtered   = pick_col(filtered, ["CI_kgCO2_per_MJ", "gwp_allocated_kgCO2_per_MJ_SAF", "ci", "ghg_kgCO2_per_MJ"], "CI (filtered)")

    cost_col_merged = cost_col_filtered if cost_col_filtered in merged.columns else pick_col(
        merged, ["cost_USD2022_per_MJ", "cost_USD_per_MJ", "cost_plus_carbon_USD2022_per_MJ", "cost"], "cost (merged)"
    )
    ci_col_merged = ci_col_filtered if ci_col_filtered in merged.columns else pick_col(
        merged, ["CI_kgCO2_per_MJ", "gwp_allocated_kgCO2_per_MJ_SAF", "ci"], "CI (merged)"
    )

    scenarios = args.scenarios[:4]
    regions = args.regions[:4]

    # Master pathway order to keep colors consistent across all figures (e.g., Fig 3.3)
    pathway_order_master = list(pd.unique(merged["pathway"]))

    outdir = Path(args.outdir) if args.outdir else (policy_outdir / "paper_figures")
    ensure_dir(outdir)

    yrs_all_merged = available_years(merged, args.year_min, args.year_max)
    yrs_all_filt   = available_years(filtered, args.year_min, args.year_max)

    # Chapter years (sparse, journal-friendly)
    chapter_years = [2025, 2030, 2035, 2040, 2045, 2050]
    years_union = set(yrs_all_merged) | set(yrs_all_filt)
    chapter_years = [y for y in chapter_years if y in years_union]
    if not chapter_years:
        chapter_years = yrs_all_merged[:6] if len(yrs_all_merged) >= 6 else yrs_all_merged

    # 1) Line plots
    line_out_base = outdir / "lineplots_baseline"
    line_out_pol  = outdir / "lineplots_policy"

    lineplots_for_all_scenarios(
        df=merged, scenarios=scenarios, regions=regions, years=yrs_all_merged,
        outdir=line_out_base, cost_col=cost_col_merged, ci_col=ci_col_merged, tag="baseline"
    )
    print(f"✅ Baseline line plots saved to: {line_out_base.resolve()}")

    lineplots_for_all_scenarios(
        df=filtered, scenarios=scenarios, regions=regions, years=yrs_all_filt,
        outdir=line_out_pol, cost_col=cost_col_filtered, ci_col=ci_col_filtered, tag="policy"
    )
    print(f"✅ Policy-feasible line plots saved to: {line_out_pol.resolve()}")

    # 2) Chapter figs (Fig F & Fig I)
    ch_out = policy_outdir / "paper_figures_chapters" / "ch3_4_compliance_cost"
    ensure_dir(ch_out)
    # 3.2 Cost–CI trade-off (baseline; all scenarios) — scatter (color=pathway, marker=scenario)
    ch32_out = policy_outdir / "paper_figures_chapters" / "ch3_2_tradeoff"
    ensure_dir(ch32_out)
    fig32_cost_ci_grid_4x4(
        df=merged,
        regions=regions,
        years=chapter_years,
        cost_col=cost_col_merged,
        ci_col=ci_col_merged,
        outpath=ch32_out / "Fig_3_2_cost_ci_grid_4x4_baseline.png",
        pathways_order=pathway_order_master,
        scenario_fixed=args.scenario_for_fig32,
    )
    print(f"✅ Chapter 3.2 figure saved to: {ch32_out.resolve()}")


    # 3.3 LCFS-style CI threshold (policy-feasible)
    ch33_out = policy_outdir / "paper_figures_chapters" / "ch3_3_lcfs_feasibility"
    ensure_dir(ch33_out)
    policy_cfg = Path(args.policy_config) if args.policy_config else (policy_outdir / "policy_config.json")
    if not policy_cfg.exists():
        # fall back to user-provided filename if present in working dir
        alt = policy_outdir / "policy_config_used.json"
        if alt.exists():
            policy_cfg = alt
    if policy_cfg is None or not policy_cfg.exists():
        print("⚠️  policy_config.json not found; generating 3.3 using DEFAULT_LCFS_CI_STANDARD_POINTS_YEAR_KGCO2_PER_MJ embedded in make_paper_figures.py.")
        policy_cfg = None

    # Use baseline (policy_merged_metrics) for distribution + pass/fail split.
    # Optionally overlay policy-feasible points (policy_filtered_metrics) with black edges.
    ci_col_33 = ci_col_merged
    if ci_col_33 not in filtered.columns:
        # Fall back to a CI column present in both tables if needed
        if ci_col_filtered in merged.columns and ci_col_filtered in filtered.columns:
            ci_col_33 = ci_col_filtered

    fig33_ci_distribution_with_threshold(
        df_total=merged,
        df_feasible=filtered,
        scenarios=scenarios,
        regions=regions,
        years=chapter_years,
        ci_col=ci_col_33,
        policy_config_path=policy_cfg,
        outpath=ch33_out / "Fig_3_3_CI_distribution_with_threshold_policy.png",
        scenario_markers=False,
    )

    fig33_ci_distribution_with_threshold(
        df_total=merged,
        df_feasible=filtered,
        scenarios=scenarios,
        regions=regions,
        years=chapter_years,
        ci_col=ci_col_33,
        policy_config_path=policy_cfg,
        outpath=ch33_out / "Fig_3_3_CI_distribution_with_threshold_policy_scenmarkers.png",
        scenario_markers=True,
    )

    print(
        "✅ Chapter 3.3 figures saved to: "
        f"{(ch33_out / 'Fig_3_3_CI_distribution_with_threshold_policy.png').resolve()} and "
        f"{(ch33_out / 'Fig_3_3_CI_distribution_with_threshold_policy_scenmarkers.png').resolve()}"
    )

    # Net cost column (prefer LCFS net cost; else fall back to cost)
    net_cost_col = "net_cost_lcfs_USD2022_per_MJ"

    # ---------- baseline (merged) ----------
    merged_for_figs = merged.copy()
    if net_cost_col not in merged_for_figs.columns:
        merged_for_figs[net_cost_col] = merged_for_figs[cost_col_merged].astype(float)

    
    figI_breakevenP_allpaths_vs_uco(
        df=merged_for_figs,
        regions=regions,
        years=chapter_years,
        cost_col=cost_col_merged,
        ci_col=ci_col_merged,
        scenario_fixed=args.scenario_for_figI,
        uco_name=args.uco_name,
        cap=float(args.cap_figI),
        outpath=ch_out / "Fig_I_allpaths_breakevenP_vs_UCO_4regions_baseline.png",
        pathways_order=pathway_order_master,
    )
    print(f"✅ Fig I (baseline) saved to: {(ch_out / 'Fig_I_allpaths_breakevenP_vs_UCO_4regions_baseline.png').resolve()}")


    figI_breakevenP_allpaths_vs_winner(
        df=merged_for_figs,
        regions=regions,
        years=chapter_years,
        cost_col=cost_col_merged,
        ci_col=ci_col_merged,
        scenario_fixed=args.scenario_for_figI,
        cap=float(args.cap_figI),
        outpath=ch_out / "Fig_I_allpaths_breakevenP_vs_Winner_4regions_baseline.png",
        pathways_order=pathway_order_master,
    )
    print(f"✅ Fig I (baseline, winner parity) saved to: {(ch_out / 'Fig_I_allpaths_breakevenP_vs_Winner_4regions_baseline.png').resolve()}")

    # ---------- policy-feasible (filtered) ----------
    filtered_for_figs = filtered.copy()
    if net_cost_col not in filtered_for_figs.columns:
        filtered_for_figs[net_cost_col] = filtered_for_figs[cost_col_filtered].astype(float)

    figF_net_cost_boxplot_with_points(
        df=filtered_for_figs,
        scenarios=scenarios,
        regions=regions,
        years=chapter_years,
        net_cost_col=net_cost_col,
        outpath=ch_out / "Fig_F_allpaths_net_cost_boxplot_4regions_points_policy.png",
    )
    print(f"✅ Fig F (policy) saved to: {(ch_out / 'Fig_F_allpaths_net_cost_boxplot_4regions_points_policy.png').resolve()}")

    figF_net_cost_boxplot_with_points(
        df=filtered_for_figs,
        scenarios=scenarios,
        regions=regions,
        years=chapter_years,
        net_cost_col=net_cost_col,
        outpath=ch_out / "Fig_F_allpaths_net_cost_boxplot_4regions_points_policy_scenmarkers.png",
        scenario_markers=True,
    )
    print(f"✅ Fig F (policy, scen markers) saved to: {(ch_out / 'Fig_F_allpaths_net_cost_boxplot_4regions_points_policy_scenmarkers.png').resolve()}")


    figI_breakevenP_allpaths_vs_uco(
        df=filtered_for_figs,
        regions=regions,
        years=chapter_years,
        cost_col=cost_col_filtered,
        ci_col=ci_col_filtered,
        scenario_fixed=args.scenario_for_figI,
        uco_name=args.uco_name,
        cap=float(args.cap_figI),
        outpath=ch_out / "Fig_I_allpaths_breakevenP_vs_UCO_4regions_policy.png",
        pathways_order=pathway_order_master,
    )
    print(f"✅ Fig I (policy) saved to: {(ch_out / 'Fig_I_allpaths_breakevenP_vs_UCO_4regions_policy.png').resolve()}")


    figI_breakevenP_allpaths_vs_winner(
        df=filtered_for_figs,
        regions=regions,
        years=chapter_years,
        cost_col=cost_col_filtered,
        ci_col=ci_col_filtered,
        scenario_fixed=args.scenario_for_figI,
        cap=float(args.cap_figI),
        outpath=ch_out / "Fig_I_allpaths_breakevenP_vs_Winner_4regions_policy.png",
        pathways_order=pathway_order_master,
    )
    print(f"✅ Fig I (policy, winner parity) saved to: {(ch_out / 'Fig_I_allpaths_breakevenP_vs_Winner_4regions_policy.png').resolve()}")


if __name__ == "__main__":
    main()
