"""
Policy-layer analyzer for coupled LCA+TEA outputs (NGFS dynamic experiment).

Inputs:
  --drivers : CSV with Scenario/Region/Year drivers (carbon price, grid EF, fossil price proxy, H2 shares...)
  --ghg     : ALL_results_total_ghg.csv (must contain gwp_allocated_kgCO2_per_MJ_SAF or gwp_unallocated_...)
  --tea     : ALL_tea_total_cost_usd_per_MJ.csv (supports 'total_cost_usd_per_MJ' or 'Total')
  --config  : policy_config.json
  --outdir  : output folder

Outputs (policy_outdir):
  policy_merged_metrics.csv
  policy_filtered_metrics.csv

  # Baseline (no policy) multi-objective results
  pareto_frontiers_baseline.csv
  winners_baseline.csv
  reversal_points_winner_baseline.csv
  reversal_points_pareto_baseline.csv

  # Policy-feasible multi-objective results
  pareto_frontiers_policy.csv
  winners_policy.csv
  reversal_points_winner_policy.csv
  reversal_points_pareto_policy.csv

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


def _norm_colname(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("-", "_")


def find_first_col(df: pd.DataFrame, candidates: List[str], allow_contains: bool = False) -> Optional[str]:
    """
    Find the first column in df that matches any candidate.
    - Exact match by normalized name
    - Optionally substring contains match by normalized name
    """
    norm_map = {_norm_colname(c): c for c in df.columns}
    target_norms = [_norm_colname(x) for x in candidates]

    for tn in target_norms:
        if tn in norm_map:
            return norm_map[tn]

    if allow_contains:
        cols_norm = [(_norm_colname(c), c) for c in df.columns]
        for tn in target_norms:
            for cn, orig in cols_norm:
                if tn in cn:
                    return orig

    return None


def ensure_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {name}: {missing}. Available={list(df.columns)}")

def _determine_case_keys(ghg: pd.DataFrame, tea: pd.DataFrame) -> List[str]:
    """Return additional keys for SA/UA cases (e.g., case_id, param, level) if present in BOTH tables.
    We only include a key if it exists in both ghg and tea, so the merge stays 1-to-1.
    """
    extra = []
    for k in ["case_id", "param", "level"]:
        if (k in ghg.columns) and (k in tea.columns):
            extra.append(k)
    return extra


def _merge_keys(ghg: pd.DataFrame, tea: pd.DataFrame) -> List[str]:
    """Base merge keys + optional SA/UA keys."""
    keys = ["pathway", "Scenario", "Region", "Year"]
    keys += _determine_case_keys(ghg, tea)
    return keys


def _group_keys(df: pd.DataFrame) -> List[str]:
    """Group keys for Pareto/Winner computation.
    If SA/UA case keys exist, compute results per-case to avoid mixing cases.
    """
    keys = ["Scenario", "Region", "Year"]
    if "case_id" in df.columns:
        keys.append("case_id")
    if ("param" in df.columns) and ("case_id" in df.columns):
        
        pass
    return keys



def pareto_front_2d(df: pd.DataFrame, obj_x: str, obj_y: str) -> pd.DataFrame:
    """
    Minimize obj_x and obj_y. Returns non-dominated subset.
    O(n^2) per group (fine for typical experiment size).
    """
    vals = df[[obj_x, obj_y]].to_numpy(dtype=float)
    n = len(df)
    is_eff = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_eff[i]:
            continue
        dominates_i = (vals[:, 0] <= vals[i, 0]) & (vals[:, 1] <= vals[i, 1]) & (
            (vals[:, 0] < vals[i, 0]) | (vals[:, 1] < vals[i, 1])
        )
        dominates_i[i] = False
        if dominates_i.any():
            is_eff[i] = False

    return df.loc[is_eff].copy()


def normalize_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    lo, hi = np.nanmin(x.values), np.nanmax(x.values)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return (x - lo) / (hi - lo)
   
    return pd.Series(np.zeros(len(x)), index=x.index)


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))



def detect_tea_cost_col(tea: pd.DataFrame) -> str:
    """
    TEA file supported columns like:
      ['pathway','Total','Scenario','Region','Year']
    Also support 'total_cost_usd_per_MJ' etc.
    """
    direct = [
        "total_cost_usd_per_MJ",
        "total_cost_usd_per_mj",
        "total_cost",
        "Total",
        "TOTAL",
        "total",
    ]
    c = find_first_col(tea, direct, allow_contains=False)
    if c is not None and c in tea.columns:
        return c

    cols = list(tea.columns)
    cand = []
    for col in cols:
        cn = _norm_colname(col)
        if cn.startswith("total") and ("mj" in cn or "per_mj" in cn):
            cand.append(col)
    if cand:
        return cand[0]

    raise RuntimeError(f"Could not find total cost column in TEA file. cols={cols}")


def detect_carbon_price_col(drivers: pd.DataFrame) -> str:
    direct = [
        "carbon_price_USD2010_per_tCO2",
        "carbon_price_usd2010_per_tco2",
        "Price|Carbon",
        "Price|Carbon (TOTAL)",
        "price|carbon",
        "pricecarbon",
        "price_carbon",
    ]
    c = find_first_col(drivers, direct, allow_contains=True)
    if c is None:
        raise RuntimeError(f"Could not find a carbon price column in drivers. cols={list(drivers.columns)}")
    return c


def detect_grid_ef_col(drivers: pd.DataFrame) -> Optional[str]:
    direct = [
        "grid_EF_gCO2_per_kWh",
        "grid_ef_gco2_per_kwh",
        "electricity_EF_gCO2_per_kWh",
        "power_EF_gCO2_per_kWh",
        "ef_grid_gco2_per_kwh",
    ]
    return find_first_col(drivers, direct, allow_contains=True)


def detect_fossil_price_col(drivers: pd.DataFrame) -> Optional[str]:
    direct = [
        "fossil_liquids_price_USD2022_per_MJ",
        "fossil_liquids_price_usd2022_per_mj",
        "fossil_price_USD2022_per_MJ",
        "Price|Final Energy|Transportation|Liquids",
        "price|finalenergy|transportation|liquids",
    ]
    return find_first_col(drivers, direct, allow_contains=True)




def _winner_from_group(
    g: pd.DataFrame,
    cost_obj: str,
    ci_obj: str,
    alpha: float,
    normalize: bool,
    restrict_to_pareto: bool,
) -> pd.DataFrame:
    """
    Select ONE winner from a group (Scenario, Region, Year).
    - If restrict_to_pareto=True, candidate set = Pareto set under (cost_obj, ci_obj).
      This guarantees the winner is Pareto-efficient.
    - Otherwise candidate set = all rows in group.
    Scalarization:
      score = cost_n + alpha * ci_n  (if normalize)
      score = cost + alpha * ci      (if not normalize)
    Tie-break (deterministic):
      1) lower score
      2) lower ci
      3) lower cost
      4) pathway name
    """
    if g.empty:
        return g.head(0)

    cand = pareto_front_2d(g, cost_obj, ci_obj) if restrict_to_pareto else g.copy()
    gg = cand.copy()

    if normalize:
        gg["cost_n"] = normalize_series(gg[cost_obj])
        gg["ci_n"] = normalize_series(gg[ci_obj])
        gg["score"] = gg["cost_n"] + alpha * gg["ci_n"]
    else:
        gg["score"] = safe_float_series(gg[cost_obj]) + alpha * safe_float_series(gg[ci_obj])

    # deterministic tie-break
    gg["_pathway_str"] = gg["pathway"].astype(str)
    w = (
        gg.sort_values(
            ["score", ci_obj, cost_obj, "_pathway_str"],
            ascending=[True, True, True, True],
            kind="mergesort",  # stable
        )
        .head(1)
        .drop(columns=["_pathway_str"])
        .copy()
    )
    return w


def compute_multiobjective(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    label: str,
) -> Dict[str, pd.DataFrame]:
    """
    Compute Pareto fronts, winners, and reversal point tables on a given dataframe.
    df must have Scenario/Region/Year/pathway and objective columns.
    """
    pareto_cfg = cfg.get("pareto", {})
    rev_cfg = cfg.get("reversal", {})
    scalar = rev_cfg.get("scalarization", {})

  
    obj_cfg = cfg.get("objectives", {}) if isinstance(cfg, dict) else {}

    # Default Pareto objective cost can include carbon cost if user wants.
    pareto_use_carbon = bool(pareto_cfg.get("include_carbon_cost_in_cost_USD2022_per_MJ", True))
    pareto_cost_obj = "cost_plus_carbon_USD2022_per_MJ" if pareto_use_carbon else "cost_USD2022_per_MJ"

    # Default CI objective
    ci_obj = "CI_kgCO2_per_MJ"

    # Override if user provides explicit objective columns
    if obj_cfg.get("pareto_cost_obj", None) is not None:
        pareto_cost_obj = str(obj_cfg["pareto_cost_obj"])
    if obj_cfg.get("ci_obj", None) is not None:
        ci_obj = str(obj_cfg["ci_obj"])


  
    winner_use_carbon = scalar.get("include_carbon_cost_in_cost_USD2022_per_MJ", None)
    if winner_use_carbon is None:
        # default: trade-off uses production cost
        winner_cost_obj = "cost_USD2022_per_MJ"
    else:
        winner_cost_obj = "cost_plus_carbon_USD2022_per_MJ" if bool(winner_use_carbon) else "cost_USD2022_per_MJ"
    # Explicit override
    if obj_cfg.get("winner_cost_obj", None) is not None:
        winner_cost_obj = str(obj_cfg["winner_cost_obj"])

    alpha = float(scalar.get("alpha", 0.2))

    # accept both keys (older/newer configs)
    normalize = scalar.get("normalize_within_each_group", None)
    if normalize is None:
        normalize = scalar.get("normalize_within_each_region_scenario_year", True)
    normalize = bool(normalize)

    restrict_to_pareto = bool(scalar.get("restrict_to_pareto_set", True))

    # ---- Pareto ----
    pareto_rows = []
    for keys, g in df.groupby(_group_keys(df), dropna=False):
        # unpack keys robustly
        if isinstance(keys, tuple):
            sc, rg, yr = keys[0], keys[1], keys[2]
            case_id = keys[3] if len(keys) > 3 else None
        else:
            sc, rg, yr = keys, None, None
            case_id = None
        if g.empty:
            continue
        pf = pareto_front_2d(g, pareto_cost_obj, ci_obj)
        pf = pf.assign(
            cost_objective=pareto_cost_obj,
            ci_objective=ci_obj,
            set_label=label,
        )
        if case_id is not None:
            pf["case_id"] = case_id
        pareto_rows.append(pf)
    pareto_df = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else df.head(0)

    # ---- Winner ----
    winners = []
    for keys, g in df.groupby(_group_keys(df), dropna=False):
        # unpack keys robustly
        if isinstance(keys, tuple):
            sc, rg, yr = keys[0], keys[1], keys[2]
            case_id = keys[3] if len(keys) > 3 else None
        else:
            sc, rg, yr = keys, None, None
            case_id = None
        if g.empty:
            continue
        w = _winner_from_group(
            g=g,
            cost_obj=winner_cost_obj,
            ci_obj=ci_obj,
            alpha=alpha,
            normalize=normalize,
            restrict_to_pareto=restrict_to_pareto,
        )
        if not w.empty:
            w = w.assign(
                alpha=alpha,
                normalize_within_each_group=normalize,
                restrict_to_pareto_set=restrict_to_pareto,
                cost_objective=winner_cost_obj,
                ci_objective=ci_obj,
                set_label=label,
            )
            if case_id is not None:
                w["case_id"] = case_id
            winners.append(w)
    winners_df = pd.concat(winners, ignore_index=True) if winners else df.head(0)


    win_changes = []
    rev_keys = ["Scenario", "Region"] + (["case_id"] if "case_id" in winners_df.columns else [])
    for k, g in winners_df.sort_values("Year").groupby(rev_keys, dropna=False):
        if isinstance(k, tuple):
            sc, rg = k[0], k[1]
            case_id = k[2] if len(k) > 2 else None
        else:
            sc, rg = k, None
            case_id = None
        prev = None
        for _, row in g.iterrows():
            if prev is not None and str(row["pathway"]) != str(prev["pathway"]):
                d = dict(
                        Scenario=sc,
                        Region=rg,
                        from_year=int(prev["Year"]),
                        to_year=int(row["Year"]),
                        from_pathway=str(prev["pathway"]),
                        to_pathway=str(row["pathway"]),
                        reason="winner_change_under_scalarization",
                        set_label=label,
                    )
                if case_id is not None:
                    d["case_id"] = case_id
                win_changes.append(d)
            prev = row
    win_changes_df = pd.DataFrame(win_changes)

   
    pareto_changes_df = pd.DataFrame()
    if bool(rev_cfg.get("also_report_pareto_set_changes", True)):
        changes = []
        chg_keys = ["Scenario", "Region"] + (["case_id"] if "case_id" in pareto_df.columns else [])
        for k, g in pareto_df.groupby(chg_keys, dropna=False):
            if isinstance(k, tuple):
                sc, rg = k[0], k[1]
                case_id = k[2] if len(k) > 2 else None
            else:
                sc, rg = k, None
                case_id = None
            years = sorted([int(x) for x in pd.unique(g["Year"].dropna())])
            prev_set = None
            prev_year = None
            for yr in years:
                s = set(g[g["Year"] == yr]["pathway"].astype(str).tolist())
                if prev_set is not None and s != prev_set:
                    d = dict(
                            Scenario=sc,
                            Region=rg,
                            from_year=int(prev_year),
                            to_year=int(yr),
                            from_pareto_set="|".join(sorted(prev_set)),
                            to_pareto_set="|".join(sorted(s)),
                            reason="pareto_set_change",
                            set_label=label,
                        )
                    if case_id is not None:
                        d["case_id"] = case_id
                    changes.append(d)
                prev_set, prev_year = s, yr
        pareto_changes_df = pd.DataFrame(changes)

    return {
        "pareto": pareto_df,
        "winners": winners_df,
        "reversal_winner": win_changes_df,
        "reversal_pareto": pareto_changes_df,
        "pareto_cost_obj": pareto_cost_obj,
        "winner_cost_obj": winner_cost_obj,
        "alpha": alpha,
        "normalize": normalize,
        "restrict_to_pareto": restrict_to_pareto,
    }



def piecewise_linear(x: float, points: List[List[float]], hold_after_last: bool = True) -> float:
    """Piecewise-linear interpolation.
    points: list of [year, value] sorted by year.
    If x is outside range: clamp to first (below) or last (above, if hold_after_last) or extrapolate (if hold_after_last=False).
    """
    if not points:
        return float("nan")
    pts = sorted([(float(a), float(b)) for a, b in points], key=lambda t: t[0])
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        if hold_after_last or len(pts) < 2:
            return pts[-1][1]
        # linear extrapolation using last segment
        x0, y0 = pts[-2]
        x1, y1 = pts[-1]
        return y1 + (x - x1) * (y1 - y0) / (x1 - x0)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            w = (x - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return pts[-1][1]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drivers", required=True, help="CSV with Scenario/Region/Year drivers")
    ap.add_argument("--ghg", required=True, help="ALL_results_total_ghg.csv")
    ap.add_argument("--tea", required=True, help="ALL_tea_total_cost_usd_per_MJ.csv")
    ap.add_argument("--config", required=True, help="policy_config.json")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    drivers = pd.read_csv(args.drivers)
    ghg = pd.read_csv(args.ghg)
    tea = pd.read_csv(args.tea)
    cfg = read_json(args.config)

    # ---- choose allocation column for CI ----
    alloc_choice = str(cfg.get("use_allocation", "allocated")).strip().lower()
    ci_col = "gwp_unallocated_kgCO2_per_MJ_SAF" if alloc_choice == "unallocated" else "gwp_allocated_kgCO2_per_MJ_SAF"
    if ci_col not in ghg.columns:
        raise RuntimeError(f"Expected '{ci_col}' in GHG results. cols={list(ghg.columns)}")

    # ---- detect TEA cost col ----
    cost_col = detect_tea_cost_col(tea)

    # ---- keys sanity ----
    ensure_cols(ghg, ["pathway", "Scenario", "Region", "Year", ci_col], "GHG")
    ensure_cols(tea, ["pathway", "Scenario", "Region", "Year", cost_col], "TEA")
    ensure_cols(drivers, ["Scenario", "Region", "Year"], "drivers")

    # normalize types
    for d in (drivers, ghg, tea):
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype("Int64")

    # ---- detect driver cols ----
    carbon_col = detect_carbon_price_col(drivers)
    grid_ef_col = detect_grid_ef_col(drivers)
    fossil_price_col = detect_fossil_price_col(drivers)

    # ---- merge ----
    base = (
        ghg[_merge_keys(ghg, tea) + [ci_col]]
        .merge(
            tea[_merge_keys(ghg, tea) + [cost_col]].rename(columns={cost_col: "cost_USD2022_per_MJ"}),
            on=_merge_keys(ghg, tea),
            how="left",
        )
        .merge(
            drivers,
            on=["Scenario", "Region", "Year"],
            how="left",
        )
    )

    # ---- currency conversion for carbon price (USD2010 -> USD2022) ----
    usd_factor = cfg.get("currency", {}).get("USD2010_to_USD2022_factor_CPIU", None)
    if usd_factor is None:
        p = Path(args.drivers).resolve().parent / "default_h2_parameters.json"
        if p.exists():
            j = read_json(p)
            usd_factor = j.get("parameters", {}).get("USD2010_to_USD2022_factor_CPIU", None)
    if usd_factor is None:
        raise RuntimeError(
            "Missing USD2010_to_USD2022_factor_CPIU. "
            "Set it in policy_config.json under currency.USD2010_to_USD2022_factor_CPIU "
            "or provide default_h2_parameters.json next to drivers."
        )
    usd_factor = float(usd_factor)

    base["CI_kgCO2_per_MJ"] = safe_float_series(base[ci_col])
    base["cost_USD2022_per_MJ"] = safe_float_series(base["cost_USD2022_per_MJ"])

    base["carbon_price_USD2010_per_tCO2"] = safe_float_series(base[carbon_col])
    base["carbon_price_USD2022_per_tCO2"] = base["carbon_price_USD2010_per_tCO2"] * usd_factor

    base["carbon_cost_USD2022_per_MJ"] = (base["CI_kgCO2_per_MJ"] / 1000.0) * base["carbon_price_USD2022_per_tCO2"]
    base["cost_plus_carbon_USD2022_per_MJ"] = base["cost_USD2022_per_MJ"] + base["carbon_cost_USD2022_per_MJ"]

    # optional: fossil liquids proxy price
    if fossil_price_col is not None:
        base["fossil_liquids_price_USD2022_per_MJ"] = safe_float_series(base[fossil_price_col])
        base["cost_diff_vs_fossil_USD2022_per_MJ"] = base["cost_plus_carbon_USD2022_per_MJ"] - base["fossil_liquids_price_USD2022_per_MJ"]
        base["cost_ratio_vs_fossil"] = base["cost_plus_carbon_USD2022_per_MJ"] / base["fossil_liquids_price_USD2022_per_MJ"]
    else:
        base["fossil_liquids_price_USD2022_per_MJ"] = np.nan
        base["cost_diff_vs_fossil_USD2022_per_MJ"] = np.nan
        base["cost_ratio_vs_fossil"] = np.nan

    # optional: grid EF
    if grid_ef_col is not None:
        base["grid_EF_gCO2_per_kWh"] = safe_float_series(base[grid_ef_col])
    else:
        base["grid_EF_gCO2_per_kWh"] = np.nan

    # ---- LCFS-style CI standard & credit/deficit pricing (optional) ----
    # Computes CI_standard and compliance-adjusted cost for downstream analysis.
    lcfs = cfg.get("lcfs_style", {})
    if bool(lcfs.get("enabled", False)):
        pts = lcfs.get("ci_standard_points_year_kgCO2_per_MJ", None)
        if not pts:
            raise RuntimeError("lcfs_style.enabled=true but ci_standard_points_year_kgCO2_per_MJ is missing/empty")
        hold = bool(lcfs.get("hold_after_last_year", True))
        base["CI_standard_kgCO2_per_MJ"] = base["Year"].astype(float).apply(
            lambda y: piecewise_linear(float(y), pts, hold_after_last=hold)
        )
        base["CI_gap_kgCO2_per_MJ"] = base["CI_kgCO2_per_MJ"] - base["CI_standard_kgCO2_per_MJ"]

        P = lcfs.get("credit_price_USD2022_per_tCO2", None)
        if P is None:
            raise RuntimeError("lcfs_style.enabled=true but credit_price_USD2022_per_tCO2 is null")
        P = float(P)

        # CI_gap < 0 => credit; CI_gap > 0 => deficit
        base["lcfs_credit_value_USD2022_per_MJ"] = (-base["CI_gap_kgCO2_per_MJ"] / 1000.0) * P
        base["lcfs_deficit_cost_USD2022_per_MJ"] = (base["CI_gap_kgCO2_per_MJ"].clip(lower=0) / 1000.0) * P

        # Net compliance-adjusted cost (can be negative if credits exceed production cost)
        base["net_cost_lcfs_USD2022_per_MJ"] = base["cost_USD2022_per_MJ"] + (base["CI_gap_kgCO2_per_MJ"] / 1000.0) * P
    else:
        base["CI_standard_kgCO2_per_MJ"] = np.nan
        base["CI_gap_kgCO2_per_MJ"] = np.nan
        base["lcfs_credit_value_USD2022_per_MJ"] = np.nan
        base["lcfs_deficit_cost_USD2022_per_MJ"] = np.nan
        base["net_cost_lcfs_USD2022_per_MJ"] = np.nan

    # ---- constraints mask (policy feasibility) ----
    cons = cfg.get("constraints", {})
    mask = pd.Series(True, index=base.index)

    # Renewable proxy by grid EF
    rp = cons.get("renewable_proxy_by_grid_EF", {})
    if bool(rp.get("enabled", False)):
        thr = rp.get("max_grid_EF_gCO2_per_kWh", None)
        if thr is None:
            raise RuntimeError("renewable_proxy_by_grid_EF.enabled=true but max_grid_EF_gCO2_per_kWh is null")
        mask &= base["grid_EF_gCO2_per_kWh"] <= float(thr)

    # Hydrogen structure constraint (if driver shares exist)
    hs = cons.get("hydrogen_structure", {})
    if bool(hs.get("enabled", False)):
        use_renorm = bool(hs.get("use_renormalized_shares", True))
        green_col = "H2_share_electricity_renorm" if use_renorm else "H2_share_electricity"
        grey_col = "H2_share_fossil_wocc_renorm" if use_renorm else "H2_share_fossil_wocc"

        if green_col not in base.columns or grey_col not in base.columns:
            raise RuntimeError(
                "hydrogen_structure.enabled=true but share columns missing. "
                f"Need '{green_col}' and '{grey_col}'. Available={list(base.columns)}"
            )

        if hs.get("min_green_h2_share", None) is not None:
            mask &= safe_float_series(base[green_col]) >= float(hs["min_green_h2_share"])
        if hs.get("max_grey_h2_share", None) is not None:
            mask &= safe_float_series(base[grey_col]) <= float(hs["max_grey_h2_share"])

    # Static CI cap (single number)
    if cons.get("max_SAF_CI_kgCO2_per_MJ", None) is not None:
        mask &= base["CI_kgCO2_per_MJ"] <= float(cons["max_SAF_CI_kgCO2_per_MJ"])

    # Dynamic LCFS-style CI cap: enforce CI <= CI_standard_kgCO2_per_MJ
    if bool(cons.get("enforce_lcfs_ci_standard", False)):
        if base["CI_standard_kgCO2_per_MJ"].isna().all():
            raise RuntimeError(
                "constraints.enforce_lcfs_ci_standard=true but lcfs_style.enabled is false "
                "or CI_standard_kgCO2_per_MJ is missing"
            )
        mask &= base["CI_kgCO2_per_MJ"] <= base["CI_standard_kgCO2_per_MJ"]

    # Minimum lifecycle reduction vs fossil baseline (CORSIA-style gate)
    if cons.get("min_emissions_reduction_vs_fossil", None) is not None:
        fossil_ci = cons.get("fossil_baseline", {}).get("CI_kgCO2_per_MJ", None)
        if fossil_ci is None:
            raise RuntimeError("min_emissions_reduction_vs_fossil set but fossil_baseline.CI_kgCO2_per_MJ is null")
        fossil_ci = float(fossil_ci)
        red = 1.0 - (base["CI_kgCO2_per_MJ"] / fossil_ci)
        mask &= red >= float(cons["min_emissions_reduction_vs_fossil"])

    # Optional cost advantage vs fossil proxy
    ca = cons.get("cost_advantage_vs_fossil", {})
    if bool(ca.get("enabled", False)):
        if base["fossil_liquids_price_USD2022_per_MJ"].isna().all():
            raise RuntimeError(
                "cost_advantage_vs_fossil.enabled=true but fossil price proxy is missing in drivers "
                "(fossil_liquids_price_USD2022_per_MJ / Price|Final Energy|Transportation|Liquids)."
            )
        if ca.get("max_cost_ratio", None) is not None:
            mask &= base["cost_ratio_vs_fossil"] <= float(ca["max_cost_ratio"])
        if ca.get("max_cost_diff_USD2022_per_MJ", None) is not None:
            mask &= base["cost_diff_vs_fossil_USD2022_per_MJ"] <= float(ca["max_cost_diff_USD2022_per_MJ"])

    filtered = base.loc[mask].copy()

    # ---- multi-objective (baseline + policy) ----
    mo_baseline = compute_multiobjective(base, cfg, label="baseline")
    mo_policy = compute_multiobjective(filtered, cfg, label="policy")

    # ---- write outputs ----
    base.to_csv(outdir / "policy_merged_metrics.csv", index=False)
    filtered.to_csv(outdir / "policy_filtered_metrics.csv", index=False)

    mo_baseline["pareto"].to_csv(outdir / "pareto_frontiers_baseline.csv", index=False)
    mo_baseline["winners"].to_csv(outdir / "winners_baseline.csv", index=False)
    mo_baseline["reversal_winner"].to_csv(outdir / "reversal_points_winner_baseline.csv", index=False)
    mo_baseline["reversal_pareto"].to_csv(outdir / "reversal_points_pareto_baseline.csv", index=False)

    mo_policy["pareto"].to_csv(outdir / "pareto_frontiers_policy.csv", index=False)
    mo_policy["winners"].to_csv(outdir / "winners_policy.csv", index=False)
    mo_policy["reversal_winner"].to_csv(outdir / "reversal_points_winner_policy.csv", index=False)
    mo_policy["reversal_pareto"].to_csv(outdir / "reversal_points_pareto_policy.csv", index=False)

    # keep legacy filenames for backward compatibility (policy results)
    mo_policy["pareto"].to_csv(outdir / "pareto_frontiers.csv", index=False)
    mo_policy["winners"].to_csv(outdir / "winners.csv", index=False)
    mo_policy["reversal_winner"].to_csv(outdir / "reversal_points_winner.csv", index=False)
    mo_policy["reversal_pareto"].to_csv(outdir / "reversal_points_pareto.csv", index=False)

    print(f"[policy_layer] wrote outputs to: {outdir}")
    print("[policy_layer] multi-objective settings:")
    print(f"  baseline Pareto cost objective: {mo_baseline['pareto_cost_obj']}")
    print(f"  baseline Winner cost objective: {mo_baseline['winner_cost_obj']} (alpha={mo_baseline['alpha']}, normalize={mo_baseline['normalize']}, restrict_to_pareto={mo_baseline['restrict_to_pareto']})")
    print(f"  policy    Pareto cost objective: {mo_policy['pareto_cost_obj']}")
    print(f"  policy    Winner cost objective: {mo_policy['winner_cost_obj']} (alpha={mo_policy['alpha']}, normalize={mo_policy['normalize']}, restrict_to_pareto={mo_policy['restrict_to_pareto']})")


if __name__ == "__main__":
    main()
