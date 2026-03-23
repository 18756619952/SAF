"""
TEA model aligned with the manuscript's published accounting boundary.

This script:
1. Loads pathway foreground inventories and manuscript input tables.
2. Prices feedstock, selected energy inputs, co-product credits, transport,
   and fixed-cost components consistent with the manuscript TEA boundary.
3. Writes pathway-level cost outputs and supporting debug tables to a results directory.

Outputs:
  <run_dir>/
    - tea_cost_table_usd_per_MJ.csv
    - tea_total_cost_usd_per_MJ.csv
    - tea_parameter_costs_usd_per_MJ.csv
    - tea_cost_table_usd_per_L.csv        (optional; if MJ/L is provided)
    - tea_debug_inputs_per_MJ.csv
    - tea_debug_unmapped_flows.csv

"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


MEAL_LHV_MJ_PER_KG_DEFAULT = 15.5
MJ_PER_L_SAF_DEFAULT = None

FOREGROUND_FILES: Dict[str, str] = {
    "HEFA_Soybean": "foreground_HEFA_Soybean.csv",
    "HEFA_UCO": "foreground_HEFA_UCO.csv",
    "ATJ_FR": "foreground_ATJ_FR.csv",
    "ATJ_Switchgrass": "foreground_ATJ_Switchgrass.csv",
    "FT_Miscanthus": "foreground_FT_Miscanthus.csv",
    "FT_Switchgrass": "foreground_FT_Switchgrass.csv",
    "PtL_DAC": "foreground_PtL_DAC.csv",
}

PROCESS_MAP = {
    "HEFA_Soybean": {"saf": ["HEFA"], "ethanol": []},
    "HEFA_UCO": {"saf": ["HEFA"], "ethanol": []},
    "FT_Miscanthus": {"saf": ["FT"], "ethanol": []},
    "FT_Switchgrass": {"saf": ["FT"], "ethanol": []},
    "ATJ_FR": {"saf": ["EtOH_to_SAF"], "ethanol": ["Bioethanol"]},
    "ATJ_Switchgrass": {"saf": ["EtOH_to_SAF"], "ethanol": ["Bioethanol"]},
    "PtL_DAC": {"saf": ["PtL"], "ethanol": []},
}

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
TEA_DIR_CANDIDATES = [
    INPUT_DIR / "tea",
    INPUT_DIR / "tea" / "tea_outputs",
]

CONV_SUMMARY = INPUT_DIR / "conversion_matrix_summary.csv"
CONV_RAW = INPUT_DIR / "conversion_matrix_raw.csv"
TRANSPORT_LCI = INPUT_DIR / "transport_LCI_per_MJ.csv"

PRICE_BG = "background_price_USD2022.csv"
PRICE_RAW = "tea_price_table_mean.csv"
ECON = "tea_economic_parameters.csv"
TCI = "tea_TCI_table_mean.csv"
BTS_2022 = "bts_freight_rate_2022_converted.csv"
BTS_ANY = "bts_freight_rate_converted.csv"


def find_tea_input_dir() -> Path:
    for candidate in TEA_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot find TEA input directory. Tried: {TEA_DIR_CANDIDATES}")


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def crf(discount_rate: float, n_years: int) -> float:
    if discount_rate <= 0:
        return 1.0 / n_years
    return discount_rate * (1 + discount_rate) ** n_years / ((1 + discount_rate) ** n_years - 1)


def unit_to_kg_if_g(unit: str, amount: float) -> Tuple[str, float]:
    if str(unit).strip() == "g":
        return "kg", amount / 1000.0
    return str(unit).strip(), amount


def load_price_lookup(tea_dir: Path) -> Dict[str, Dict[str, object]]:
    preferred = tea_dir / PRICE_BG
    fallback = tea_dir / PRICE_RAW

    if preferred.exists():
        df = pd.read_csv(preferred)
        out: Dict[str, Dict[str, object]] = {}
        for _, row in df.iterrows():
            out[str(row["product"]).strip()] = {
                "price": float(row["price_usd_2022"]),
                "unit": str(row["unit"]).strip(),
                "physical_basis": str(row.get("physical_basis", "")).strip(),
            }
        return out

    require(fallback, "A.23 price table")
    df = pd.read_csv(fallback)
    out = {}
    for _, row in df.iterrows():
        item = str(row["item"]).strip()
        out[item] = {"price": float(row["mean"]), "unit": str(row["unit"]).strip(), "physical_basis": ""}
        cleaned = re.sub(r"\s*\([^)]*(19|20)\d{2}[^)]*\)\s*$", "", item).strip()
        if cleaned and cleaned not in out:
            out[cleaned] = out[item]
    return out


def load_econ_params(tea_dir: Path) -> Dict[str, float]:
    path = tea_dir / ECON
    require(path, "A.22 economic parameters")
    df = pd.read_csv(path)
    params = {str(row["parameter"]).strip(): float(row["value"]) for _, row in df.iterrows()}
    required = [
        "plant_scale_MT_feedstock_per_day",
        "capacity_factor",
        "opex_pct_of_TCI_per_year",
        "labor_pct_of_TCI_per_year",
        "discount_rate",
        "plant_life_years",
    ]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"A.22 parameter table is missing: {missing}")
    return params


def load_tci_lookup(tea_dir: Path) -> Dict[str, float]:
    path = tea_dir / TCI
    require(path, "A.24 TCI table")
    df = pd.read_csv(path)
    cols = {col.lower(): col for col in df.columns}
    process_col = cols.get("process") or df.columns[0]
    tci_col = cols.get("tci_mean_musd")
    if tci_col is None:
        for col in df.columns:
            if "tci" in col.lower() and "musd" in col.lower():
                tci_col = col
                break
    if tci_col is None:
        raise ValueError(f"Cannot find a TCI_mean_MUSD column in {path}")
    return {str(row[process_col]).strip(): float(row[tci_col]) for _, row in df.iterrows()}


def load_bts_rates(tea_dir: Path) -> Dict[str, float]:
    path_2022 = tea_dir / BTS_2022
    path_any = tea_dir / BTS_ANY
    if path_2022.exists():
        df = pd.read_csv(path_2022)
    else:
        require(path_any, "BTS freight rate table")
        df = pd.read_csv(path_any)

    cols = {col.lower(): col for col in df.columns}
    usd_col = cols.get("usd_per_metric_t_km")
    if usd_col is None:
        for col in df.columns:
            if "usd" in col.lower() and "km" in col.lower():
                usd_col = col
                break
    if usd_col is None:
        raise ValueError(f"Cannot locate a USD per t-km column in {df.columns.tolist()}")

    mode_col = cols.get("mode") or df.columns[0]
    rates: Dict[str, float] = {}
    for _, row in df.iterrows():
        mode_raw = str(row[mode_col]).strip()
        usd = float(row[usd_col])

        if mode_raw.lower().startswith("heavy heavy-duty truck") or mode_raw.lower().startswith("truck"):
            rates["Heavy Heavy-Duty Truck"] = usd
        elif mode_raw.lower().startswith("rail"):
            rates["Rail"] = usd
        elif mode_raw.lower().startswith("barge") or "water" in mode_raw.lower():
            rates["Barge"] = usd
    return rates


def intermediate_factors_from_conv_raw(conv_raw: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Extract intermediate-to-SAF factors used for basis conversion."""
    sub = conv_raw[conv_raw["bridge_type"].astype(str).str.contains("intermediate_per_MJ_SAF", na=False)]
    out: Dict[str, Dict[str, float]] = {}
    for _, row in sub.iterrows():
        pathway = row["pathway"]
        flow = str(row["from_flow"]).strip()
        out.setdefault(pathway, {})[flow] = float(row["amount"])
    return out


def factor_to_MJSAF(pathway: str, basis: str, conv_sum: pd.DataFrame, inter: Dict[str, Dict[str, float]]) -> float:
    """Convert a foreground inventory basis to 1 MJ SAF."""
    basis_clean = str(basis).strip()
    if basis_clean == "per_MJ_SAF":
        return 1.0
    if basis_clean in {"per_MJ_oil", "per_MJ_Oil"}:
        factor = inter.get(pathway, {}).get("Bio-oil (feedstock)")
        if factor is None:
            raise KeyError(f"Missing Bio-oil factor for {pathway} in conversion_matrix_raw.csv")
        return factor
    if basis_clean in {"per_MJ_EtOH", "per_MJ_ethanol", "per_MJ_Ethanol"}:
        factor = inter.get(pathway, {}).get("EtOH")
        if factor is None:
            raise KeyError(f"Missing ethanol factor for {pathway} in conversion_matrix_raw.csv")
        return factor
    if basis_clean in {"per_dry_kg_feedstock", "per_kg_feedstock", "per_kg_CO2_captured"}:
        row = conv_sum.loc[conv_sum["pathway"] == pathway]
        if row.empty:
            raise KeyError(f"Missing pathway '{pathway}' in conversion_matrix_summary.csv")
        return float(row["kg_feedstock_per_MJ_SAF"].iloc[0])
    raise KeyError(f"Unknown basis '{basis_clean}' for pathway '{pathway}'")


def price_get(price_lookup: Dict[str, Dict[str, object]], key: str) -> Tuple[float | None, str | None]:
    entry = price_lookup.get(key)
    if entry is None:
        return None, None
    return float(entry["price"]), str(entry["unit"]).strip()


def run_tea_model(
    out_dir: str | Path,
    meal_lhv_mj_per_kg: float = MEAL_LHV_MJ_PER_KG_DEFAULT,
    mj_per_l_saf: float | None = MJ_PER_L_SAF_DEFAULT,
) -> None:
    """Run the TEA workflow and write outputs to the provided results directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tea_dir = find_tea_input_dir()

    require(CONV_SUMMARY, "conversion_matrix_summary.csv")
    require(CONV_RAW, "conversion_matrix_raw.csv")
    require(TRANSPORT_LCI, "transport_LCI_per_MJ.csv")
    for pathway, filename in FOREGROUND_FILES.items():
        require(INPUT_DIR / filename, f"foreground file for {pathway}")

    price = load_price_lookup(tea_dir)
    econ = load_econ_params(tea_dir)
    tci = load_tci_lookup(tea_dir)
    bts = load_bts_rates(tea_dir)

    conv_sum = pd.read_csv(CONV_SUMMARY)
    conv_raw = pd.read_csv(CONV_RAW)
    inter = intermediate_factors_from_conv_raw(conv_raw)
    transport_lci = pd.read_csv(TRANSPORT_LCI)

    plant_scale_mt_day = econ["plant_scale_MT_feedstock_per_day"]
    capacity_factor = econ["capacity_factor"]
    opex_pct = econ["opex_pct_of_TCI_per_year"]
    labor_pct = econ["labor_pct_of_TCI_per_year"]
    discount_rate = econ["discount_rate"]
    plant_life_years = int(econ["plant_life_years"])
    capital_recovery_factor = crf(discount_rate, plant_life_years)

    rows = []
    debug_inputs = []
    debug_ignored = []

    for pathway, fg_filename in FOREGROUND_FILES.items():
        foreground = pd.read_csv(INPUT_DIR / fg_filename)

        row = conv_sum.loc[conv_sum["pathway"] == pathway]
        if row.empty:
            raise KeyError(f"Missing pathway '{pathway}' in conversion_matrix_summary.csv")
        feedstock_name = str(row["feedstock_name"].iloc[0]).strip()
        kg_feed_per_MJ = float(row["kg_feedstock_per_MJ_SAF"].iloc[0])

        feed_key = "CO2 capture" if feedstock_name.lower().startswith("co2") else feedstock_name
        feed_price, feed_unit = price_get(price, feed_key)
        if feed_price is None:
            raise KeyError(f"Missing A.23 price for feedstock '{feed_key}' (pathway={pathway})")
        if feed_unit != "$/t":
            raise ValueError(f"Feedstock '{feed_key}' is expected to use $/t, got {feed_unit}")

        feed_cost = (feed_price / 1000.0) * kg_feed_per_MJ
        feed_bucket = "CCS" if feed_key == "CO2 capture" else "Feedstock"

        debug_inputs.append(
            {
                "pathway": pathway,
                "category": feed_bucket,
                "item": feed_key,
                "quantity_per_MJ_SAF": kg_feed_per_MJ,
                "unit": "kg",
                "price": feed_price,
                "price_unit": feed_unit,
                "usd_per_MJ": feed_cost,
                "note": "Delivered feedstock from conversion_matrix_summary.csv",
            }
        )

        ng_MJ = 0.0
        electricity_MJ = 0.0
        h2_kg = 0.0
        coproduct_credit = 0.0

        for _, fg_row in foreground.iterrows():
            stage = str(fg_row["stage"]).strip()
            basis = str(fg_row["basis"]).strip()
            flow = str(fg_row["flow_name"]).strip()
            flow_type = str(fg_row["flow_type"]).strip()
            is_coproduct = int(fg_row["is_coproduct"])

            factor = factor_to_MJSAF(pathway, basis, conv_sum, inter)
            amount = float(fg_row["amount"]) * factor
            unit, amount = unit_to_kg_if_g(str(fg_row["unit"]), amount)

            if flow in {"Natural gas", "NG"} and flow_type == "input" and unit == "MJ":
                ng_MJ += amount
                continue
            if flow == "Electricity" and flow_type == "input" and unit == "MJ":
                electricity_MJ += amount
                continue
            if flow == "H2" and flow_type == "input":
                if unit != "kg":
                    raise ValueError(f"H2 must be expressed in kg after normalization. Got {unit} in pathway {pathway}.")
                h2_kg += amount
                continue

            if is_coproduct == 1 and flow_type == "output":
                if flow in {"Diesel", "Naphtha", "Propane", "Gasoline", "Heavy fuel oil"}:
                    if unit != "MJ":
                        raise ValueError(f"Co-product {flow} is expected in MJ, got {unit} in pathway {pathway}")
                    co_price, co_unit = price_get(price, flow)
                    if co_price is None:
                        raise KeyError(f"Missing A.23 co-product price for {flow}")
                    if co_unit != "$/MJ":
                        raise ValueError(f"Co-product {flow} is expected to use $/MJ, got {co_unit}")
                    credit = amount * co_price
                    coproduct_credit += credit
                    debug_inputs.append(
                        {
                            "pathway": pathway,
                            "category": "Co-product",
                            "item": flow,
                            "quantity_per_MJ_SAF": amount,
                            "unit": "MJ",
                            "price": co_price,
                            "price_unit": co_unit,
                            "usd_per_MJ": -credit,
                            "note": "Output credit",
                        }
                    )
                    continue

                if flow == "Meal":
                    meal_key = "Soybean Meal" if pathway == "HEFA_Soybean" else "Meal (other kind)"
                    meal_price, meal_unit = price_get(price, meal_key)
                    if meal_price is None:
                        raise KeyError(f"Missing A.23 co-product price for meal in pathway {pathway}")
                    if meal_unit != "$/t":
                        raise ValueError(f"Meal is expected to use $/t, got {meal_unit}")

                    if unit == "MJ":
                        kg_meal = amount / meal_lhv_mj_per_kg
                    elif unit == "kg":
                        kg_meal = amount
                    else:
                        raise ValueError(f"Meal must be expressed in MJ or kg. Got {unit} in pathway {pathway}")

                    credit = (meal_price / 1000.0) * kg_meal
                    coproduct_credit += credit
                    debug_inputs.append(
                        {
                            "pathway": pathway,
                            "category": "Co-product",
                            "item": "Meal",
                            "quantity_per_MJ_SAF": kg_meal,
                            "unit": "kg",
                            "price": meal_price,
                            "price_unit": meal_unit,
                            "usd_per_MJ": -credit,
                            "note": f"Meal credit; MJ to kg conversion uses {meal_lhv_mj_per_kg} MJ/kg when needed",
                        }
                    )
                    continue

            debug_ignored.append(
                {
                    "pathway": pathway,
                    "stage": stage,
                    "flow_name": flow,
                    "flow_type": flow_type,
                    "is_coproduct": is_coproduct,
                    "amount_per_MJ_SAF": amount,
                    "unit": unit,
                    "basis": basis,
                    "ignored_reason": "Not priced individually under the manuscript TEA boundary",
                }
            )

        ng_price, ng_unit = price_get(price, "NG")
        if ng_price is None:
            ng_price, ng_unit = price_get(price, "Natural gas")
        if ng_price is None:
            raise KeyError("Missing A.23 price for natural gas")
        if ng_unit != "$/MJ":
            raise ValueError(f"Natural gas is expected to use $/MJ, got {ng_unit}")
        ng_cost = ng_MJ * ng_price

        electricity_price, electricity_unit = price_get(price, "Electricity")
        if electricity_price is None:
            raise KeyError("Missing A.23 price for electricity")
        if electricity_unit != "$/MJ":
            raise ValueError(f"Electricity is expected to use $/MJ, got {electricity_unit}")
        electricity_cost = electricity_MJ * electricity_price

        h2_price, h2_unit = price_get(price, "H2")
        if h2_price is None:
            raise KeyError("Missing A.23 price for H2")
        if h2_unit != "$/kg":
            raise ValueError(f"H2 is expected to use $/kg, got {h2_unit}")
        h2_cost = h2_kg * h2_price

        if ng_MJ:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Energy input",
                    "item": "NG",
                    "quantity_per_MJ_SAF": ng_MJ,
                    "unit": "MJ",
                    "price": ng_price,
                    "price_unit": ng_unit,
                    "usd_per_MJ": ng_cost,
                    "note": "Foreground natural gas use",
                }
            )
        if electricity_MJ:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Energy input",
                    "item": "Electricity",
                    "quantity_per_MJ_SAF": electricity_MJ,
                    "unit": "MJ",
                    "price": electricity_price,
                    "price_unit": electricity_unit,
                    "usd_per_MJ": electricity_cost,
                    "note": "Foreground electricity use",
                }
            )
        if h2_kg:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Energy input",
                    "item": "H2",
                    "quantity_per_MJ_SAF": h2_kg,
                    "unit": "kg",
                    "price": h2_price,
                    "price_unit": h2_unit,
                    "usd_per_MJ": h2_cost,
                    "note": "Foreground H2 use",
                }
            )

        transport_subset = transport_lci.loc[transport_lci["pathway"] == pathway]
        transport_cost = 0.0
        for _, tr_row in transport_subset.iterrows():
            flow_name = str(tr_row["flow_name"])
            unit = str(tr_row["unit"]).strip()
            amount = float(tr_row["amount"])
            if unit != "tkm":
                raise ValueError(
                    f"Transport unit must be tkm in transport_LCI_per_MJ.csv. Got {unit} for {flow_name} ({pathway})"
                )

            mode = flow_name.split("|")[-1].strip()
            if mode not in bts:
                raise KeyError(f"Missing BTS rate for transport mode '{mode}'")
            rate = float(bts[mode])
            item_cost = amount * rate
            transport_cost += item_cost

            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Transportation",
                    "item": mode,
                    "quantity_per_MJ_SAF": amount,
                    "unit": "tkm",
                    "price": rate,
                    "price_unit": "$/t-km",
                    "usd_per_MJ": item_cost,
                    "note": "BTS rate × transport demand",
                }
            )

        annual_feed_kg = plant_scale_mt_day * 1000.0 * 365.0 * capacity_factor
        annual_MJ_SAF = annual_feed_kg / kg_feed_per_MJ

        process_map = PROCESS_MAP.get(pathway)
        if process_map is None:
            raise KeyError(f"Missing PROCESS_MAP entry for pathway '{pathway}'")

        def tci_sum_musd(processes: list[str]) -> float:
            missing = [proc for proc in processes if proc not in tci]
            if missing:
                raise KeyError(f"Missing TCI entries {missing} in pathway {pathway}")
            return sum(float(tci[proc]) for proc in processes)

        tci_eth_musd = tci_sum_musd(process_map["ethanol"]) if process_map["ethanol"] else 0.0
        tci_saf_musd = tci_sum_musd(process_map["saf"]) if process_map["saf"] else 0.0
        tci_total_usd = (tci_eth_musd + tci_saf_musd) * 1e6

        cap_cost_eth = (tci_eth_musd * 1e6 * capital_recovery_factor) / annual_MJ_SAF if tci_eth_musd else 0.0
        cap_cost_saf = (tci_saf_musd * 1e6 * capital_recovery_factor) / annual_MJ_SAF if tci_saf_musd else 0.0
        opex_cost = (tci_total_usd * opex_pct) / annual_MJ_SAF
        labor_cost = (tci_total_usd * labor_pct) / annual_MJ_SAF

        if cap_cost_eth:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Capital",
                    "item": "Ethanol_production_cost",
                    "quantity_per_MJ_SAF": 1.0,
                    "unit": "MJ_SAF",
                    "price": cap_cost_eth,
                    "price_unit": "$/MJ_SAF",
                    "usd_per_MJ": cap_cost_eth,
                    "note": "Annualized capex via CRF",
                }
            )
        if cap_cost_saf:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Capital",
                    "item": "SAF_production_cost",
                    "quantity_per_MJ_SAF": 1.0,
                    "unit": "MJ_SAF",
                    "price": cap_cost_saf,
                    "price_unit": "$/MJ_SAF",
                    "usd_per_MJ": cap_cost_saf,
                    "note": "Annualized capex via CRF",
                }
            )
        if opex_cost:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Fixed_cost",
                    "item": "OPEX",
                    "quantity_per_MJ_SAF": 1.0,
                    "unit": "MJ_SAF",
                    "price": opex_cost,
                    "price_unit": "$/MJ_SAF",
                    "usd_per_MJ": opex_cost,
                    "note": "Operating expenditure derived from A.22 parameters",
                }
            )
        if labor_cost:
            debug_inputs.append(
                {
                    "pathway": pathway,
                    "category": "Fixed_cost",
                    "item": "Labor",
                    "quantity_per_MJ_SAF": 1.0,
                    "unit": "MJ_SAF",
                    "price": labor_cost,
                    "price_unit": "$/MJ_SAF",
                    "usd_per_MJ": labor_cost,
                    "note": "Labor expenditure derived from A.22 parameters",
                }
            )

        output_row = {
            "pathway": pathway,
            "Ethanol_production_cost": cap_cost_eth,
            "SAF_production_cost": cap_cost_saf,
            "OPEX": opex_cost,
            "Labor": labor_cost,
            "NG": ng_cost,
            "Electricity": electricity_cost,
            "H2": h2_cost,
            "CCS": feed_cost if feed_bucket == "CCS" else 0.0,
            "Feedstock": feed_cost if feed_bucket == "Feedstock" else 0.0,
            "Transportation": transport_cost,
            "Co_product_credit": -coproduct_credit,
        }
        output_row["Total"] = sum(value for key, value in output_row.items() if key != "pathway")
        rows.append(output_row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "tea_cost_table_usd_per_MJ.csv", index=False)
    out_df[["pathway", "Total"]].to_csv(out_dir / "tea_total_cost_usd_per_MJ.csv", index=False)

    df_param_cost = pd.DataFrame(debug_inputs)
    if not df_param_cost.empty:
        keep_cols = [col for col in ["pathway", "category", "item", "usd_per_MJ"] if col in df_param_cost.columns]
        df_param_cost = df_param_cost[keep_cols].copy()
        df_param_cost_agg = (
            df_param_cost.groupby(["pathway", "category", "item"], as_index=False)
            .agg(usd_per_MJ=("usd_per_MJ", "sum"))
        )
        df_param_cost_agg.to_csv(out_dir / "tea_parameter_costs_usd_per_MJ.csv", index=False)
    else:
        pd.DataFrame(columns=["pathway", "category", "item", "usd_per_MJ"]).to_csv(
            out_dir / "tea_parameter_costs_usd_per_MJ.csv", index=False
        )

    if mj_per_l_saf is not None:
        out_l = out_df.copy()
        for col in out_l.columns:
            if col != "pathway":
                out_l[col] = out_l[col] * float(mj_per_l_saf)
        out_l.to_csv(out_dir / "tea_cost_table_usd_per_L.csv", index=False)

    pd.DataFrame(debug_inputs).to_csv(out_dir / "tea_debug_inputs_per_MJ.csv", index=False)
    pd.DataFrame(debug_ignored).to_csv(out_dir / "tea_debug_unmapped_flows.csv", index=False)


# Backward-compatible alias for existing runner scripts.
run_author_aligned_tea = run_tea_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SAF TEA model.")
    parser.add_argument("--out_dir", required=True, help="Results directory, typically results/run_*")
    parser.add_argument("--meal_lhv", type=float, default=MEAL_LHV_MJ_PER_KG_DEFAULT)
    parser.add_argument(
        "--mj_per_l_saf",
        type=float,
        default=None,
        help="If provided, also write a $/L output using this MJ/L conversion factor.",
    )
    args = parser.parse_args()

    run_tea_model(
        out_dir=args.out_dir,
        meal_lhv_mj_per_kg=args.meal_lhv,
        mj_per_l_saf=args.mj_per_l_saf,
    )
    print(f"TEA run completed. Outputs written to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
