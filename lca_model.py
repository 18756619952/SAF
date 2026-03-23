"""
LCA model for seven SAF pathways.

This script:
1. Loads pathway foreground inventories and background emission-factor tables.
2. Converts all relevant quantities to a common functional basis of 1 MJ SAF.
3. Applies basis-aware and unit-aware matching to background emission factors.
4. Computes unallocated and allocated GHG contributions by flow and by stage.
5. Writes run outputs and a run manifest to a results directory.

Outputs:
  <results_dir>/
    - results_flow_contributions.csv
    - results_parameter_contributions.csv
    - results_total_ghg.csv
    - coverage_missing_background_EF.csv   (only if unmatched flows exist)
    - run_manifest.json

"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "inputs")

FOREGROUND_FILES: Dict[str, str] = {
    "HEFA_Soybean": "foreground_HEFA_Soybean.csv",
    "HEFA_UCO": "foreground_HEFA_UCO.csv",
    "ATJ_FR": "foreground_ATJ_FR.csv",
    "ATJ_Switchgrass": "foreground_ATJ_Switchgrass.csv",
    "FT_Miscanthus": "foreground_FT_Miscanthus.csv",
    "FT_Switchgrass": "foreground_FT_Switchgrass.csv",
    "PtL_DAC": "foreground_PtL_DAC.csv",
}

BACKGROUND_ENERGY_FILE = "background_energy_EF.csv"
BACKGROUND_PROCESS_FILE = "background_process_EF.csv"
CONV_RAW_FILE = "conversion_matrix_raw.csv"
CONV_SUMMARY_FILE = "conversion_matrix_summary.csv"
TRANSPORT_LCI_FILE = "transport_LCI_per_MJ.csv"

LHV_DICT = {
    "diesel": 43.0,
    "gasoline": 43.5,
    "lpg": 46.0,
    "natural gas": 50.0,
    "naphtha": 44.0,
    "propane": 46.4,
    "n-hexane": 44.752,
    "hexane": 44.752,
}

FLOW_RENAME = {
    "cobalts": "Cobalt",
    "electricity, high voltage": "Electricity",
}

TRANSPORT_FLOW_MAP = {
    "freight | uco | heavy heavy-duty truck": "Heavy-heavy Truck",
    "freight | soybean | heavy heavy-duty truck": "Heavy-heavy Truck",
    "freight | forest residues | heavy heavy-duty truck": "Heavy-heavy Truck",
    "freight | switchgrass | heavy heavy-duty truck": "Heavy-heavy Truck",
    "freight | miscanthus | heavy heavy-duty truck": "Heavy-heavy Truck",
    "freight | msw | heavy heavy-duty truck": "Heavy-heavy Truck",
    "freight | uco | rail": "Rail",
    "freight | soybean | rail": "Rail",
    "freight | saf | barge": "Barge",
    "freight | saf | rail": "Rail",
    "freight | saf | heavy heavy-duty truck": "Heavy-heavy Truck",
    "distribution | saf | heavy heavy-duty truck": "Heavy-heavy Truck",
}

IGNORE_LIST = {"co2"}
INTERNAL_TOKENS = ["feedstock", "bio-oil", "etoh", "fr"]
TRANSPORT_TOKENS = ["truck", "rail", "barge"]

FLOW_GHG_LOG: List[List[object]] = []
VERBOSE = False

# Lazily populated globals used by helper functions
background_df: pd.DataFrame | None = None
conv_raw: pd.DataFrame | None = None
conv_summary: pd.DataFrame | None = None
kg_feedstock_per_MJ: Dict[str, float] = {}


def _norm(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} must contain columns {required}; missing {missing}")


def load_energy_EF(path: str) -> pd.DataFrame:
    """Load energy emission factors and normalize to kg CO2e per physical unit."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    _require_columns(df, {"energy_type", "unit", "range_mean"}, "background energy EF file")

    records = []
    for _, row in df.iterrows():
        product = str(row["energy_type"]).strip()
        unit = str(row["unit"]).strip().lower()
        mean_val = float(row["range_mean"])

        if "/mj" in unit:
            records.append(
                {
                    "product": product,
                    "product_lower": product.lower(),
                    "physical_basis": "MJ",
                    "gwp_per_unit": mean_val / 1000.0,
                }
            )
        elif "/g" in unit:
            records.append(
                {
                    "product": product,
                    "product_lower": product.lower(),
                    "physical_basis": "kg",
                    "gwp_per_unit": mean_val,
                }
            )

    return pd.DataFrame.from_records(records)


def load_process_EF(path: str) -> pd.DataFrame:
    """Load process emission factors."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    _require_columns(df, {"Product", "gwp_value", "physical_basis"}, "background process EF file")

    df = df.rename(columns={"Product": "product", "gwp_value": "gwp_per_unit"})
    df["product_lower"] = df["product"].astype(str).str.strip().str.lower()
    return df[["product", "product_lower", "physical_basis", "gwp_per_unit"]]


def load_foreground_inputs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "flow_type" in df.columns:
        df = df[df["flow_type"].astype(str).str.lower() == "input"].copy()
    return df


def load_foreground_full(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def initialize_data() -> None:
    """Load background tables and conversion matrices into module globals."""
    global background_df, conv_raw, conv_summary, kg_feedstock_per_MJ

    energy_EF = load_energy_EF(os.path.join(DATA_DIR, BACKGROUND_ENERGY_FILE))
    process_EF = load_process_EF(os.path.join(DATA_DIR, BACKGROUND_PROCESS_FILE))

    background_df = pd.concat([energy_EF, process_EF], ignore_index=True)
    background_df["product_lower"] = background_df["product_lower"].astype(str)
    mask_transport = background_df["product_lower"].str.contains("|".join(TRANSPORT_TOKENS), na=False)
    background_df.loc[mask_transport, "physical_basis"] = "tkm"

    conv_raw = pd.read_csv(os.path.join(DATA_DIR, CONV_RAW_FILE))
    conv_raw.columns = conv_raw.columns.str.strip()
    conv_summary = pd.read_csv(os.path.join(DATA_DIR, CONV_SUMMARY_FILE))
    conv_summary.columns = conv_summary.columns.str.strip()
    _require_columns(conv_summary, {"pathway", "kg_feedstock_per_MJ_SAF"}, "conversion_matrix_summary.csv")

    kg_feedstock_per_MJ = {
        row["pathway"]: float(row["kg_feedstock_per_MJ_SAF"])
        for _, row in conv_summary.iterrows()
    }


def get_co2_per_MJ_SAF(pathway: str) -> float:
    """Return pathway-specific CO2 requirement in kg CO2 per MJ SAF."""
    assert conv_raw is not None
    _require_columns(
        conv_raw,
        {"pathway", "bridge_type", "from_flow", "amount"},
        "conversion_matrix_raw.csv",
    )

    sub = conv_raw[
        (conv_raw["pathway"] == pathway)
        & (conv_raw["bridge_type"].astype(str).str.contains("intermediate_per_MJ_SAF", case=False, na=False))
        & (conv_raw["from_flow"].astype(str).str.contains("co2", case=False, na=False))
    ].copy()

    if sub.empty:
        raise ValueError(
            f"Cannot find pathway-specific CO2 requirement for '{pathway}' in conversion_matrix_raw.csv."
        )
    if len(sub) != 1:
        raise ValueError(
            f"Multiple CO2 requirement rows found for '{pathway}' in conversion_matrix_raw.csv:\n"
            f"{sub[['pathway', 'bridge_type', 'from_flow', 'amount']].to_string(index=False)}"
        )
    return float(sub.iloc[0]["amount"])


def basis_factor(pathway: str, basis: object) -> float:
    """Convert a pathway inventory basis to the functional unit basis of 1 MJ SAF."""
    if not isinstance(basis, str):
        return 1.0

    basis_clean = basis.replace(" ", "").lower()
    assert conv_raw is not None

    if "per_mj_saf" in basis_clean:
        return 1.0

    if "kg_feed" in basis_clean or "perdrykgfeedstock" in basis_clean:
        return kg_feedstock_per_MJ[pathway]

    if "per_mj_ethanol" in basis_clean:
        sub = conv_raw[
            (conv_raw["pathway"] == pathway)
            & (conv_raw["bridge_type"].astype(str).str.contains("intermediate_per_MJ_SAF", case=False, na=False))
            & (conv_raw["from_flow"].astype(str).str.contains("etoh", case=False, na=False))
        ]
        return float(sub.iloc[0]["amount"]) if not sub.empty else 0.0

    if "per_mj_oil" in basis_clean:
        sub = conv_raw[
            (conv_raw["pathway"] == pathway)
            & (conv_raw["bridge_type"].astype(str).str.contains("intermediate_per_MJ_SAF", case=False, na=False))
            & (conv_raw["from_flow"].astype(str).str.lower().str.contains("bio-oil"))
        ]
        return float(sub.iloc[0]["amount"]) if not sub.empty else 0.0

    if "per_kg_co2_captured" in basis_clean or ("per_kg_co2" in basis_clean and "captur" in basis_clean):
        return get_co2_per_MJ_SAF(pathway)

    if "co2" in basis_clean and "per_" in basis_clean and "mj_saf" not in basis_clean:
        raise ValueError(
            f"Unrecognized CO2-related basis '{basis}' for pathway '{pathway}'."
        )

    return 1.0


def unify_unit(amount: float, unit: object) -> Tuple[float, object]:
    """Normalize grams to kilograms; leave other units unchanged."""
    if not isinstance(unit, str):
        return amount, unit
    unit_clean = unit.lower().strip()
    if unit_clean in {"g", "gram", "grams"}:
        return amount / 1000.0, "kg"
    return amount, unit


def _canon_unit(unit: object) -> object:
    if not isinstance(unit, str):
        return unit
    text = unit.strip()
    text_lower = text.lower()
    if text_lower == "mj":
        return "MJ"
    if text_lower == "kg":
        return "kg"
    if text_lower == "tkm":
        return "tkm"
    return text


def convert_to_basis(flow_lower: str, amount: float, unit: object, basis: object) -> float:
    """Convert a foreground quantity to the physical basis used by the matched EF row."""
    unit_c = _canon_unit(unit)
    basis_c = _canon_unit(basis)

    if unit_c == basis_c:
        return amount

    if unit_c == "MJ" and basis_c == "kg":
        if flow_lower not in LHV_DICT:
            raise ValueError(f"LHV is not defined for flow '{flow_lower}'.")
        return amount / LHV_DICT[flow_lower]

    if unit_c == "kg" and basis_c == "MJ":
        if flow_lower not in LHV_DICT:
            raise ValueError(f"LHV is not defined for flow '{flow_lower}'.")
        return amount * LHV_DICT[flow_lower]

    if unit_c == "tkm" and basis_c in {"tkm", "kg"}:
        return amount

    raise ValueError(
        f"Unit mismatch for flow '{flow_lower}': foreground unit={unit_c}, background basis={basis_c}"
    )


def add_transport_demand(demand: Dict[str, float], pathway: str) -> Dict[str, float]:
    """Add transport demand quantities (already per MJ SAF and in tkm)."""
    assert background_df is not None
    df = pd.read_csv(os.path.join(DATA_DIR, TRANSPORT_LCI_FILE))
    df.columns = df.columns.str.strip()
    df_p = df[df["pathway"] == pathway].copy()

    for _, row in df_p.iterrows():
        flow = str(row["flow_name"]).strip()
        unit = str(row["unit"]).strip()
        amount = float(row["amount"])
        flow_lower = _norm(flow)

        if flow_lower in FLOW_RENAME:
            flow = FLOW_RENAME[flow_lower]
            flow_lower = _norm(flow)
        if flow_lower in TRANSPORT_FLOW_MAP:
            flow = TRANSPORT_FLOW_MAP[flow_lower]
            flow_lower = _norm(flow)

        match = background_df[background_df["product_lower"] == flow_lower]
        if match.empty:
            match = background_df[background_df["product_lower"].str.contains(flow_lower, na=False)]
        if match.empty:
            raise ValueError(f"No background EF match found for transport flow '{flow}' (pathway={pathway}).")

        matched = match.iloc[0]
        basis_bg = matched["physical_basis"]
        amount_in_basis = convert_to_basis(flow_lower, amount, unit, basis_bg)

        key = matched["product"]
        demand[key] = demand.get(key, 0.0) + amount_in_basis

    return demand


ENERGY_COPRODUCTS = ["naphtha", "propane", "diesel", "gasoline", "kerosene", "electricity", "meal"]


def get_etoh_per_MJ_SAF(pathway: str) -> float:
    assert conv_raw is not None
    sub = conv_raw[
        (conv_raw["pathway"] == pathway)
        & (conv_raw["bridge_type"].astype(str).str.contains("intermediate_per_MJ_SAF", case=False, na=False))
        & (conv_raw["from_flow"].astype(str).str.contains("etoh", case=False, na=False))
    ]
    if sub.empty:
        return 1.0
    return float(sub.iloc[0]["amount"])


def oil_allocation_factor(pathway: str) -> float:
    """
    Allocation factor for oil-stage co-products (used for HEFA_Soybean upstream stages).
    """
    csv_path = os.path.join(DATA_DIR, FOREGROUND_FILES[pathway])
    df = load_foreground_full(csv_path).copy()

    df["basis"] = df["basis"].astype(str)
    df["flow_type"] = df["flow_type"].astype(str)
    df["flow_name"] = df["flow_name"].astype(str)

    oil_mask = df["basis"].str.contains("per_MJ_oil", case=False, na=False)
    out_mask = df["flow_type"].str.lower().eq("output")
    cop_mask = df["is_coproduct"] == 1

    oil_cop = df[oil_mask & out_mask & cop_mask].copy()
    if oil_cop.empty:
        return 1.0

    oil_cop["flow_name_lower"] = oil_cop["flow_name"].str.lower()
    oil_cop = oil_cop[oil_cop["flow_name_lower"].str.contains("|".join(ENERGY_COPRODUCTS), na=False)]
    if oil_cop.empty:
        return 1.0

    energy_cop_oil = oil_cop["amount"].astype(float).sum()
    factor = 1.0 / (1.0 + energy_cop_oil) if energy_cop_oil > 0 else 1.0

    if VERBOSE:
        print(
            f"[DEBUG] Oil-stage allocation for {pathway}: "
            f"E_cop_oil_per_MJ_oil={energy_cop_oil:.4f}, f_oil={factor:.4f}"
        )
    return factor


def energy_allocation_factor(pathway: str) -> float:
    """Calculate the SAF energy allocation factor from foreground outputs."""
    csv_path = os.path.join(DATA_DIR, FOREGROUND_FILES[pathway])
    df = load_foreground_full(csv_path).copy()

    df["basis"] = df["basis"].astype(str)
    df["flow_type"] = df["flow_type"].astype(str)
    df["flow_name"] = df["flow_name"].astype(str)

    saf_mask = df["basis"].str.contains("per_MJ_SAF", case=False, na=False)
    out_mask = df["flow_type"].str.lower().eq("output")
    cop_mask = df["is_coproduct"] == 1

    saf_cop = df[saf_mask & out_mask & cop_mask].copy()
    if not saf_cop.empty:
        saf_cop["flow_name_lower"] = saf_cop["flow_name"].str.lower()
        saf_cop = saf_cop[saf_cop["flow_name_lower"].str.contains("|".join(ENERGY_COPRODUCTS), na=False)]
        energy_cop_saf = saf_cop["amount"].astype(float).sum()
    else:
        energy_cop_saf = 0.0

    etoh_mask = df["basis"].str.contains("per_MJ_ethanol", case=False, na=False)
    etoh_cop = df[etoh_mask & out_mask & cop_mask].copy()

    if not etoh_cop.empty:
        etoh_cop["flow_name_lower"] = etoh_cop["flow_name"].str.lower()
        etoh_cop = etoh_cop[etoh_cop["flow_name_lower"].str.contains("|".join(ENERGY_COPRODUCTS), na=False)]
        energy_cop_etoh = etoh_cop["amount"].astype(float).sum()
    else:
        energy_cop_etoh = 0.0

    etoh_per_MJ_SAF = get_etoh_per_MJ_SAF(pathway) if energy_cop_etoh > 0 else 0.0
    energy_cop_total = energy_cop_saf + energy_cop_etoh * etoh_per_MJ_SAF

    factor = 1.0 if energy_cop_total <= 0 else 1.0 / (1.0 + energy_cop_total)

    if VERBOSE:
        print(
            f"[DEBUG] Energy allocation for {pathway}: "
            f"E_cop_saf={energy_cop_saf:.4f}, "
            f"E_cop_etoh={energy_cop_etoh:.4f}, "
            f"EtOH_per_MJ_SAF={etoh_per_MJ_SAF:.4f}, "
            f"E_cop_total={energy_cop_total:.4f}, "
            f"f_SAF={factor:.4f}"
        )

    return factor


def add_transport_stage_emissions(stage_gwp: Dict[str, float], pathway: str) -> None:
    """
    Add transport emissions and assign feedstock transport to upstream stages and
    SAF transport/distribution to the Transportation stage.
    """
    assert background_df is not None
    df = pd.read_csv(os.path.join(DATA_DIR, TRANSPORT_LCI_FILE))
    df.columns = df.columns.str.strip()
    df_p = df[df["pathway"] == pathway].copy()

    for _, row in df_p.iterrows():
        stage_raw = str(row.get("stage", "")).strip().lower()
        stage_label = "Feedstock production & collection" if stage_raw == "feedstock_transport" else "Transportation"

        flow = str(row["flow_name"]).strip()
        unit = str(row["unit"]).strip()
        amount = float(row["amount"])
        flow_lower = _norm(flow)

        if flow_lower in FLOW_RENAME:
            flow = FLOW_RENAME[flow_lower]
            flow_lower = _norm(flow)
        if flow_lower in TRANSPORT_FLOW_MAP:
            flow = TRANSPORT_FLOW_MAP[flow_lower]
            flow_lower = _norm(flow)

        match = background_df[background_df["product_lower"] == flow_lower]
        if match.empty:
            match = background_df[background_df["product_lower"].str.contains(flow_lower, na=False)]
        if match.empty:
            raise ValueError(f"No background EF match found for transport flow '{flow}' (pathway={pathway}).")

        matched = match.iloc[0]
        physical_basis = matched["physical_basis"]
        ef = float(matched["gwp_per_unit"])
        amount_in_basis = convert_to_basis(flow_lower, amount, unit, physical_basis)
        gwp_unallocated = ef * amount_in_basis

        stage_gwp[stage_label] += gwp_unallocated

        FLOW_GHG_LOG.append(
            [
                pathway,
                stage_label,
                matched["product"],
                float(amount),
                unit,
                "per_MJ_SAF",
                physical_basis,
                float(amount_in_basis),
                ef,
                gwp_unallocated,
            ]
        )


def calc_stage_breakdown(pathway: str) -> Dict[str, float]:
    """Calculate unallocated GHG contributions by stage for one pathway."""
    assert background_df is not None

    csv_path = os.path.join(DATA_DIR, FOREGROUND_FILES[pathway])
    df = load_foreground_inputs(csv_path).copy()

    stage_gwp = {
        "Feedstock production & collection": 0.0,
        "Extraction": 0.0,
        "Ethanol production": 0.0,
        "CO2 capture": 0.0,
        "CO2 stored": 0.0,
        "H2": 0.0,
        "SAF production": 0.0,
        "Transportation": 0.0,
    }

    for _, row in df.iterrows():
        flow_name = str(row["flow_name"]).strip()
        amount_original = float(row["amount"])
        unit_original = str(row["unit"])
        basis_original = row["basis"]
        stage_raw = str(row.get("stage", "")).strip().lower()

        factor = basis_factor(pathway, basis_original)
        amount_per_MJ_SAF = amount_original * factor
        amount_phys, unit_phys = unify_unit(amount_per_MJ_SAF, unit_original)
        flow_lower = _norm(flow_name)

        if any(token in flow_lower for token in INTERNAL_TOKENS):
            continue
        if flow_lower in IGNORE_LIST:
            continue

        if flow_lower in FLOW_RENAME:
            flow_name = FLOW_RENAME[flow_lower]
            flow_lower = _norm(flow_name)
        if flow_lower in TRANSPORT_FLOW_MAP:
            flow_name = TRANSPORT_FLOW_MAP[flow_lower]
            flow_lower = _norm(flow_name)

        match = background_df[background_df["product_lower"] == flow_lower]
        if match.empty:
            match = background_df[background_df["product_lower"].str.contains(flow_lower, na=False)]
        if match.empty:
            raise ValueError(f"No background EF match found for flow '{flow_name}' (pathway={pathway}).")

        matched = match.iloc[0]
        physical_basis = matched["physical_basis"]
        ef = float(matched["gwp_per_unit"])
        amount_in_basis = convert_to_basis(flow_lower, amount_phys, unit_phys, physical_basis)
        gwp_unallocated = ef * amount_in_basis

        stage_label = "SAF production"
        basis_text = str(basis_original).lower() if isinstance(basis_original, str) else ""
        if ("co2" in basis_text) and ("captur" in basis_text):
            stage_label = "CO2 capture"
        elif stage_raw in {"feedstock_production", "feedstock"}:
            stage_label = "Feedstock production & collection"
        elif stage_raw in {"oil_extraction", "rendering", "pretreatment"}:
            stage_label = "Extraction"
        elif stage_raw == "etoh_production":
            stage_label = "Ethanol production"
        elif stage_raw == "saf_production":
            stage_label = "H2" if flow_lower == "h2" else "SAF production"

        stage_gwp[stage_label] += gwp_unallocated

        FLOW_GHG_LOG.append(
            [
                pathway,
                stage_label,
                flow_name,
                amount_original,
                unit_original,
                str(basis_original),
                physical_basis,
                amount_in_basis,
                ef,
                gwp_unallocated,
            ]
        )

    add_transport_stage_emissions(stage_gwp, pathway)
    return stage_gwp


def write_stage_contributions_transport_unified(
    df_flow: pd.DataFrame,
    df_stage: pd.DataFrame,
    run_dir: str,
) -> None:
    """
    Optional reporting helper: collapse all tkm-based transport rows into a single
    Transportation bucket while preserving the allocation factors already applied.
    """
    f_lookup = {(row["pathway"], row["stage"]): float(row["f_SAF"]) for _, row in df_stage.iterrows()}

    df = df_flow.copy()
    df["physical_basis"] = df["physical_basis"].astype(str).str.strip().str.lower()
    df["is_transport"] = df["physical_basis"].eq("tkm")

    df["f_SAF_applied"] = df.apply(lambda row: f_lookup.get((row["pathway"], row["stage"]), float("nan")), axis=1)
    if df["f_SAF_applied"].isna().any():
        missing = df[df["f_SAF_applied"].isna()][["pathway", "stage"]].drop_duplicates()
        raise ValueError(f"Missing allocation lookup for some (pathway, stage) pairs:\n{missing}")

    df["gwp_allocated_kgCO2_per_MJ_SAF"] = (
        df["gwp_unallocated_kgCO2_per_MJ_SAF"] * df["f_SAF_applied"]
    )

    df["stage_report"] = df["stage"]
    df.loc[df["is_transport"], "stage_report"] = "Transportation"

    out = (
        df.groupby(["pathway", "stage_report"], as_index=False)
        .agg(
            gwp_unallocated_kgCO2_per_MJ_SAF=("gwp_unallocated_kgCO2_per_MJ_SAF", "sum"),
            gwp_allocated_kgCO2_per_MJ_SAF=("gwp_allocated_kgCO2_per_MJ_SAF", "sum"),
            f_SAF_min=("f_SAF_applied", "min"),
            f_SAF_max=("f_SAF_applied", "max"),
        )
    )
    out["f_SAF_effective"] = out.apply(
        lambda row: (
            row["gwp_allocated_kgCO2_per_MJ_SAF"] / row["gwp_unallocated_kgCO2_per_MJ_SAF"]
            if row["gwp_unallocated_kgCO2_per_MJ_SAF"] > 0
            else float("nan")
        ),
        axis=1,
    )

    stage_order = [
        "Feedstock production & collection",
        "Extraction",
        "Ethanol production",
        "CO2 capture",
        "CO2 stored",
        "H2",
        "SAF production",
        "Transportation",
    ]
    out["stage_report"] = pd.Categorical(out["stage_report"], categories=stage_order, ordered=True)
    out.sort_values(["pathway", "stage_report"], inplace=True)

    out_path = os.path.join(run_dir, "results_stage_contributions_transport_unified.csv")
    out.to_csv(out_path, index=False)


def check_background_coverage(run_dir: str) -> None:
    """Check that all relevant foreground inputs can be matched to a background EF."""
    assert background_df is not None
    missing = []

    for pathway, csv_file in FOREGROUND_FILES.items():
        df = load_foreground_full(os.path.join(DATA_DIR, csv_file)).copy()
        df["flow_type"] = df["flow_type"].astype(str)
        df["flow_name"] = df["flow_name"].astype(str)

        for _, row in df.iterrows():
            original = row["flow_name"]
            flow_type = str(row["flow_type"]).lower()
            flow = str(original).strip().lower()

            if flow_type == "output":
                continue
            if any(token in flow for token in INTERNAL_TOKENS):
                continue

            flow_norm = FLOW_RENAME.get(flow, flow).lower()
            if flow_norm in IGNORE_LIST:
                continue

            match = background_df[background_df["product_lower"] == flow_norm]
            if match.empty:
                match = background_df[background_df["product_lower"].str.contains(flow_norm, na=False)]
            if match.empty:
                missing.append([pathway, original])

    if missing:
        missing_df = pd.DataFrame(missing, columns=["pathway", "flow_name"])
        out_missing = os.path.join(run_dir, "coverage_missing_background_EF.csv")
        missing_df.to_csv(out_missing, index=False)


def make_run_dir(out_dir: str | None = None) -> str:
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    results_base = os.path.join(BASE_DIR, "results")
    os.makedirs(results_base, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_base, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def write_manifest(run_dir: str) -> None:
    manifest = {
        "run_dir": run_dir,
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "data_dir": DATA_DIR,
        "units": {
            "stage_results": "kg CO2eq / MJ SAF",
            "flow_results": "kg CO2eq / MJ SAF (unallocated per-flow)",
            "note": "Multiply by 1000 for g CO2eq / MJ SAF.",
        },
        "inputs": {
            "background_energy_file": os.path.join(DATA_DIR, BACKGROUND_ENERGY_FILE),
            "background_process_file": os.path.join(DATA_DIR, BACKGROUND_PROCESS_FILE),
            "conversion_raw_file": os.path.join(DATA_DIR, CONV_RAW_FILE),
            "conversion_summary_file": os.path.join(DATA_DIR, CONV_SUMMARY_FILE),
            "transport_lci_file": os.path.join(DATA_DIR, TRANSPORT_LCI_FILE),
            "foreground_files": {key: os.path.join(DATA_DIR, value) for key, value in FOREGROUND_FILES.items()},
        },
        "pathways": list(FOREGROUND_FILES.keys()),
        "co2_factor_rule": (
            "per_kg_CO2_captured is converted using the pathway-specific CO2 requirement "
            "derived from conversion_matrix_raw.csv"
        ),
        "co2_stage_rule": "Bases containing both 'co2' and 'captur' are assigned to stage 'CO2 capture'",
    }

    out_path = os.path.join(run_dir, "run_manifest.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def run_lca(out_dir: str | None = None, verbose: bool = False) -> str:
    """Run the LCA workflow and return the output directory."""
    global FLOW_GHG_LOG, VERBOSE
    VERBOSE = verbose

    initialize_data()
    run_dir = make_run_dir(out_dir)
    write_manifest(run_dir)
    FLOW_GHG_LOG = []

    for pathway in FOREGROUND_FILES:
        calc_stage_breakdown(pathway)

    df_flow = pd.DataFrame(
        FLOW_GHG_LOG,
        columns=[
            "pathway",
            "stage",
            "flow_name",
            "amount_original",
            "unit_original",
            "basis_original",
            "physical_basis",
            "amount_in_physical_basis",
            "EF_kgCO2_per_basis",
            "gwp_unallocated_kgCO2_per_MJ_SAF",
        ],
    )

    f_saf_map = {pathway: float(energy_allocation_factor(pathway)) for pathway in FOREGROUND_FILES}
    f_oil_map = {pathway: (float(oil_allocation_factor(pathway)) if pathway == "HEFA_Soybean" else 1.0)
                 for pathway in FOREGROUND_FILES}

    def applied_factor(row: pd.Series) -> float:
        pathway = row["pathway"]
        stage = str(row["stage"])
        factor = f_saf_map.get(pathway, 1.0)
        if pathway == "HEFA_Soybean" and stage in {"Feedstock production & collection", "Extraction"}:
            factor *= f_oil_map[pathway]
        return float(factor)

    df_flow["f_SAF_applied"] = df_flow.apply(applied_factor, axis=1)
    df_flow["gwp_allocated_kgCO2_per_MJ_SAF"] = (
        df_flow["gwp_unallocated_kgCO2_per_MJ_SAF"] * df_flow["f_SAF_applied"]
    )

    out_flow = os.path.join(run_dir, "results_flow_contributions.csv")
    df_flow.to_csv(out_flow, index=False)

    df_param = (
        df_flow.groupby(["pathway", "flow_name"], as_index=False)
        .agg(
            gwp_unallocated_kgCO2_per_MJ_SAF=("gwp_unallocated_kgCO2_per_MJ_SAF", "sum"),
            gwp_allocated_kgCO2_per_MJ_SAF=("gwp_allocated_kgCO2_per_MJ_SAF", "sum"),
        )
    )
    df_param.to_csv(os.path.join(run_dir, "results_parameter_contributions.csv"), index=False)

    df_total = (
        df_flow.groupby("pathway", as_index=False)
        .agg(
            gwp_unallocated_kgCO2_per_MJ_SAF=("gwp_unallocated_kgCO2_per_MJ_SAF", "sum"),
            gwp_allocated_kgCO2_per_MJ_SAF=("gwp_allocated_kgCO2_per_MJ_SAF", "sum"),
        )
    )
    df_total.to_csv(os.path.join(run_dir, "results_total_ghg.csv"), index=False)

    check_background_coverage(run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SAF LCA model.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional existing or new output directory. If omitted, a timestamped results/run_* directory is created.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print additional diagnostic information.")
    args = parser.parse_args()

    run_dir = run_lca(out_dir=args.out_dir, verbose=args.verbose)
    print(f"LCA run completed. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()
