import pandas as pd, numpy as np, json, re
from pathlib import Path

def melt_years(df):
    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    out = df.melt(
        id_vars=[c for c in df.columns if c not in year_cols],
        value_vars=year_cols,
        var_name="Year",
        value_name="Value"
    )
    out["Year"] = out["Year"].astype(int)
    return out

def load_params(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df["parameter"], df["value"]))

def main():
    root = Path(__file__).resolve().parent
    
    snap = pd.read_csv(root/"ngfs_snapshot_1767525921.csv")
    
    elec_price_snapshot = root/"ngfs_snapshot_1768608708.csv"
    snap_price = pd.read_csv(elec_price_snapshot) if elec_price_snapshot.exists() else snap
    bulk = pd.read_csv(root/"b418bbd2-196f-44b0-aa7f-12180c4b8fa8.csv")
    liq  = pd.read_csv(root/"ngfs_liquids_price.csv")
    params = load_params(root/"h2_parameters_editable.csv")

    snap_long = melt_years(snap)
    snap_price_long = melt_years(snap_price)
    elec_prod = snap_long[snap_long["Variable"]=="Secondary Energy|Electricity"].copy()
    elec_emi  = snap_long[snap_long["Variable"]=="Emissions|CO2|Energy|Supply|Electricity"].copy()
    carbon_p  = snap_long[snap_long["Variable"]=="Price|Carbon"].copy()
    elec_price = snap_price_long[snap_price_long["Variable"]=="Price|Secondary Energy|Electricity"].copy()

    elec = elec_prod.merge(
        elec_emi[["Model","Scenario","Region","Year","Value"]].rename(columns={"Value":"Emissions_MtCO2_per_yr"}),
        on=["Model","Scenario","Region","Year"],
        how="left"
    ).rename(columns={"Value":"Electricity_EJ_per_yr"})
    elec["grid_EF_gCO2_per_MJ"] = elec["Emissions_MtCO2_per_yr"] / elec["Electricity_EJ_per_yr"]
    elec["grid_EF_gCO2_per_kWh"] = elec["grid_EF_gCO2_per_MJ"] * 3.6

    
    elec_price = elec_price.rename(columns={"Value":"electricity_price_USD2010_per_GJ"})[[
        "Model","Scenario","Region","Year","electricity_price_USD2010_per_GJ"
    ]]

    carbon_p = carbon_p.rename(columns={"Value":"carbon_price_USD2010_per_tCO2"})[["Model","Scenario","Region","Year","carbon_price_USD2010_per_tCO2"]]

    bulk_long = melt_years(bulk)
    h2_routes = [
        "Secondary Energy|Hydrogen|Electricity",
        "Secondary Energy|Hydrogen|Fossil|w/ CCS",
        "Secondary Energy|Hydrogen|Fossil|w/o CCS",
        "Secondary Energy|Hydrogen|Biomass|w/ CCS",
        "Secondary Energy|Hydrogen|Biomass|w/o CCS",
        "Secondary Energy|Hydrogen",
        "Price|Secondary Energy|Hydrogen",
    ]
    h2 = bulk_long[bulk_long["Variable"].isin(h2_routes)].copy()
    h2_piv = h2.pivot_table(index=["Model","Scenario","Region","Year"], columns="Variable", values="Value", aggfunc="first").reset_index()
    for v in h2_routes:
        if v not in h2_piv.columns:
            h2_piv[v] = np.nan

    route_cols = [
        "Secondary Energy|Hydrogen|Electricity",
        "Secondary Energy|Hydrogen|Fossil|w/ CCS",
        "Secondary Energy|Hydrogen|Fossil|w/o CCS",
        "Secondary Energy|Hydrogen|Biomass|w/ CCS",
        "Secondary Energy|Hydrogen|Biomass|w/o CCS",
    ]
    h2_piv[route_cols] = h2_piv[route_cols].fillna(0.0)
    h2_total = h2_piv["Secondary Energy|Hydrogen"]

    out_h2 = h2_piv.rename(columns={"Price|Secondary Energy|Hydrogen":"H2_price_USD2010_per_GJ"})[["Model","Scenario","Region","Year","H2_price_USD2010_per_GJ"]]
    out_h2["H2_share_electricity"]  = np.where(h2_total>0, h2_piv["Secondary Energy|Hydrogen|Electricity"]/h2_total, np.nan)
    out_h2["H2_share_fossil_wccs"]  = np.where(h2_total>0, h2_piv["Secondary Energy|Hydrogen|Fossil|w/ CCS"]/h2_total, np.nan)
    out_h2["H2_share_fossil_wocc"]  = np.where(h2_total>0, h2_piv["Secondary Energy|Hydrogen|Fossil|w/o CCS"]/h2_total, np.nan)
    out_h2["H2_share_biomass_wccs"] = np.where(h2_total>0, h2_piv["Secondary Energy|Hydrogen|Biomass|w/ CCS"]/h2_total, np.nan)
    out_h2["H2_share_biomass_wocc"] = np.where(h2_total>0, h2_piv["Secondary Energy|Hydrogen|Biomass|w/o CCS"]/h2_total, np.nan)
    out_h2["H2_share_sum"] = out_h2[["H2_share_electricity","H2_share_fossil_wccs","H2_share_fossil_wocc","H2_share_biomass_wccs","H2_share_biomass_wocc"]].sum(axis=1, skipna=False)

    renorm = np.where((out_h2["H2_share_sum"]>0) & np.isfinite(out_h2["H2_share_sum"]), out_h2["H2_share_sum"], np.nan)
    for c in ["H2_share_electricity","H2_share_fossil_wccs","H2_share_fossil_wocc","H2_share_biomass_wccs","H2_share_biomass_wocc"]:
        out_h2[c+"_renorm"] = np.where(np.isfinite(renorm), out_h2[c]/renorm, np.nan)

    drivers = elec[["Model","Scenario","Region","Year","grid_EF_gCO2_per_kWh","grid_EF_gCO2_per_MJ"]].merge(
        carbon_p, on=["Model","Scenario","Region","Year"], how="left"
    ).merge(
        out_h2, on=["Model","Scenario","Region","Year"], how="left"
    ).merge(
        elec_price, on=["Model","Scenario","Region","Year"], how="left"
    )

   
    electrolyzer_kWh = float(params["electrolyzer_kWh_per_kgH2"])
    ci_fossil_wocc = float(params["CI_fossil_wocc_kgCO2_per_kgH2"])
    ci_fossil_wccs = float(params["CI_fossil_wccs_kgCO2_per_kgH2"])
    ci_biomass_wocc = float(params["CI_biomass_wocc_kgCO2_per_kgH2"])
    ci_biomass_wccs = float(params["CI_biomass_wccs_kgCO2_per_kgH2"])
    lhv = float(params["H2_LHV_MJ_per_kg"])
    usd_factor = float(params["USD2010_to_USD2022_factor_CPIU"])

    drivers["CI_h2_electricity_kgCO2_per_kgH2"] = drivers["grid_EF_gCO2_per_kWh"] * electrolyzer_kWh / 1000.0
    drivers["CI_h2_fossil_wccs_kgCO2_per_kgH2"] = ci_fossil_wccs
    drivers["CI_h2_fossil_wocc_kgCO2_per_kgH2"] = ci_fossil_wocc
    drivers["CI_h2_biomass_wccs_kgCO2_per_kgH2"] = ci_biomass_wccs
    drivers["CI_h2_biomass_wocc_kgCO2_per_kgH2"] = ci_biomass_wocc

    drivers["H2_CI_weighted_kgCO2_per_kgH2"] = (
        drivers["H2_share_electricity_renorm"]*drivers["CI_h2_electricity_kgCO2_per_kgH2"] +
        drivers["H2_share_fossil_wccs_renorm"]*drivers["CI_h2_fossil_wccs_kgCO2_per_kgH2"] +
        drivers["H2_share_fossil_wocc_renorm"]*drivers["CI_h2_fossil_wocc_kgCO2_per_kgH2"] +
        drivers["H2_share_biomass_wccs_renorm"]*drivers["CI_h2_biomass_wccs_kgCO2_per_kgH2"] +
        drivers["H2_share_biomass_wocc_renorm"]*drivers["CI_h2_biomass_wocc_kgCO2_per_kgH2"]
    )

    drivers["H2_price_USD2010_per_kg"] = drivers["H2_price_USD2010_per_GJ"] * (lhv/1000.0)
    drivers["H2_price_USD2022_per_kg"] = drivers["H2_price_USD2010_per_kg"] * usd_factor

   
    if "electricity_price_USD2010_per_GJ" in drivers.columns:
        drivers["Electricity_price_USD2022_per_MJ"] = drivers["electricity_price_USD2010_per_GJ"].astype(float) * usd_factor / 1000.0
    drivers["Electricity_EF_override_gCO2_per_MJ"] = drivers["grid_EF_gCO2_per_MJ"]
    drivers["H2_EF_override_gCO2_per_g"] = drivers["H2_CI_weighted_kgCO2_per_kgH2"]

    drivers.to_csv(root/"ngfs_drivers_derived.csv", index=False)

   
    liq_long = melt_years(liq)
    liq_long = liq_long[liq_long["Variable"]=="Price|Final Energy|Transportation|Liquids"].copy()
    liq_long["fossil_liquids_price_USD2022_per_MJ"] = liq_long["Value"].astype(float) * usd_factor / 1000.0
    liq_keep = ["Model","Scenario","Region","Year","fossil_liquids_price_USD2022_per_MJ"]
    drivers2 = drivers.merge(liq_long[liq_keep], on=["Scenario","Region","Year"], how="left")
    drivers2.to_csv(root/"ngfs_drivers_with_fossil_price.csv", index=False)

    with open(root/"default_h2_parameters.json","w",encoding="utf-8") as f:
        json.dump({"parameters": params}, f, ensure_ascii=False, indent=2)

    print("Wrote ngfs_drivers_derived.csv and ngfs_drivers_with_fossil_price.csv")

if __name__ == "__main__":
    main()
