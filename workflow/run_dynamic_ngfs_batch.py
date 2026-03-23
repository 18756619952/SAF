from __future__ import annotations
import argparse, shutil, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
import numpy as np

def find_one(root: Path, filename: str, prefer_dir_contains: str | None = None) -> Path:
    hits = list(root.rglob(filename))
    if len(hits) == 0:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    if prefer_dir_contains:
        preferred = [p for p in hits if prefer_dir_contains in p.as_posix()]
        if preferred:
            hits = preferred
    hits = sorted(hits, key=lambda p: len(p.parts))
    return hits[0]

def override_background_energy_ef(path: Path, elec_g_per_MJ: float, h2_kg_per_kg: float) -> None:
    df = pd.read_csv(path)
    if "energy_type" not in df.columns or "range_mean" not in df.columns:
        raise RuntimeError(f"{path} missing required columns. columns={list(df.columns)}")
    mask_e = df["energy_type"].astype(str).str.lower().eq("electricity")
    if mask_e.sum() != 1:
        raise RuntimeError(f"Expected exactly 1 Electricity row in {path}, found {mask_e.sum()}")
    df.loc[mask_e, "range_mean"] = float(elec_g_per_MJ)
    mask_h2 = df["energy_type"].astype(str).str.lower().isin(["h2","hydrogen"])
    if mask_h2.sum() != 1:
        raise RuntimeError(f"Expected exactly 1 H2 row in {path}, found {mask_h2.sum()}")
    df.loc[mask_h2, "range_mean"] = float(h2_kg_per_kg)
    df.to_csv(path, index=False)

def override_h2_price(path: Path, h2_price_usd2022_per_kg: float) -> None:
    """Override H2 price inside a TEA price table."""
    df = pd.read_csv(path)
    if "product" not in df.columns:
        raise RuntimeError(f"{path} missing 'product' column. columns={list(df.columns)}")

    # Be robust to different column names (some tables use e.g. price_usd_unit)
    preferred = ["price_USD2022","price_usd_2022","price_usd2022","price_usd_unit","price"]
    price_col = next((c for c in preferred if c in df.columns), None)
    if price_col is None:
        # fall back: first column containing 'price'
        price_like = [c for c in df.columns if "price" in str(c).lower()]
        if not price_like:
            raise RuntimeError(f"{path} missing price-like column. columns={list(df.columns)}")
        price_col = price_like[0]

    mask = df["product"].astype(str).str.lower().isin(["h2","hydrogen"])
    if mask.sum() < 1:
        raise RuntimeError(f"Could not find H2 rows in {path}")
    df.loc[mask, price_col] = float(h2_price_usd2022_per_kg)
    df.to_csv(path, index=False)


def override_electricity_price(path: Path, elec_price_usd2022_per_MJ: float) -> None:
    """Override Electricity price inside a TEA price table."""
    df = pd.read_csv(path)
    if "product" not in df.columns:
        raise RuntimeError(f"{path} missing 'product' column. columns={list(df.columns)}")

    preferred = ["price_USD2022","price_usd_2022","price_usd2022","price_usd_unit","price"]
    price_col = next((c for c in preferred if c in df.columns), None)
    if price_col is None:
        price_like = [c for c in df.columns if "price" in str(c).lower()]
        if not price_like:
            raise RuntimeError(f"{path} missing price-like column. columns={list(df.columns)}")
        price_col = price_like[0]

    mask = df["product"].astype(str).str.lower().isin(["electricity","electric power","power"])
    if mask.sum() < 1:
        raise RuntimeError(f"Could not find Electricity rows in {path}")
    df.loc[mask, price_col] = float(elec_price_usd2022_per_MJ)
    df.to_csv(path, index=False)

def run_one(base_model_folder: Path, row: pd.Series, outdir: Path) -> dict:
    scenario = str(row["Scenario"])
    region = str(row["Region"])
    year = int(row["Year"])
    tag = f"{scenario}__{region}__{year}".replace(" ","_").replace("|","_")
    work = Path(tempfile.mkdtemp(prefix=f"ngfs_{tag}_"))
    try:
        shutil.copytree(base_model_folder, work/"model", dirs_exist_ok=True)
        model = work/"model"
        energy_csv = find_one(model, "background_energy_EF.csv", prefer_dir_contains="/inputs/")
        price_csv  = find_one(model, "background_price_USD2022.csv", prefer_dir_contains="/inputs/tea/")

        override_background_energy_ef(
            energy_csv,
            float(row["Electricity_EF_override_gCO2_per_MJ"]),
            float(row["H2_EF_override_gCO2_per_g"]),
        )
        override_h2_price(price_csv, float(row["H2_price_USD2022_per_kg"]))

        
        if "Electricity_price_USD2022_per_MJ" in row.index and pd.notna(row["Electricity_price_USD2022_per_MJ"]):
            override_electricity_price(price_csv, float(row["Electricity_price_USD2022_per_MJ"]))

        cmd = [sys.executable, "run_coupled_LCA_TEA_author_aligned_v2.py"]
        p = subprocess.run(cmd, cwd=str(model), capture_output=True, text=True)

        (outdir/"logs").mkdir(parents=True, exist_ok=True)
        (outdir/"logs"/f"{tag}.stdout.txt").write_text(f"[INFO] energy_csv={energy_csv}\\n[INFO] price_csv={price_csv}\\n\\n"+p.stdout, encoding="utf-8")
        (outdir/"logs"/f"{tag}.stderr.txt").write_text(p.stderr, encoding="utf-8")

        if p.returncode != 0:
            return {"Scenario":scenario, "Region":region, "Year":year, "status":"ERROR", "error":"nonzero returncode"}

        results_root = model/"results"
        run_dirs = sorted([d for d in results_root.glob("run_*") if d.is_dir()])
        if not run_dirs:
            return {"Scenario":scenario, "Region":region, "Year":year, "status":"ERROR", "error":"no results dir"}
        latest = run_dirs[-1]

        ghg = pd.read_csv(latest/"results_total_ghg.csv")
        tea = pd.read_csv(latest/"tea_total_cost_usd_per_MJ.csv")
        for df in (ghg, tea):
            df["Scenario"] = scenario
            df["Region"] = region
            df["Year"] = year

        per_run = outdir/"per_run"/tag
        per_run.mkdir(parents=True, exist_ok=True)
        ghg.to_csv(per_run/"results_total_ghg.csv", index=False)
        tea.to_csv(per_run/"tea_total_cost_usd_per_MJ.csv", index=False)

        return {"Scenario":scenario, "Region":region, "Year":year, "status":"OK", "results_dir":str(latest)}
    finally:
        shutil.rmtree(work, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_folder", required=True)
    ap.add_argument("--drivers_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--years", nargs="+", type=int, default=[2025,2030,2035,2040,2045,2050])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--regions", nargs="*", default=None)
    args = ap.parse_args()

    base = Path(args.base_model_folder).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    drivers = pd.read_csv(args.drivers_csv)
    drivers = drivers[drivers["Year"].isin(args.years)].copy()
    if args.scenarios:
        drivers = drivers[drivers["Scenario"].isin(args.scenarios)].copy()
    if args.regions:
        drivers = drivers[drivers["Region"].isin(args.regions)].copy()

    required = ["Electricity_EF_override_gCO2_per_MJ","H2_EF_override_gCO2_per_g","H2_price_USD2022_per_kg"]
    if "Electricity_price_USD2022_per_MJ" in drivers.columns:
        required.append("Electricity_price_USD2022_per_MJ")
    drivers = drivers.dropna(subset=required)

    status_rows = []
    for _, row in drivers.iterrows():
        status_rows.append(run_one(base, row, outdir))

    pd.DataFrame(status_rows).to_csv(outdir/"batch_status.csv", index=False)

    ghg_all, tea_all = [], []
    for s in status_rows:
        if s.get("status") != "OK":
            continue
        tag = f"{s['Scenario']}__{s['Region']}__{s['Year']}".replace(" ","_").replace("|","_")
        per_run = outdir/"per_run"/tag
        if (per_run/"results_total_ghg.csv").exists():
            ghg_all.append(pd.read_csv(per_run/"results_total_ghg.csv"))
        if (per_run/"tea_total_cost_usd_per_MJ.csv").exists():
            tea_all.append(pd.read_csv(per_run/"tea_total_cost_usd_per_MJ.csv"))

    if ghg_all:
        pd.concat(ghg_all, ignore_index=True).to_csv(outdir/"ALL_results_total_ghg.csv", index=False)
    if tea_all:
        pd.concat(tea_all, ignore_index=True).to_csv(outdir/"ALL_tea_total_cost_usd_per_MJ.csv", index=False)

    print("Done. See:", outdir)

if __name__ == "__main__":
    main()
