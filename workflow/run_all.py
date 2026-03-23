import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_and_log(cmd: List[str], log_path: Path) -> None:
    """
    Run a command and write both stdout+stderr into log_path.
    Also stream output to console.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)

        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, cmd)


def try_git_info(repo_dir: Path) -> Dict[str, Any]:
    """
    Best-effort git info, SILENT if not a git repo.
    """
    info: Dict[str, Any] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,  
        ).strip()
        info["git_commit"] = commit

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        info["git_dirty"] = bool(status.strip())
    except Exception:
        pass
    return info


def safe_symlink(latest_link: Path, target_dir: Path) -> None:
    """
    Create/replace a symlink latest_link -> target_dir, robustly.

    - DO NOT resolve() latest_link before passing here.
    - If latest_link is an existing real directory (not a symlink), rename it as a backup.
    - If latest_link is an existing file or symlink (including broken symlink), remove it.
    - Then create the symlink.
    """
    latest_link = Path(latest_link)
    target_dir = Path(target_dir)

    if latest_link.exists() or latest_link.is_symlink():
        
        if latest_link.is_dir() and not latest_link.is_symlink():
            backup = latest_link.with_name(f"{latest_link.name}_backup_{now_ts()}")
            k = 1
            while backup.exists() or backup.is_symlink():
                backup = latest_link.with_name(f"{latest_link.name}_backup_{now_ts()}_{k}")
                k += 1
            latest_link.rename(backup)
        else:
            
            latest_link.unlink()

    latest_link.symlink_to(target_dir, target_is_directory=True)


def pick_one(candidates: List[Path], label: str) -> Path:
    """
    Choose one candidate deterministically.
    Prefer the most recently modified file.
    """
    if not candidates:
        raise FileNotFoundError(f"Cannot find required {label} file in outdir.")
    if len(candidates) == 1:
        return candidates[0]
    candidates_sorted = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates_sorted[0]


def find_ghg_and_tea(outdir: Path) -> tuple[Path, Path]:
    """
    Auto-detect GHG and TEA summary CSVs produced by the dynamic batch.
    We'll search typical filenames/patterns inside outdir.
    """
    ghg_patterns = [
        "ALL*ghg*.csv",
        "ALL*GHG*.csv",
        "ALL*emission*.csv",
        "ALL*results*ghg*.csv",
        "*ghg*.csv",
    ]
    tea_patterns = [
        "ALL*tea*.csv",
        "ALL*TEA*.csv",
        "ALL*cost*.csv",
        "ALL*total_cost*.csv",
        "*tea*.csv",
    ]

    ghg_candidates: List[Path] = []
    tea_candidates: List[Path] = []

    for pat in ghg_patterns:
        ghg_candidates.extend([p for p in outdir.glob(pat) if p.is_file()])

    for pat in tea_patterns:
        tea_candidates.extend([p for p in outdir.glob(pat) if p.is_file()])

    ghg_candidates = [p for p in ghg_candidates if "policy_layer" not in str(p)]
    tea_candidates = [p for p in tea_candidates if "policy_layer" not in str(p)]

    ghg = pick_one(ghg_candidates, "GHG")
    tea = pick_one(tea_candidates, "TEA")

    return ghg.resolve(), tea.resolve()


def pick_4_scenarios_and_regions(policy_filtered_csv: Path) -> Tuple[List[str], List[str]]:
    """
    Read policy_filtered_metrics.csv and pick 4 scenarios + 4 regions robustly.
    Prefer canonical NGFS names and EU-15/USA/China/Japan if present.
    """
    import pandas as pd

    df = pd.read_csv(policy_filtered_csv)
    df["Scenario"] = df["Scenario"].astype(str)
    df["Region"] = df["Region"].astype(str)

    scenarios_all = sorted(df["Scenario"].unique().tolist())
    regions_all = sorted(df["Region"].unique().tolist())

    preferred_scenarios = [
        "Current Policies",
        "Delayed transition",
        "Fragmented World",
        "Net Zero 2050",
    ]
    scenarios = [s for s in preferred_scenarios if s in scenarios_all]
    if len(scenarios) < 4:
        # fill up deterministically
        for s in scenarios_all:
            if s not in scenarios:
                scenarios.append(s)
            if len(scenarios) == 4:
                break
    scenarios = scenarios[:4]

    
    preferred_region_suffixes = ["EU-15", "USA", "China", "Japan"]
    regions = []
    for suf in preferred_region_suffixes:
        hit = [r for r in regions_all if r.endswith(suf)]
        if hit:
            # pick shortest match (usually the cleanest)
            regions.append(sorted(hit, key=len)[0])

    if len(regions) < 4:
        for r in regions_all:
            if r not in regions:
                regions.append(r)
            if len(regions) == 4:
                break
    regions = regions[:4]

    return scenarios, regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click pipeline: NGFS dynamic batch -> policy layer -> paper figures, with timestamped output folder."
    )

    parser.add_argument(
        "--base-model-folder",
        dest="base_model_folder",
        default=str(Path.home() / "LCASAF"),
        help="Base model folder for run_dynamic_ngfs_batch.py (default: ~/LCASAF)",
    )
    parser.add_argument(
        "--drivers-csv",
        dest="drivers_csv",
        default="ngfs_drivers_with_fossil_price.csv",
        help="Drivers CSV for run_dynamic_ngfs_batch.py (default: ./ngfs_drivers_with_fossil_price.csv)",
    )

    parser.add_argument(
        "--base-outdir",
        dest="base_outdir",
        default="results_ngfs_dynamic",
        help="Prefix for timestamped output folder (default: results_ngfs_dynamic)",
    )
    parser.add_argument(
        "--policy-config",
        dest="policy_config",
        default="policy_config.json",
        help="Policy config json for policy_layer_analyze.py (default: ./policy_config.json)",
    )

    parser.add_argument("--years", nargs="*", default=None, help="Years to run (e.g., 2025 2030 2040 2050)")
    parser.add_argument("--scenarios", nargs="*", default=None, help='Scenarios list (quote names with spaces)')
    parser.add_argument("--regions", nargs="*", default=None, help="Regions list (e.g., EU-15 China USA)")

    parser.add_argument(
        "--dynamic-script",
        dest="dynamic_script",
        default="run_dynamic_ngfs_batch.py",
        help="Dynamic batch script filename (default: run_dynamic_ngfs_batch.py)",
    )
    parser.add_argument(
        "--policy-script",
        dest="policy_script",
        default="policy_layer_analyze.py",
        help="Policy layer script filename (default: policy_layer_analyze.py)",
    )

    # Paper figures
    parser.add_argument(
        "--no-paper-figures",
        action="store_true",
        help="Disable paper-figure generation step.",
    )
    parser.add_argument(
        "--no-chapter-figures",
        action="store_true",
        help="Disable chapter-figure generation inside make_paper_figures.py.",
    )

    # Extra figures (raincloud / scenario-marker variants, etc.)
    parser.add_argument(
        "--no-extra-figures",
        action="store_true",
        help="Disable generation of extra figure variants (raincloud, scenario-marker versions).",
    )
    parser.add_argument(
        "--extra-figures-script",
        dest="extra_figures_script",
        default="make_extra_figures.py",
        help="Script to generate extra figure variants (default: make_extra_figures.py next to run_all.py)",
    )
    parser.add_argument("--pareto-year", default="2050")
    parser.add_argument("--space-year", default="2050")

    args = parser.parse_args()

    workdir = Path.cwd()
    dynamic_script = (workdir / args.dynamic_script).resolve()
    policy_script = (workdir / args.policy_script).resolve()

    base_model_folder = Path(args.base_model_folder).expanduser().resolve()
    drivers_csv = Path(args.drivers_csv).expanduser().resolve()
    policy_config = Path(args.policy_config).expanduser().resolve()

    if not dynamic_script.exists():
        raise SystemExit(f"Dynamic script not found: {dynamic_script}")
    if not policy_script.exists():
        raise SystemExit(f"Policy script not found: {policy_script}")
    if not base_model_folder.exists():
        raise SystemExit(f"base_model_folder not found: {base_model_folder}")
    if not drivers_csv.exists():
        raise SystemExit(f"drivers_csv not found: {drivers_csv}")
    if not policy_config.exists():
        raise SystemExit(f"policy_config not found: {policy_config}")

    ts = now_ts()
    outdir = (workdir / f"{args.base_outdir}_{ts}").resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    latest_link = (workdir / f"{args.base_outdir}_latest")

    meta: Dict[str, Any] = {
        "timestamp": ts,
        "cwd": str(workdir),
        "python": sys.executable,
        "base_model_folder": str(base_model_folder),
        "drivers_csv": str(drivers_csv),
        "policy_config": str(policy_config),
        "outdir": str(outdir),
        "dynamic_script": str(dynamic_script),
        "policy_script": str(policy_script),
        "years": args.years,
        "scenarios": args.scenarios,
        "regions": args.regions,
        "paper_figures_enabled": (not args.no_paper_figures),
        "pareto_year": args.pareto_year,
        "space_year": args.space_year,
        **try_git_info(workdir),
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"==> Output dir: {outdir}")

    # [1/3] dynamic
    cmd1: List[str] = [
        sys.executable,
        str(dynamic_script),
        "--base_model_folder",
        str(base_model_folder),
        "--drivers_csv",
        str(drivers_csv),
        "--outdir",
        str(outdir),
    ]
    if args.years:
        cmd1 += ["--years", *args.years]
    if args.scenarios:
        cmd1 += ["--scenarios", *args.scenarios]
    if args.regions:
        cmd1 += ["--regions", *args.regions]

    print("\n[1/3] Running dynamic NGFS batch...")
    run_and_log(cmd1, outdir / "logs" / "01_dynamic_batch.log")

    # [2/3] policy layer
    print("\n[2/3] Running policy layer...")

    try:
        ghg_csv, tea_csv = find_ghg_and_tea(outdir)
    except Exception as e:
        files = sorted([p.name for p in outdir.glob("*.csv")])
        raise SystemExit(
            "Policy layer step needs GHG and TEA summary CSVs, but auto-detection failed.\n"
            f"Reason: {e}\n"
            f"CSV files currently in outdir:\n- " + "\n- ".join(files) + "\n\n"
            "Fix: confirm the dynamic batch produced ALL_* summary CSVs, or adjust the filename patterns in find_ghg_and_tea()."
        )

    policy_outdir = (outdir / "policy_layer").resolve()
    policy_outdir.mkdir(parents=True, exist_ok=True)

    cmd2: List[str] = [
        sys.executable,
        str(policy_script),
        "--drivers",
        str(drivers_csv),
        "--ghg",
        str(ghg_csv),
        "--tea",
        str(tea_csv),
        "--config",
        str(policy_config),
        "--outdir",
        str(policy_outdir),
    ]
    run_and_log(cmd2, outdir / "logs" / "02_policy_layer.log")

    # [3/3] paper figures
    if not args.no_paper_figures:
        print("\n[3/3] Generating paper figures...")
        make_fig_script = (workdir / "make_paper_figures.py").resolve()
        if not make_fig_script.exists():
            raise SystemExit(f"make_paper_figures.py not found next to run_all.py: {make_fig_script}")

        policy_filtered_csv = policy_outdir / "policy_filtered_metrics.csv"
        if not policy_filtered_csv.exists():
            raise SystemExit(f"Missing: {policy_filtered_csv}")

        scenarios4, regions4 = pick_4_scenarios_and_regions(policy_filtered_csv)
        print("Using scenarios:", scenarios4)
        print("Using regions:", regions4)

        cmd3: List[str] = [
            sys.executable,
            str(make_fig_script),
            "--policy-outdir",
            str(policy_outdir),
            "--scenarios",
            *scenarios4,
            "--regions",
            *regions4,
            "--year-min",
            "2025",
            "--year-max",
            "2050",
            "--pareto-year",
            str(args.pareto_year),
            "--space-year",
            str(args.space_year),
            "--heatmap-source",
            "both",
        ]
        if args.no_chapter_figures:
            cmd3 += ["--no-chapter-figures"]
        run_and_log(cmd3, outdir / "logs" / "03_paper_figures.log")

        print(f"Paper figures saved to: {policy_outdir / 'paper_figures'}")


        # [4/4] extra figure variants (optional)
        if not args.no_extra_figures:
            print("\n[4/4] Generating extra figure variants...")
            extra_script = (workdir / args.extra_figures_script).resolve()
            if not extra_script.exists():
                print(f"⚠️  Extra figures skipped: script not found: {extra_script}")
            else:
                cmd4: List[str] = [
                    sys.executable,
                    str(extra_script),
                    "--policy-outdir",
                    str(policy_outdir),
                    "--scenarios",
                    *scenarios4,
                    "--regions",
                    *regions4,
                ]
                run_and_log(cmd4, outdir / "logs" / "04_extra_figures.log")

    # Update latest only if pipeline succeeded
    safe_symlink(latest_link, outdir)

    print("\n✅ DONE.")
    print(f"Latest link: {latest_link} -> {outdir}")
    print(f"Policy layer outputs: {policy_outdir}")
    print(f"Key file (expected): {policy_outdir / 'policy_filtered_metrics.csv'}")


if __name__ == "__main__":
    main()
