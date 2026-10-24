"""Run the analysis pipeline in dependency order.

Usage:
    python run_all.py --check   # verify inputs only
    python run_all.py           # full run

Steps whose inputs are optional are skipped when those inputs are absent; the
committed reference outputs in data/data03 and data/tables03 remain usable.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"


def has(*paths: str) -> bool:
    return all((ROOT / p).exists() for p in paths)


def n_files(pattern: str) -> int:
    return len(list(ROOT.glob(pattern)))


def prerequisites() -> dict[str, bool]:
    return {
        "core": has("data/ERA5_MJO_1950-2024.csv",
                    "data/typhoon_output/landfall_typhoons_info.csv",
                    "data/typhoon_output/landfall_typhoons_tracks.csv",
                    "data/shapefiles/china_country.shp"),
        "pre": n_files("data/pre/pre_*.nc") > 0,
        "ncar": n_files("data/Data-NCAR/daily/*.nc") > 0,
        "ibtracs": has("data/IBTrACS/IBTrACS.ALL.v04r01.nc"),
    }


# (script, required prerequisites)
STEPS = [
    ("tests_stats_yb.py", []),
    ("00_build_event_table.py", ["pre"]),
    ("r2_inference.py", []),
    ("01_tables.py", []),
    ("02_fig01.py", []),
    ("fig02_landfall_maps.py", []),
    ("03_fig03_regional.py", []),
    ("04_fig04_tcp_intensity.py", ["pre"]),
    ("05_fig05_phase_decomp.py", []),
    ("06_fig06_genesis_track.py", []),
    ("07_fig07_genesis_mjo.py", []),
    ("08_fig08_genesis_landfall.py", []),
    ("09r_composite_jjaso.py", ["ncar"]),
    ("10r_fig09_circulation.py", ["ncar"]),
    ("11_ibtracs_robustness.py", ["ibtracs"]),
    ("figS1_monthly.py", []),
    ("figS2_landfall_intensity.py", []),
    ("figS4_coastal_sensitivity.py", ["pre"]),
    ("make_fig_doc_data.py", ["pre"]),
    ("sensitivity_daylevel.py", []),
    ("make_numbers_report.py", []),
]


def report(flags: dict[str, bool]) -> None:
    print("Inputs")
    print(f"  core tables and index : {'ok' if flags['core'] else 'MISSING'}")
    print(f"  data/pre              : {'ok' if flags['pre'] else 'missing (00/04/figS4/make_fig_doc_data skipped)'}")
    print(f"  data/Data-NCAR/daily  : {'ok' if flags['ncar'] else 'missing (09r/10r skipped)'}")
    print(f"  data/IBTrACS          : {'ok' if flags['ibtracs'] else 'missing (11 skipped)'}")


def main() -> int:
    flags = prerequisites()
    report(flags)
    if not flags["core"]:
        print("Core inputs are required. See data/*/README.md.")
        return 1
    if "--check" in sys.argv:
        return 0

    LOGS.mkdir(exist_ok=True)
    for script, needs in STEPS:
        if any(not flags[key] for key in needs):
            print(f"skip {script}")
            continue
        print(f"run  {script}", flush=True)
        log = LOGS / f"{script}.log"
        with log.open("w") as fh:
            proc = subprocess.run([sys.executable, "-u", script], cwd=ROOT,
                                  stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print(f"failed: {script} (see {log})")
            return proc.returncode
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
