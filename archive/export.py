"""P2-2 — Open Research archive export.

Freezes the reproducible-analysis package: the exact ERA5-derived MJO index file,
the event-level derived table, the figure-ready summary CSVs, and a manifest
documenting units, variable definitions, data sources, and the software
environment. Run after the pipeline has produced data/data03/ and data/tables03/.
"""
from __future__ import annotations
import sys, shutil, json, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import config as C

ARCHIVE = C.ARCHIVE_DIR / "package"
ARCHIVE.mkdir(parents=True, exist_ok=True)


def main():
    # 1. exact MJO index file (copy)
    shutil.copy(C.MJO_CSV, ARCHIVE / "ERA5_MJO_1950-2024.csv")

    # 2. event-level derived table + summaries
    for f in ["event_table.csv"]:
        src = C.DATA_DIR / f
        if src.exists():
            shutil.copy(src, ARCHIVE / f)
    for f in (C.TABLES_DIR).glob("*.csv"):
        shutil.copy(f, ARCHIVE / f"summary_{f.name}")

    # 3. figure-ready phase summary (one row per phase group x intensity)
    df = pd.read_csv(C.DATA_DIR / "event_table.csv")
    df = df[df['in_jjaso'] & df['active_landfall']]
    rows = []
    for g in C.GROUP_KEY:
        for cat in [0] + C.CATEGORY_ORDER:
            sub = df if cat == 0 else df[df['lmi_category'] == cat]
            sub = sub[sub['group_landfall'] == g]
            rows.append(dict(phase_group=g,
                             intensity=('All' if cat == 0 else C.CATEGORY_FULL[cat]),
                             n=len(sub),
                             total_TCP_1e6m3=round(float(sub['tcp_total'].sum()), 1),
                             mean_event_TCP_1e6m3=round(float(sub['tcp_total'].mean()), 1)
                             if len(sub) else None))
    pd.DataFrame(rows).to_csv(ARCHIVE / "figure_ready_phase_summary.csv", index=False)

    # 4. manifest
    manifest = {
        "title": "Paper-4-1 major-revision reproducible analysis package",
        "manuscript": "HYDROL-S-26-03282 (major revision)",
        "period": f"{C.YEAR_START}-{C.YEAR_END}",
        "season_months": C.SEASON_MONTHS,
        "alpha": C.ALPHA,
        "data_sources": {
            "CMA_Best_Track": "databank CMABSTdata (1949-2024), via src02/01_landfall.py",
            "MJO_index": "ERA5-derived RMM1/RMM2/phase/amplitude, 1950-2024 (exact file archived)",
            "precipitation": "CHM_PRE_V2 daily, 0.1 deg, 1960-2024 (databank/TPDC)",
            "reanalysis": "NCEP/NCAR daily uwnd/vwnd/hgt/slp, 17 levels (NOAA PSL)",
            "IBTrACS": "IBTrACS.ALL v04r01 (alternative best-track robustness)",
        },
        "variable_definitions": {
            "tcp_total": "event-total TCP over China land cells, area-weighted, 10^6 m^3",
            "tcp_depth": "area-weighted mean event depth over wet China cells, mm",
            "affected_area": "wet China cell area (>0.5 mm), km^2",
            "coastal/inland": "split at 200 km from coastline (china_country.shp)",
            "NDAYS_group": "active-MJO days (amplitude>=1) in season with phase in group",
            "rate_per_1000_phase_days": "storm count / NDAYS_group * 1000",
            "ratio_no_modulation": "rate / (total events / total active days); 1.0 = null",
            "vort_850": "850-hPa relative vorticity = dv/dx - du/dy, s^-1",
            "z_anomaly": "(X - Xbar_month)/sigma_month, calendar-month climatology 1960-2024",
        },
        "significance": "two-sided alpha=0.10; phase tests use residence-time permutation null; "
                        "field composites use one-sample t=composite*sqrt(n), |t|>1.645",
        "environment": {"python": platform.python_version(),
                        "pandas": pd.__version__,
                        "platform": platform.platform()},
    }
    (ARCHIVE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"archived package -> {ARCHIVE}")
    print("files:", sorted(p.name for p in ARCHIVE.iterdir()))


if __name__ == "__main__":
    main()
