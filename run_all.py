"""End-to-end runner for the major-revision analysis layer.

Runs every step in dependency order. The heavy steps are 00 (per-storm TCP),
09 (NCEP calendar-month composites), and 11 (IBTrACS match); the rest are seconds.
Run from the repo root:   python run_all.py
"""
import subprocess, sys

STEPS = [
    "00_build_event_table.py",        # event_table.csv (keystone)
    "01_tables.py",                   # Table 2 / 3 / S1
    "02_fig01.py",                    # revised Fig 1
    "fig02_landfall_maps.py",         # Fig 2
    "03_fig03_regional.py",           # Fig 3
    "04_fig04_tcp_intensity.py",      # revised Fig 4
    "05_fig05_phase_decomp.py",       # revised Fig 5
    "06_fig06_stormlevel.py",         # revised Fig 6
    "07_fig07_genesis_track.py",      # revised Fig 7
    "08_fig08_genesis_mjo.py",        # revised Fig 8
    "09_seasonal_composites.py",      # calendar-month composites (heavy)
    "10_fig09_dynamics.py",           # revised Fig 9 + S5-S8
    "11_ibtracs_robustness.py",       # Fig S3 (IBTrACS match, heavy)
    "12_all_wnp_compare.py",          # Fig S4
    "13_october_decision.py",         # October decision + Fig S10
    "figS1_monthly.py",               # Fig S1
    "figS2_landfall_intensity.py",    # Fig S2
    "figS9_coastal_sensitivity.py",   # Fig S9 (coastal/inland 100/200/300 km)
]


def run(script):
    print("\n" + "=" * 70 + f"\n>>> {script}\n" + "=" * 70)
    r = subprocess.run([sys.executable, "-u", script])
    if r.returncode != 0:
        raise SystemExit(f"{script} failed (exit {r.returncode})")


if __name__ == "__main__":
    for s in STEPS:
        run(s)
    print("\nAll done. Figures in fig/, tables in data/tables03/, data in data/data03/.")
