# src/06_build_and_clean_modelC_from_cleanA.py
# Rebuild Model C from the CLEANED Model A panel, then validate and save.
import pandas as pd
import numpy as np
from pathlib import Path

IN_A  = Path("data_final/panel_modelA_song_quarter_clean.parquet")  # use the cleaned A
OUT_C = Path("data_final/panel_modelC_song_quarter_clean.parquet")
OUT_C_CSV = Path("data_final/panel_modelC_song_quarter_clean.csv")
CUTS  = Path("data_final/modelC_cutoffs_clean.csv")

ESSENTIAL = ["artist_norm","track_norm","inverse_rank","recession_t","event_time_k","release_q","energy"]

def main():
    print("➡️ Loading CLEAN Model A panel...")
    try:
        a = pd.read_parquet(IN_A)
    except Exception as e:
        print("Parquet failed, trying CSV:", e)
        a = pd.read_csv(IN_A.with_suffix(".csv"))

    print("Rows x Cols (A clean):", a.shape)

    # Ensure types
    a["event_time_k"] = pd.to_numeric(a["event_time_k"], errors="coerce").astype("Int64")
    a["recession_t"]  = pd.to_numeric(a["recession_t"],  errors="coerce").astype("Int64")
    a["inverse_rank"] = pd.to_numeric(a["inverse_rank"], errors="coerce")
    a["energy"]       = pd.to_numeric(a["energy"],       errors="coerce")

    # Drop rows missing *essential* fields for Model C
    before = len(a)
    a = a.dropna(subset=[c for c in ESSENTIAL if c in a.columns])
    print(f"Dropped {before - len(a)} rows missing essentials for Model C.")

    # Build HighEnergy cutoff based on PRE period only (k < 0)
    pre_mask = a["event_time_k"] < 0
    energy_cut = a.loc[pre_mask, "energy"].quantile(0.70)
    a["HighEnergy"] = (a["energy"] >= energy_cut).astype(int)

    # Treatment and interaction
    a["treat"] = a["recession_t"].astype(int)
    a["did"]   = a["HighEnergy"] * a["treat"]

    # Minimal sanity checks
    assert set(a["HighEnergy"].dropna().unique()).issubset({0,1})
    assert set(a["treat"].dropna().unique()).issubset({0,1})
    assert set(a["did"].dropna().unique()).issubset({0,1})

    # Save outputs
    out = a.copy()
    out.to_parquet(OUT_C, index=False)
    out.to_csv(OUT_C_CSV, index=False)
    pd.DataFrame({"energy_cutoff_preperiod":[energy_cut]}).to_csv(CUTS, index=False)

    print("✅ Saved:", OUT_C)
    print("✅ Saved:", OUT_C_CSV)
    print("✅ Saved cutoff file:", CUTS)
    print("Rows x Cols (C clean):", out.shape)
    print(f"Energy cutoff (70th pct pre-period): {energy_cut:.3f}")

if __name__ == "__main__":
    main()

