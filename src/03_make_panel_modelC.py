# src/03_make_panel_modelC.py
import pandas as pd
from pathlib import Path

IN  = Path("data_final")
OUT = Path("data_final")

def main():
    print("➡️ Loading Model A panel...")
    df = pd.read_parquet(IN / "panel_modelA_song_quarter.parquet")

    # --- Define the treatment group based on energy ---
    # Pre-period (before recession starts)
    pre = df["event_time_k"] < 0

    # 70th percentile of 'energy' in pre-period
    energy_cut = df.loc[pre, "energy"].dropna().quantile(0.70)
    df["HighEnergy"] = (df["energy"] >= energy_cut).astype(int)

    # --- Standard DiD variables ---
    df["treat"] = df["recession_t"].astype(int)
    df["did"] = df["HighEnergy"] * df["treat"]

    # --- Save outputs ---
    pd.DataFrame({"energy_cutoff_preperiod": [energy_cut]}).to_csv(
        OUT / "modelC_cutoffs.csv", index=False
    )
    df.to_parquet(OUT / "panel_modelC_song_quarter.parquet", index=False)
    df.to_csv(OUT / "panel_modelC_song_quarter.csv", index=False)

    print("✅ Saved:", OUT / "panel_modelC_song_quarter.parquet")
    print(f"   Energy cutoff (70th percentile pre-period): {energy_cut:.3f}")

if __name__ == "__main__":
    main()
