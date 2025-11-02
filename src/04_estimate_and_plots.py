# src/04_estimate_and_plots.py
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

IN = Path("data_final")
OUT = Path("data_final")

def main():
    # ======================
    # 1. Load data
    # ======================
    print("➡️ Loading panels...")
    a = pd.read_parquet(IN / "panel_modelA_song_quarter.parquet")
    c = pd.read_parquet(IN / "panel_modelC_song_quarter.parquet")
    print(f"Model A rows: {len(a):,}, Model C rows: {len(c):,}")

    # ======================
    # 2. MODEL A: Event Study
    # ======================
    print("➡️ Estimating Model A (event-study)...")
    # baseline event window -8 ≤ k ≤ +8, exclude k=-1 as reference
    dfA = a[(a["event_time_k"] >= -8) & (a["event_time_k"] <= 8)].copy()
    dfA["event_time_k"] = dfA["event_time_k"].astype(int)
    dfA["event_time_str"] = dfA["event_time_k"].astype(str)

    # Estimate regression: inverse_rank ~ C(event_time_k)
    modA = smf.ols("inverse_rank ~ C(event_time_k)", data=dfA).fit(cov_type="HC1")
    coefs = modA.params.filter(like="C(event_time_k)").reset_index()
    coefs.columns = ["term", "beta"]
    ses = modA.bse.filter(like="C(event_time_k)").reset_index()
    ses.columns = ["term", "se"]
    plot_df = coefs.merge(ses, on="term")
    plot_df["k"] = plot_df["term"].str.extract(r"(\-?\d+)").astype(int)

    # Plot β_k
    plt.figure(figsize=(8,5))
    plt.axhline(0, color="gray", lw=1)
    plt.errorbar(plot_df["k"], plot_df["beta"], yerr=1.96*plot_df["se"],
                 fmt="o-", capsize=4)
    plt.title("Model A: Event-Study Estimates (inverse_rank)")
    plt.xlabel("Event time k (quarters relative to 2007Q4)")
    plt.ylabel("Effect on inverse rank")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "modelA_eventstudy.png", dpi=300)
    print("✅ Model A plot saved as modelA_eventstudy.png")

    # ======================
    # 3. MODEL C: DiD
    # ======================
    print("➡️ Estimating Model C (DiD)...")
    modC = smf.ols("inverse_rank ~ treat + HighEnergy + did", data=c).fit(cov_type="HC1")
    print(modC.summary().tables[1])
    theta = modC.params["did"]
    se = modC.bse["did"]
    print(f"✅ DiD effect θ = {theta:.3f} (s.e. = {se:.3f})")

    # Bar plot for DiD
    plt.figure(figsize=(5,4))
    plt.bar(["DiD θ"], [theta], yerr=[1.96*se], color="steelblue", capsize=5)
    plt.title("Model C: Difference-in-Differences Effect")
    plt.ylabel("Effect on inverse rank")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "modelC_did.png", dpi=300)
    print("✅ Model C plot saved as modelC_did.png")

if __name__ == "__main__":
    main()
