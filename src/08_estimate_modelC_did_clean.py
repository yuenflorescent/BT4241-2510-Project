# src/08_estimate_modelC_did_clean.py
# Difference-in-Differences on cleaned Model C panel
# Outputs:
#   data_final/modelC_did_summary.txt
#   data_final/modelC_did.png  (event-time heterogeneity plot, optional)

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path

IN = Path("data_final")
OUT_SUMMARY = IN / "modelC_did_summary.txt"
OUT_PNG     = IN / "modelC_did.png"

def load_clean_modelC():
    pqt = IN / "panel_modelC_song_quarter_clean.parquet"
    csv = IN / "panel_modelC_song_quarter_clean.csv"
    if pqt.exists():
        return pd.read_parquet(pqt)
    elif csv.exists():
        return pd.read_csv(csv)
    else:
        raise FileNotFoundError("Clean Model C not found in data_final/")

def main():
    print("➡️ Loading CLEAN Model C panel…")
    df = load_clean_modelC()
    print("Rows x Cols:", df.shape)

    # Keep analysis window like Model A (optional)
    if "release_q" not in df.columns:
        raise ValueError("release_q is required.")
    df["release_q"] = pd.PeriodIndex(df["release_q"].astype(str), freq="Q").astype(str)

    # Ensure types
    for c in ["inverse_rank","energy","danceability","valence","tempo","duration_ms","artist_popularity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["HighEnergy","treat","did","recession_t"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)

    # Basic controls (subset if missing)
    possible_controls = ["energy","danceability","valence","tempo","duration_ms","artist_popularity"]
    controls = [c for c in possible_controls if c in df.columns]

    # Fixed effects: genre + calendar quarter
    df["genre"] = df.get("genre", "unknown").astype(str)
    df = df.dropna(subset=["inverse_rank","HighEnergy","treat","did","genre","release_q"]).copy()

    # Build DiD formula:
    #   inverse_rank ~ did + HighEnergy + treat + controls + FE_time + FE_genre
    # Note: FE absorb confounding; did is the key coefficient.
    rhs = ["did", "HighEnergy", "treat", "C(release_q)", "C(genre)"] + controls
    formula = "inverse_rank ~ " + " + ".join(rhs)
    print("Formula:\n ", formula)

    # Cluster by artist if available; else by release_q
    cluster_col = "artist_norm" if "artist_norm" in df.columns else "release_q"
    df[cluster_col] = df[cluster_col].astype(str)

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]}
    )

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    print("✅ Saved summary:", OUT_SUMMARY)

        # Optional: dynamic heterogeneity (event-time x group) plot
    # Visualize when the HighEnergy–LowEnergy gap opens.
    # Use ONLY the interaction (:) with time FE present to avoid collinearity.
    if "event_time_k" in df.columns:
        dfd = df[(df["event_time_k"] >= -6) & (df["event_time_k"] <= 8)].copy()
        dfd["event_time_k"] = pd.to_numeric(dfd["event_time_k"], errors="coerce").astype(int)
        dfd["HighEnergy"]   = pd.to_numeric(dfd["HighEnergy"], errors="coerce").astype(int)

        # Only the interaction with event-time (no standalone C(event_time_k) terms),
        # while keeping calendar-quarter FE and genre FE.
        ctrl_terms = " + ".join(controls) if controls else "0"
        dyn_formula = (
            "inverse_rank ~ "
            "C(event_time_k, Treatment(reference=-1)) : HighEnergy "
            "+ HighEnergy "
            "+ C(release_q) + C(genre) "
            + ("+ " + ctrl_terms if controls else "")
        )
        print("Dynamic DiD formula:\n ", dyn_formula)

        dyn_mod = smf.ols(dyn_formula, data=dfd).fit(
            cov_type="cluster",
            cov_kwds={"groups": dfd[cluster_col]}
        )

        # Select interaction coefficients robustly (no boolean-length mismatch)
        idx = dyn_mod.params.index
        sel = [name for name in idx if "C(event_time_k" in name and ":HighEnergy" in name]
        coefs = dyn_mod.params.loc[sel]
        ses   = dyn_mod.bse.loc[sel]

        # Parse k even if labels have decimals like [T.-6.0]
        rows = []
        for term, beta in coefs.items():
            # Example: 'C(event_time_k, Treatment(reference=-1))[T.-3.0]:HighEnergy'
            k_str = term.split("[T.")[1].split("]")[0]
            k = int(float(k_str))
            rows.append((k, beta, ses[term]))

        est = pd.DataFrame(rows, columns=["k", "beta", "se"]).sort_values("k")

        plt.figure(figsize=(8, 5))
        plt.axhline(0, linewidth=1)
        plt.errorbar(est["k"], est["beta"], yerr=1.96*est["se"], fmt="o-", capsize=4)
        plt.title("Model C: HighEnergy vs LowEnergy — event-time gap (interaction β_k)")
        plt.xlabel("Event time k (quarters, ref = k = -1)")
        plt.ylabel("HighEnergy − LowEnergy effect on inverse_rank")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=300)
        plt.close()
        print("✅ Saved dynamic DiD plot:", OUT_PNG)


if __name__ == "__main__":
    main()
