# src/07_estimate_modelA_eventstudy_clean.py
# ================================================================
# EVENT STUDY (INTERRUPTED TIME SERIES) ON THE CLEANED MODEL A PANEL
#
# Main outputs:
#   data_final/modelA_eventstudy_clean_summary.txt
#   data_final/modelA_eventstudy_clean.png
#
# Robustness outputs:
#   *_refm2_summary.txt / *_refm2.png     (ref k = -2)
#   *_window6_summary.txt / *_window6.png (window [-6,+6])
#   *_nocontrols_summary.txt / *_nocontrols.png (no controls)
#
# ================================================================
# ⚙️ WHY WE REMOVE C(release_q):
# In a single-event design (same macro shock for all), event_time_k 
# is perfectly aligned with release_q (calendar quarters).
# Including both causes near-perfect collinearity.
# We KEEP event_time_k (the event variable of interest)
# and DROP C(release_q) to stabilize estimation.
#
# ================================================================
# 🧩 ADJUSTMENTS TO REDUCE MULTICOLLINEARITY:
# 1. Collapse sparse genre×year cells (rare categories cause singularity)
# 2. Center event_time_k (reduces intercept correlation)
# 3. Optionally trim extreme quarters ([-6,+8]) for numerical stability
# ================================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

IN = Path("data_final")

# --------------------------
# Load Clean Panel
# --------------------------
def load_clean_modelA():
    pqt = IN / "panel_modelA_song_quarter_clean.parquet"
    csv = IN / "panel_modelA_song_quarter_clean.csv"
    if pqt.exists():
        return pd.read_parquet(pqt)
    elif csv.exists():
        return pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            "Clean Model A not found. Expected *_clean.parquet or *_clean.csv in data_final/"
        )

# --------------------------
# Helper: convert 'YYYYQn' to sortable number (e.g. 2008Q2 -> 200802)
# --------------------------
def qnum(qstr: str) -> int:
    qstr = str(qstr)
    y = int(qstr[:4]); q = int(qstr[-1])
    return y * 100 + q

# --------------------------
# Fit regression and plot βₖ coefficients
# --------------------------
def fit_and_plot(
    df: pd.DataFrame,
    ref_k: int,
    controls: list,
    cluster_col: str,
    out_prefix: str,
):
    # Build formula WITHOUT quarter FE (see top note)
    base_terms = [
        f"C(event_time_k_centered, Treatment(reference={ref_k}))",  # βₖ with ref omitted
        "C(genre)"  # genre fixed effects
    ]
    base_terms += controls
    formula = "inverse_rank ~ " + " + ".join(base_terms)

    print("\nFormula:\n ", formula)
    print(f"➡️ Estimating with cluster-robust SEs by: {cluster_col}")

    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]}
    )

    # Save text summary
    out_summary = IN / f"{out_prefix}_summary.txt"
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())
    print("✅ Saved summary:", out_summary)

    # Extract βₖ for plotting
    coefs = model.params.filter(like="C(event_time_k_centered").copy()
    ses   = model.bse.filter(like="C(event_time_k_centered").copy()

    rows = []
    for term, beta in coefs.items():
    # Example term: 'C(event_time_k_centered, Treatment(reference=-1))[T.-6.0]'
        k_str = term.split("[T.")[1].rstrip("]")
        try:
            k = int(float(k_str))  # handles cases like '-6.0'
        except ValueError:
            print(f"⚠️ Warning: could not parse k from term '{term}'")
            continue
        rows.append((k, beta, ses[term]))

    est = pd.DataFrame(rows, columns=["k", "beta", "se"]).sort_values("k")

    # Plot event-time coefficients
    out_png = IN / f"{out_prefix}.png"
    plt.figure(figsize=(8, 5))
    plt.axhline(0, linewidth=1)
    plt.errorbar(est["k"], est["beta"], yerr=1.96*est["se"], fmt="o-", capsize=4)
    plt.title(f"Model A (Clean): Event-Study (ref k={ref_k})")
    plt.xlabel(f"Event time k (quarters, ref = {ref_k})")
    plt.ylabel("Effect on inverse rank")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print("✅ Saved plot:", out_png)

    # Quick console diagnostics
    pre = est[est["k"] < 0]["beta"].abs().mean()
    post = est[est["k"] >= 0]["beta"].abs().mean()
    print(f"Avg |beta_k| pre: {pre:.3f} | post: {post:.3f}")

# --------------------------
# MAIN SCRIPT
# --------------------------
def main():
    print("➡️ Loading CLEAN Model A panel…")
    df = load_clean_modelA()
    print("Rows x Cols:", df.shape)

    # --------------------------
    # 1️⃣ Restrict to 2005Q1–2011Q4 analysis window
    # --------------------------
    if "release_q" not in df.columns:
        raise ValueError("release_q is required for filtering and FE.")
    df["release_q"] = pd.PeriodIndex(df["release_q"].astype(str), freq="Q").astype(str)
    df["rq_num"] = df["release_q"].map(qnum)
    lo, hi = qnum("2005Q1"), qnum("2011Q4")
    df = df[(df["rq_num"] >= lo) & (df["rq_num"] <= hi)].copy()

    # --------------------------
    # 2️⃣ Feature Engineering
    # --------------------------
    df["release_year"] = df["release_q"].str.slice(0, 4)

    # Controls (only those that exist)
    possible_controls = [
        "energy", "danceability", "valence", "tempo",
        "duration_ms", "artist_popularity", "explicit"
    ]
    controls = [c for c in possible_controls if c in df.columns]

    # Create genre×year FE
    df["genre_year"] = (
        df.get("genre", "NA").astype(str) + "×" + df["release_year"].astype(str)
    )

    # Ensure numeric types for main variables
    df["inverse_rank"] = pd.to_numeric(df["inverse_rank"], errors="coerce")
    df["event_time_k"] = pd.to_numeric(df["event_time_k"], errors="coerce")
    df = df.dropna(subset=["inverse_rank", "event_time_k"]).copy()
    df["event_time_k"] = df["event_time_k"].astype(int)

    # --------------------------
    # 3️⃣ Mitigation: Collapse sparse genre×year cells
    # --------------------------
    print("\n🔧 Checking for sparse genre×year cells...")
    ct = df["genre_year"].value_counts()
    rare = ct[ct < 5].index  # threshold: fewer than 5 songs
    if len(rare):
        print(f"Collapsing {len(rare)} rare genre×year groups into 'Other×year'...")
        df.loc[df["genre_year"].isin(rare), "genre_year"] = (
            df["release_year"].astype(str) + "×Other"
        )
    print("Remaining unique genre×year:", df["genre_year"].nunique())

    # --------------------------
    # 4️⃣ Mitigation: Center event_time_k
    # --------------------------
    df["event_time_k_centered"] = df["event_time_k"] - df["event_time_k"].mean().round()

    # Optionally trim extreme quarters (comment out if not needed)
    df = df[(df["event_time_k"] >= -6) & (df["event_time_k"] <= 8)].copy()

    # --------------------------
    # 5️⃣ Check for multicollinearity among numeric controls (VIF)
    # --------------------------
    controls_to_check = [
        c for c in ["energy","danceability","valence","tempo","duration_ms","artist_popularity"]
        if c in df.columns
    ]
    if len(controls_to_check) >= 2:
        vif_df = df[controls_to_check].dropna().copy()
        if len(vif_df) >= 10:
            X = vif_df.assign(constant=1.0)
            vif_rows = []
            for i, col in enumerate(X.columns):
                if col == "constant":
                    continue
                vif_val = variance_inflation_factor(X.values, i)
                vif_rows.append((col, float(vif_val)))
            vif_table = pd.DataFrame(vif_rows, columns=["Variable","VIF"]).sort_values("VIF", ascending=False)
            out_vif = IN / "modelA_controls_vif.csv"
            vif_table.to_csv(out_vif, index=False)
            print("\n🧮 Variance Inflation Factors (controls):")
            print(vif_table.to_string(index=False))
            high = vif_table[vif_table["VIF"] > 10]
            if not high.empty:
                print("\n⚠️ High multicollinearity detected:")
                print(high.to_string(index=False))
            else:
                print("\n✅ No severe multicollinearity among numeric controls (VIF ≤ 10).")
        else:
            print("\n(Info) Not enough rows to compute VIF robustly.")
    else:
        print("\n(Info) Too few controls for VIF check.")

    # --------------------------
    # 6️⃣ Cluster ID setup
    # --------------------------
    cluster_col_main = "artist_norm" if "artist_norm" in df.columns else "release_q"
    df[cluster_col_main] = df[cluster_col_main].astype(str)

    # --------------------------
    # 7️⃣ Run main and robustness models
    # --------------------------
    # MAIN: ref k = -1
    fit_and_plot(
        df=df,
        ref_k=-1,
        controls=controls,
        cluster_col=cluster_col_main,
        out_prefix="modelA_eventstudy_clean"
    )

    # ROBUSTNESS 1: ref k = -2
    fit_and_plot(
        df=df,
        ref_k=-2,
        controls=controls,
        cluster_col=cluster_col_main,
        out_prefix="modelA_eventstudy_clean_refm2"
    )

    # ROBUSTNESS 2: window [-6, +6]
    df_w = df[(df["event_time_k"] >= -6) & (df["event_time_k"] <= 6)].copy()
    fit_and_plot(
        df=df_w,
        ref_k=-1,
        controls=controls,
        cluster_col=cluster_col_main,
        out_prefix="modelA_eventstudy_clean_window6"
    )

    # ROBUSTNESS 3: no controls, cluster by release_q
    cluster_col_alt = "release_q"
    df[cluster_col_alt] = df["release_q"].astype(str)
    fit_and_plot(
        df=df,
        ref_k=-1,
        controls=[],  # no controls
        cluster_col=cluster_col_alt,
        out_prefix="modelA_eventstudy_clean_nocontrols"
    )

if __name__ == "__main__":
    main()


