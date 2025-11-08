# src/09_modelC_robustness.py
# -------------------------------------------------------------------
# Model C robustness suite:
#  1) Continuous heterogeneity: inverse_rank ~ treat*energy + FE + controls
#  2) Binary cutoffs (60/70/80 pct pre-period): did = HighEnergy * treat
#  3) Top-10 LPM (binary outcome): top10 ~ treat*energy (and did for cutoffs)
#  4) Compact comparison table with coef / se / p / N / R2
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf

IN  = Path("data_final")
OUT = IN / "modelC_robustness_summary.csv"

# --------- Helpers ---------
def _load_panelC_clean():
    """Load cleaned Model C panel (parquet preferred, fallback to CSV)."""
    pqt = IN / "panel_modelC_song_quarter_clean.parquet"
    csv = IN / "panel_modelC_song_quarter_clean.csv"
    if pqt.exists():
        df = pd.read_parquet(pqt)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError("Clean Model C not found in data_final/")
    return df

def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce key columns to proper dtypes."""
    # time FE
    if "release_q" not in df.columns:
        raise ValueError("release_q is required for time fixed effects.")
    df["release_q"] = pd.PeriodIndex(df["release_q"].astype(str), freq="Q").astype(str)

    # basic numerics
    for c in ["inverse_rank","energy","danceability","valence","tempo","duration_ms","artist_popularity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # binarys/int
    for c in ["HighEnergy","treat","did","recession_t","top10"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # outcome fallbacks
    if "top10" not in df.columns:
        # If 'top10' missing, construct from inverse_rank if available (Top10 => peak rank <= 10)
        # inverse_rank = 101 - peak_position  => Top10 ⇔ inverse_rank >= 91
        if "inverse_rank" in df.columns:
            df["top10"] = (df["inverse_rank"] >= 91).astype(int)
        else:
            raise ValueError("Neither 'top10' nor 'inverse_rank' is available to construct Top-10 outcome.")

    # genre FE presence
    df["genre"] = df.get("genre", "unknown").astype(str)

    # cluster id: prefer artist, else quarter
    cluster_col = "artist_norm" if "artist_norm" in df.columns else "release_q"
    df[cluster_col] = df[cluster_col].astype(str)
    return df, cluster_col

def _controls(df):
    poss = ["energy","danceability","valence","tempo","duration_ms","artist_popularity"]
    return [c for c in poss if c in df.columns]

def _fit_ols(formula, df, cluster_col):
    """OLS with cluster-robust SEs."""
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_col]}
    )
    return model

def _pick(df, cols):
    return [c for c in cols if c in df.columns]

def _add_row(rows, spec, coef_name, model, n=None):
    params = model.params
    bse    = model.bse
    pvals  = model.pvalues
    if coef_name not in params.index:
        rows.append(dict(
            spec=spec, term=coef_name, coef=np.nan, se=np.nan, p=np.nan,
            N=n if n is not None else int(model.nobs),
            R2=float(getattr(model, "rsquared", np.nan))
        ))
        return
    rows.append(dict(
        spec=spec,
        term=coef_name,
        coef=float(params[coef_name]),
        se=float(bse[coef_name]),
        p=float(pvals[coef_name]),
        N=n if n is not None else int(model.nobs),
        R2=float(getattr(model, "rsquared", np.nan))
    ))

# --------- Main robustness suite ---------
def main():
    print("➡️ Loading cleaned Model C panel…")
    df = _load_panelC_clean()
    df, cluster_col = _ensure_types(df)
    controls = _controls(df)

    # Common FE terms
    FE = "C(release_q) + C(genre)"

    rows = []

    # ---------------------------------------------------
    # (1) Continuous heterogeneity: treat * energy
    # ---------------------------------------------------
    # inverse_rank outcome
    rhs = "treat * energy + " + FE
    if controls:
        # Add other controls but avoid duplicate 'energy'
        extra = [c for c in controls if c != "energy"]
        if extra:
            rhs += " + " + " + ".join(extra)
    f1 = "inverse_rank ~ " + rhs
    print("\n[1] Continuous interaction (inverse_rank):\n ", f1)
    m1 = _fit_ols(f1, df.dropna(subset=["inverse_rank","treat","energy"]), cluster_col)
    _add_row(rows, "cont:inverse_rank", "treat:energy", m1)

    # top10 LPM
    f1b = "top10 ~ " + rhs
    print("\n[1b] Continuous interaction (top10 LPM):\n ", f1b)
    m1b = _fit_ols(f1b, df.dropna(subset=["top10","treat","energy"]), cluster_col)
    _add_row(rows, "cont:top10", "treat:energy", m1b)

    # ---------------------------------------------------
    # (2) Binary cutoffs: 60/70/80th pre-period
    # ---------------------------------------------------
    if "event_time_k" not in df.columns:
        raise ValueError("event_time_k is required to define pre-period cutoffs.")
    df["event_time_k"] = pd.to_numeric(df["event_time_k"], errors="coerce")

    for q in (0.60, 0.70, 0.80):
        sub = df.copy()
        pre_mask = sub["event_time_k"] < 0
        cut = sub.loc[pre_mask, "energy"].quantile(q)
        sub["HighEnergy_q"] = (sub["energy"] >= cut).astype(int)
        sub["did_q"] = (sub["HighEnergy_q"] * sub["treat"].fillna(0)).astype(int)

        # inverse_rank outcome
        rhs_q = "did_q + HighEnergy_q + treat + " + FE
        # Controls (avoid duplicate energy as level control if you prefer)
        extra = [c for c in controls]  # keep full control set here
        if extra:
            rhs_q += " + " + " + ".join(extra)
        f_q = "inverse_rank ~ " + rhs_q
        print(f"\n[2] Cutoff q={int(q*100)} (inverse_rank):\n ", f_q)
        m_q = _fit_ols(f_q, sub.dropna(subset=["inverse_rank","did_q","HighEnergy_q","treat"]), cluster_col)
        _add_row(rows, f"cut{int(q*100)}:inverse_rank", "did_q", m_q)

        # top10 LPM
        f_qb = "top10 ~ " + rhs_q
        print(f"\n[2b] Cutoff q={int(q*100)} (top10 LPM):\n ", f_qb)
        m_qb = _fit_ols(f_qb, sub.dropna(subset=["top10","did_q","HighEnergy_q","treat"]), cluster_col)
        _add_row(rows, f"cut{int(q*100)}:top10", "did_q", m_qb)

    # ---------------------------------------------------
    # Collect & save results
    # ---------------------------------------------------
    res = pd.DataFrame(rows, columns=["spec","term","coef","se","p","N","R2"])
    res = res.sort_values(["spec"]).reset_index(drop=True)
    print("\n================ ROBUSTNESS SUMMARY ================\n")
    print(res.to_string(index=False))
    res.to_csv(OUT, index=False)
    print(f"\n✅ Saved: {OUT}")

if __name__ == "__main__":
    main()
