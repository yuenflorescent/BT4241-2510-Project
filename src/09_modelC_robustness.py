# src/09_modelC_robustness.py
# -------------------------------------------------------------------
# Model C robustness (memory-safe):
# - Absorb C(release_q) + C(genre) via two-way demeaning
# - Cluster-robust SEs
# - Deduplicate column lists to avoid "duplicate labels" errors
# - Keep pandas DataFrames in OLS so coef names are preserved
# -------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

IN  = Path("data_final")
OUT = IN / "modelC_robustness_summary.csv"

# --------------- IO ---------------
def _load_panelC_clean():
    pqt = IN / "panel_modelC_song_quarter_clean.parquet"
    csv = IN / "panel_modelC_song_quarter_clean.csv"
    if pqt.exists():
        return pd.read_parquet(pqt)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError("Clean Model C not found in data_final/")

# --------------- Prep ---------------
def _ensure_types(df: pd.DataFrame):
    # time FE
    if "release_q" not in df.columns:
        raise ValueError("release_q is required for time fixed effects.")
    df["release_q"] = pd.PeriodIndex(df["release_q"].astype(str), freq="Q").astype(str)

    # numerics
    for c in ["inverse_rank","energy","danceability","valence","tempo","duration_ms","artist_popularity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # binaries
    for c in ["HighEnergy","treat","did","recession_t","top10"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # outcome fallback
    if "top10" not in df.columns:
        if "inverse_rank" in df.columns:
            df["top10"] = (pd.to_numeric(df["inverse_rank"], errors="coerce") >= 91).astype(int)
        else:
            raise ValueError("Need 'top10' or 'inverse_rank'.")

    # genre FE presence
    df["genre"] = df.get("genre", "unknown").astype(str)

    # cluster id: prefer artist, else release_q
    cluster_col = "artist_norm" if "artist_norm" in df.columns else "release_q"
    df[cluster_col] = df[cluster_col].astype(str)

    return df, cluster_col

def _controls(df):
    poss = ["energy","danceability","valence","tempo","duration_ms","artist_popularity"]
    return [c for c in poss if c in df.columns]

# --------------- FE absorption (within) ---------------
def _twoway_demean(df_with_fe: pd.DataFrame, cols, fe1="release_q", fe2="genre"):
    """
    De-mean each column in `cols` by FE1 and FE2: x - x_FE1 - x_FE2 + x_overall
    Returns a DataFrame with the same column names (float32).
    """
    out = df_with_fe[cols].astype("float32").copy()
    overall = out.mean(axis=0)

    if fe1 not in df_with_fe.columns or fe2 not in df_with_fe.columns:
        raise ValueError("Both FE columns must exist for two-way demeaning.")

    m1 = out.groupby(df_with_fe[fe1]).transform("mean")
    m2 = out.groupby(df_with_fe[fe2]).transform("mean")
    out = out - m1 - m2 + overall
    return out

def _fit_absorb(y_df: pd.Series, X_df: pd.DataFrame, cluster):
    """
    y_df: pandas Series (demeaned)
    X_df: pandas DataFrame (demeaned) — keeps column names for statsmodels
    """
    model = sm.OLS(y_df, X_df, hasconst=False).fit(
        cov_type="cluster",
        cov_kwds={"groups": cluster}
    )
    return model

def _add_row(rows, spec, coef_name, model):
    params = model.params  # pandas Series (has .get and index names)
    bse    = model.bse
    pvals  = model.pvalues
    rows.append(dict(
        spec=spec,
        term=coef_name,
        coef=float(params.get(coef_name, np.nan)),
        se=float(bse.get(coef_name, np.nan)),
        p=float(pvals.get(coef_name, np.nan)),
        N=int(model.nobs),
        R2=float(getattr(model, "rsquared", np.nan))
    ))

# --------------- Main ---------------
def main():
    print("➡️ Loading cleaned Model C panel…")
    df = _load_panelC_clean()
    df, cluster_col = _ensure_types(df)
    controls = _controls(df)

    FE1, FE2 = "release_q", "genre"
    rows = []

    # ===================================================
    # (1) Continuous heterogeneity: treat * energy
    # ===================================================
    # Build the column set we need (deduplicated, order-preserving)
    need = ["treat", "energy", "inverse_rank", "top10", FE1, FE2, cluster_col] + controls
    need = list(dict.fromkeys(need))  # dedupe while preserving order
    d1 = df[need].copy()
    d1 = d1.loc[:, ~d1.columns.duplicated()].copy()  # extra safety

    # ensure numeric for interaction vars
    d1["treat"]  = pd.to_numeric(d1["treat"], errors="coerce")
    d1["energy"] = pd.to_numeric(d1["energy"], errors="coerce")

    d1 = d1.dropna(subset=["treat","energy"])
    d1["treat_energy"] = d1["treat"] * d1["energy"]

    rhs_cols = ["treat","energy","treat_energy"] + [c for c in controls if c != "energy"]
    rhs_cols = list(dict.fromkeys(rhs_cols))  # dedupe

    # a) inverse_rank
    d1_inv = d1.dropna(subset=["inverse_rank"]).copy()
    y = d1_inv["inverse_rank"].astype("float32")
    X = d1_inv[rhs_cols].astype("float32")

    # two-way demeaning, KEEP DATAFRAME
    XY = pd.concat([y.rename("inverse_rank"), X], axis=1)
    XY_dm = _twoway_demean(
        pd.concat([d1_inv[[FE1,FE2]], XY], axis=1),
        cols=XY.columns, fe1=FE1, fe2=FE2
    )
    y_dm = XY_dm["inverse_rank"]
    X_dm = XY_dm.drop(columns=["inverse_rank"])
    m1 = _fit_absorb(y_dm, X_dm, d1_inv[cluster_col])
    _add_row(rows, "cont:inverse_rank", "treat_energy", m1)

    # b) top10 LPM
    d1_top = d1.dropna(subset=["top10"]).copy()
    yb = d1_top["top10"].astype("float32")
    Xb = d1_top[rhs_cols].astype("float32")

    XYb = pd.concat([yb.rename("top10"), Xb], axis=1)
    XYb_dm = _twoway_demean(
        pd.concat([d1_top[[FE1,FE2]], XYb], axis=1),
        cols=XYb.columns, fe1=FE1, fe2=FE2
    )
    yb_dm = XYb_dm["top10"]
    Xb_dm = XYb_dm.drop(columns=["top10"])
    m1b = _fit_absorb(yb_dm, Xb_dm, d1_top[cluster_col])
    _add_row(rows, "cont:top10", "treat_energy", m1b)

    # ===================================================
    # (2) Binary cutoffs: 60/70/80 pre-period
    # ===================================================
    if "event_time_k" not in df.columns:
        raise ValueError("event_time_k is required to define pre-period cutoffs.")
    df["event_time_k"] = pd.to_numeric(df["event_time_k"], errors="coerce")

    for q in (0.60, 0.70, 0.80):
        sub_cols = ["energy","treat","inverse_rank","top10", FE1, FE2, cluster_col] + controls
        sub_cols = list(dict.fromkeys(sub_cols))
        sub = df[sub_cols].copy()
        sub = sub.loc[:, ~sub.columns.duplicated()].copy()

        sub["energy"] = pd.to_numeric(sub["energy"], errors="coerce")
        sub["treat"]  = pd.to_numeric(sub["treat"], errors="coerce")

        # pre-period cutoff computed on the same index
        pre_mask = df["event_time_k"] < 0
        cut = sub.loc[pre_mask, "energy"].quantile(q)

        sub["HighEnergy_q"] = (sub["energy"] >= cut).astype(int)
        sub["did_q"] = (sub["HighEnergy_q"] * sub["treat"]).astype(int)

        rhs_q = ["did_q", "HighEnergy_q", "treat"] + controls
        rhs_q = list(dict.fromkeys(rhs_q))

        # a) inverse_rank
        si = sub.dropna(subset=["inverse_rank","did_q","HighEnergy_q","treat"]).copy()
        Yi = si["inverse_rank"].astype("float32")
        Xi = si[rhs_q].astype("float32")
        XYi = pd.concat([Yi.rename("inverse_rank"), Xi], axis=1)
        XYi_dm = _twoway_demean(
            pd.concat([df.loc[si.index, [FE1,FE2]], XYi], axis=1),
            cols=XYi.columns, fe1=FE1, fe2=FE2
        )
        Yi_dm = XYi_dm["inverse_rank"]
        Xi_dm = XYi_dm.drop(columns=["inverse_rank"])
        m_q = _fit_absorb(Yi_dm, Xi_dm, df.loc[si.index, cluster_col])
        _add_row(rows, f"cut{int(q*100)}:inverse_rank", "did_q", m_q)

        # b) top10 LPM
        st = sub.dropna(subset=["top10","did_q","HighEnergy_q","treat"]).copy()
        Yt = st["top10"].astype("float32")
        Xt = st[rhs_q].astype("float32")
        XYt = pd.concat([Yt.rename("top10"), Xt], axis=1)
        XYt_dm = _twoway_demean(
            pd.concat([df.loc[st.index, [FE1,FE2]], XYt], axis=1),
            cols=XYt.columns, fe1=FE1, fe2=FE2
        )
        Yt_dm = XYt_dm["top10"]
        Xt_dm = XYt_dm.drop(columns=["top10"])
        m_qb = _fit_absorb(Yt_dm, Xt_dm, df.loc[st.index, cluster_col])
        _add_row(rows, f"cut{int(q*100)}:top10", "did_q", m_qb)

    # ===================================================
    # Save summary
    # ===================================================
    res = pd.DataFrame(rows, columns=["spec","term","coef","se","p","N","R2"]).sort_values("spec")
    print("\n================ ROBUSTNESS SUMMARY ================\n")
    print(res.to_string(index=False))
    res.to_csv(OUT, index=False)
    print(f"\n✅ Saved: {OUT.resolve()}")

if __name__ == "__main__":
    main()



