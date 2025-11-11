# src/05_clean_panel_modelA.py
# Clean & impute Model A panel with vectorized operations.
# Inputs : data_final/panel_modelA_song_quarter.parquet
# Outputs: data_final/panel_modelA_song_quarter_clean.parquet / .csv
#          data_final/panel_modelA_clean_audit.csv

import pandas as pd
import numpy as np
from pathlib import Path

INP    = Path("data_final/panel_modelA_song_quarter.parquet")
OUTP   = Path("data_final/panel_modelA_song_quarter_clean.parquet")
OUTCSV = Path("data_final/panel_modelA_song_quarter_clean.csv")
AUDIT  = Path("data_final/panel_modelA_clean_audit.csv")

NUM_01   = ["acousticness","danceability","energy","instrumentalness",
            "liveness","speechiness","valence"]
NUM_MISC = ["tempo","loudness","duration_ms","artist_popularity"]
CATEG    = ["genre","key","mode","time_signature"]
ESSENTIAL= ["artist_norm","track_norm","inverse_rank","peak_position",
            "recession_t","event_time_k","release_q"]

def mode_or_na(s: pd.Series):
    s = s.dropna()
    m = s.mode()
    return m.iat[0] if not m.empty else pd.NA

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Dates
    for col in ["first_week","last_week"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Numeric (safe coercion)
    for col in set(NUM_01 + NUM_MISC + ["inverse_rank","peak_position","weeks_on"]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Keep genre as string for grouping; others we’ll finalize later
    if "genre" in df.columns:
        df["genre"] = df["genre"].astype("string")
    return df

def clamp_and_sanitize(df: pd.DataFrame) -> pd.DataFrame:
    # Clamp [0,1] features
    for col in NUM_01:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    # Sanity on misc numerics
    if "tempo" in df.columns:
        df.loc[df["tempo"] <= 0, "tempo"] = np.nan
    if "loudness" in df.columns:
        # loudness is typically <= 0 dB
        df.loc[df["loudness"] > 0, "loudness"] = np.nan
    if "duration_ms" in df.columns:
        df.loc[df["duration_ms"] <= 0, "duration_ms"] = np.nan
    return df

def impute_numeric_by_genre(df: pd.DataFrame, cols: list) -> dict:
    """Vectorized imputation: genre-median, then global median."""
    imp_counts = {}
    # Ensure a grouping key
    if "genre" not in df.columns:
        df["genre"] = "unknown"
    df["genre"] = df["genre"].fillna("unknown").astype("string")

    # Precompute genre medians only for the needed cols
    med_by_genre = df.groupby("genre")[cols].median()

    for col in cols:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()

        # 1) fill from genre median (align using map on genre)
        if col in med_by_genre.columns:
            df[col] = df[col].fillna(df["genre"].map(med_by_genre[col]))

        # 2) global median fallback
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

        imp_counts[col] = before - df[col].isna().sum()
    return imp_counts

def impute_categorical_by_genre(df: pd.DataFrame, cols: list) -> dict:
    imp_counts = {}
    if "genre" not in df.columns:
        df["genre"] = "unknown"
    df["genre"] = df["genre"].fillna("unknown").astype("string")

    for col in cols:
        if col not in df.columns:
            continue
        before = df[col].isna().sum() if df[col].isna().any() else int((df[col] == "").sum())

        # mode per genre
        mode_map = (df.groupby("genre")[col]
                      .apply(lambda s: mode_or_na(s))
                      .to_dict())
        # fill genre-mode
        df[col] = df[col].astype("string")
        mask_missing = df[col].isna() | (df[col] == "")
        df.loc[mask_missing, col] = df.loc[mask_missing, "genre"].map(mode_map)

        # global mode fallback
        gm = df[col].dropna().mode()
        if not gm.empty:
            df[col] = df[col].fillna(gm.iat[0]).replace("", gm.iat[0])

        # finalize dtype
        try:
            if col in ["key","mode","time_signature"]:
                # keep as integers if possible
                df[col] = pd.to_numeric(df[col], errors="ignore")
            df[col] = df[col].astype("category")
        except Exception:
            pass

        after = df[col].isna().sum() if df[col].isna().any() else int((df[col] == "").sum())
        imp_counts[col] = max(0, before - after)
    return imp_counts

def drop_unusable(df: pd.DataFrame):
    essential = [c for c in ESSENTIAL if c in df.columns]
    before = len(df)
    df = df.dropna(subset=essential)
    return df, before - len(df), essential

def finalize_dtypes_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Avoid Int64Dtype issues in statsmodels; use plain ints/floats where needed."""
    for col in ["recession_t","event_time_k","top10"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    # keep FE as strings
    for col in ["release_q"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df

def main():
    print("➡️ Loading panel…")
    df = pd.read_parquet(INP)
    print("Rows x Cols:", df.shape)

    # Quick pre-audit
    pre_miss = (df.isna().mean()*100).round(2)

    print("➡️ Coercing types + basic sanitization…")
    df = coerce_types(df)
    df = clamp_and_sanitize(df)

    # Vectorized imputations
    print("➡️ Imputing numeric features by genre/global…")
    num_cols = [c for c in NUM_01 + NUM_MISC if c in df.columns]
    imp_num = impute_numeric_by_genre(df, num_cols)

    print("➡️ Imputing categorical features by genre/global mode…")
    cat_cols = [c for c in CATEG if c in df.columns]
    imp_cat = impute_categorical_by_genre(df, cat_cols)

    print("➡️ Dropping rows with missing essential columns…")
    df, dropped, essentials = drop_unusable(df)
    print(f"   Dropped {dropped} rows; essentials = {essentials}")

    # Remove exact duplicates (but keep multiple quarters per song)
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Removed {before - len(df)} exact duplicate rows.")

    df = finalize_dtypes_for_modeling(df)

    # Post-audit + save audit table
    post_miss = (df.isna().mean()*100).round(2)
    audit = (pd.DataFrame({"pre_missing_%": pre_miss, "post_missing_%": post_miss})
               .sort_values("post_missing_%", ascending=False))
    audit.to_csv(AUDIT, index=True)

    print("➡️ Saving…")
    df.to_parquet(OUTP, index=False)
    df.to_csv(OUTCSV, index=False)
    print("✅ Saved:", OUTP)
    print("✅ Saved:", OUTCSV)
    print("Rows x Cols (clean):", df.shape)
    print("Imputed numerics:", imp_num)
    print("Imputed categoricals:", imp_cat)
    print("📄 Audit:", AUDIT)

if __name__ == "__main__":
    main()
    