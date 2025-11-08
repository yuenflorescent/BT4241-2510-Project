#------Check missing values--------
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet("data_final/panel_modelA_song_quarter.parquet")

# Missingness overview
nulls = df.isna().sum().sort_values(ascending=False)
print("Missing counts:\n", nulls)
print("\nMissing %:\n", (df.isna().mean()*100).round(2))

# By column groups (audio features vs meta)
features = ["artist_popularity","acousticness","danceability","duration_ms","energy",
            "instrumentalness","key","liveness","loudness","mode","speechiness",
            "tempo","time_signature","valence"]
print("\nFeatures missing %:\n", (df[features].isna().mean()*100).round(2))

# Sanity: value ranges
rng = df[["energy","danceability","acousticness","valence","speechiness",
          "liveness","instrumentalness"]].describe().T
print("\nRange check (0–1 features):\n", rng[["min","max"]])

# Duplicates
print("\nExact duplicate rows:", df.duplicated().sum())
print("Duplicate songs (artist_norm, track_norm):",
      df.duplicated(["artist_norm","track_norm"]).sum())


#------Impute missing values---------
# Impute the Model A panel (genre-level median/mode with global fallback)
# Inputs : data_final/panel_modelA_song_quarter.parquet
# Outputs: data_final/panel_modelA_song_quarter_clean.parquet/.csv

INP    = Path("data_final/panel_modelA_song_quarter.parquet")
OUTP   = Path("data_final/panel_modelA_song_quarter_clean.parquet")
OUTCSV = Path("data_final/panel_modelA_song_quarter_clean.csv")

# Columns in your panel (per your schema)
NUM_01   = ["acousticness","danceability","energy","instrumentalness","liveness","speechiness","valence"]
NUM_MISC = ["tempo","loudness","duration_ms","artist_popularity"]
CATEG    = ["genre","key","mode","time_signature"]
ESSENTIAL= ["artist_norm","track_norm","inverse_rank","peak_position","recession_t","event_time_k","release_q"]

def mode_or_na(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else pd.NA

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Dates (if present)
    for col in ["first_week","last_week"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Numeric (force invalid to NaN)
    for col in set(NUM_01 + NUM_MISC + ["inverse_rank","peak_position","weeks_on"]):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Categoricals
    for col in CATEG:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # Treatment/int-like columns as nullable ints if present
    for col in ["recession_t","event_time_k","top10","key","mode","time_signature"]:
        if col in df.columns:
            try:
                df[col] = df[col].astype("Int64")
            except Exception:
                pass
    return df

def clamp_and_sanitize(df: pd.DataFrame) -> pd.DataFrame:
    # Clamp [0,1] features
    for col in NUM_01:
        if col in df.columns:
            df[col] = df[col].clip(0, 1)
    # Basic sanity on misc numerics
    if "tempo" in df.columns:
        df.loc[df["tempo"] <= 0, "tempo"] = np.nan
    if "loudness" in df.columns:
        # loudness in dB is typically <= 0
        df.loc[df["loudness"] > 0, "loudness"] = np.nan
    if "duration_ms" in df.columns:
        df.loc[df["duration_ms"] <= 0, "duration_ms"] = np.nan
    return df

def impute_numeric_by_genre(df: pd.DataFrame, cols: list):
    # Ensure genre exists for grouping
    if "genre" not in df.columns:
        df["genre"] = "unknown"
    df["genre"] = df["genre"].astype("string").fillna("unknown")

    # Pre-compute genre medians for all numeric cols (will only keep intersecting ones)
    med_by_genre = df.groupby("genre").median(numeric_only=True)

    imputed_counts = {}
    for col in cols:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()

        # 1) Fill from genre median where available
        def fill_genre(row):
            if pd.notna(row[col]): 
                return row[col]
            g = row["genre"]
            if g in med_by_genre.index and col in med_by_genre.columns:
                return med_by_genre.loc[g, col]
            return np.nan

        df[col] = df.apply(fill_genre, axis=1)

        # 2) Global median fallback
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

        after = df[col].isna().sum()
        imputed_counts[col] = before - after
    return df, imputed_counts

def impute_categorical_by_genre(df: pd.DataFrame, cols: list):
    if "genre" not in df.columns:
        df["genre"] = "unknown"
    df["genre"] = df["genre"].astype("string").fillna("unknown")

    imputed_counts = {}
    for col in cols:
        if col not in df.columns:
            continue
        before = df[col].isna().sum() if df[col].isna().any() else int((df[col] == "").sum())
        # Mode per-genre
        mode_by_genre = (
            df.groupby("genre")[col]
              .apply(lambda s: mode_or_na(s.dropna()))
        )
        def fill_cat(row):
            v = row[col]
            if pd.notna(v) and v != "":
                return v
            mg = mode_by_genre.get(row["genre"], pd.NA)
            if pd.notna(mg):
                return mg
            gm = df[col].mode(dropna=True)
            return gm.iloc[0] if not gm.empty else ("unknown" if col=="genre" else pd.NA)

        df[col] = df.apply(fill_cat, axis=1)
        try:
            df[col] = df[col].astype("category")
        except Exception:
            pass
        after = df[col].isna().sum() if df[col].isna().any() else int((df[col] == "").sum())
        imputed_counts[col] = max(0, before - after)
    return df, imputed_counts

def drop_unusable(df: pd.DataFrame):
    essential = [c for c in ESSENTIAL if c in df.columns]
    before = len(df)
    df = df.dropna(subset=essential)
    return df, before - len(df), essential

def main():
    print("➡️ Loading panel…")
    df = pd.read_parquet(INP)
    print("Rows x Cols:", df.shape)

    print("➡️ Initial missingness (top):")
    miss = (df.isna().mean()*100).sort_values(ascending=False).round(2)
    print(miss.head(15))

    print("➡️ Coercing types + basic sanitization…")
    df = coerce_types(df)
    df = clamp_and_sanitize(df)

    # Impute numerics (0–1 + misc) by genre, fallback to global median
    print("➡️ Imputing numeric features by genre/global…")
    df, imp_num = impute_numeric_by_genre(df, [c for c in NUM_01 + NUM_MISC if c in df.columns])

    # Impute categoricals by genre/global mode
    print("➡️ Imputing categorical features by genre/global mode…")
    df, imp_cat = impute_categorical_by_genre(df, [c for c in CATEG if c in df.columns])

    # Drop rows still missing essential pillars (very rare after imputation)
    print("➡️ Dropping rows with missing essential columns…")
    df, dropped, essential = drop_unusable(df)
    print(f"Dropped {dropped} rows; essentials = {essential}")

    # De-duplicate exact duplicates (keep panel duplicates across quarters)
    dup_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {dup_before - len(df)} exact duplicate rows.")

    print("➡️ Post-imputation missingness (top):")
    miss2 = (df.isna().mean()*100).sort_values(ascending=False).round(2)
    print(miss2.head(15))

    print("➡️ Imputed counts (numerics):", imp_num)
    print("➡️ Imputed counts (categoricals):", imp_cat)

    print("➡️ Saving…")
    df.to_parquet(OUTP, index=False)
    df.to_csv(OUTCSV, index=False)
    print("✅ Saved:", OUTP)
    print("✅ Saved:", OUTCSV)
    print("Rows x Cols (clean):", df.shape)

if __name__ == "__main__":
    main()

# Save the cleaned panel dataset
df.to_csv("data_final/panel_modelA_song_quarter_clean.csv", index=False)
print("✅ Cleaned dataset saved to data_final/panel_modelA_song_quarter_clean.csv")



