# src/02_make_panel_modelA.py
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from utils import norm  # text normalization helper

RAW = Path("data_raw")
IN = Path("data_interim")
OUT = Path("data_processed")
OUT.mkdir(parents=True, exist_ok=True)


def primary_genre_cell(x):
    if pd.isna(x):
        return pd.NA
    x = str(x)
    for sep in [";", ",", "|", "/"]:
        if sep in x:
            return x.split(sep)[0].strip()
    return x.strip()


def mode_or_first(s):
    s = s.dropna()
    m = s.mode()
    return m.iat[0] if not m.empty else (s.iloc[0] if len(s) else pd.NA)


def load_recession_calendar(csv_path: Path) -> pd.DataFrame:
    """
    Load a quarterly recession indicator CSV (e.g., FRED USRECQ) and return:
      quarter (YYYYQ#), recession_t (0/1)
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty")

    # date column
    date_col = None
    for c in df.columns:
        if c in ["DATE", "date", "observation_date", "quarter", "Quarter"]:
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            lc = c.lower()
            if "date" in lc or "quarter" in lc:
                date_col = c
                break
    if date_col is None:
        raise ValueError(f"Could not detect date column. Available: {list(df.columns)}")

    # recession column
    rec_col = None
    for c in df.columns:
        lc = c.lower()
        if "usrecq" in lc or "recession" in lc or lc == "rec" or lc.endswith("_rec"):
            rec_col = c
            break
    if rec_col is None:
        non_date = [c for c in df.columns if c != date_col]
        if len(df.columns) == 2 and len(non_date) == 1:
            rec_col = non_date[0]
    if rec_col is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        for c in numeric_cols:
            if c != date_col:
                rec_col = c
                break
    if rec_col is None:
        raise ValueError(f"Could not detect recession flag column. Available: {list(df.columns)}")

    q = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("Q")
    if q.isna().all():
        raise ValueError(f"Date column '{date_col}' could not be parsed to dates.")
    rec = pd.to_numeric(df[rec_col], errors="coerce").fillna(0)
    rec = (rec > 0).astype(int)

    out = pd.DataFrame({"quarter": q.astype(str), "recession_t": rec})
    # 0/1 max per quarter, dedup
    out = out.groupby("quarter", as_index=False)["recession_t"].max()
    return out


def build_quarter_index(cal_q: pd.DataFrame):
    """Return (idx->quarter, quarter->idx) using calendar order."""
    quarters_sorted = (
        pd.Series(cal_q["quarter"].unique()).astype("period[Q]").sort_values().astype(str).tolist()
    )
    idx_to_quarter = {i: q for i, q in enumerate(quarters_sorted)}
    quarter_to_idx = {q: i for i, q in idx_to_quarter.items()}
    return idx_to_quarter, quarter_to_idx


def main():
    print("➡️ Loading intermediates...")
    peaks = pd.read_parquet(IN / "billboard_peaks.parquet")          # artist_norm, track_norm, first_week, peak_position...
    weeks = pd.read_parquet(IN / "billboard_spotify_weeks.parquet")  # includes weekly Spotify features
    sp    = pd.read_csv(RAW / "spotify_tracks.csv")

    # ---------- Weekly genre map from Step 01 (preferred) ----------
    has_weekly_genre = "genre" in weeks.columns
    if has_weekly_genre:
        genre_map = (
            weeks.groupby(["artist_norm", "track_norm"])["genre"]
                 .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else pd.NA)
                 .rename("genre_weekly")
                 .reset_index()
        )
        n_weekly = genre_map["genre_weekly"].notna().sum()
        print(f"   weekly genre map rows: {len(genre_map)} (non-null genres: {n_weekly})")
    else:
        print("⚠️ weeks table has no 'genre' column—will rely on Spotify fallback.")
        genre_map = pd.DataFrame(columns=["artist_norm", "track_norm", "genre_weekly"])

    # ---------- Spotify features (fallback genre) ----------
    if "artist_norm" not in sp.columns or "track_norm" not in sp.columns:
        artist_col = next(c for c in ["artist", "artists", "artist_name"] if c in sp.columns)
        track_col  = next(c for c in ["track_name","track_nam","name","track","song"] if c in sp.columns)
        sp["artist_norm"] = sp[artist_col].map(norm)
        sp["track_norm"]  = sp[track_col].map(norm)

    keep = [
        "artist_norm","track_norm","genre","track_id","popularity",
        "acousticness","danceability","duration_ms","energy",
        "instrumentalness","key","liveness","loudness","mode",
        "speechiness","tempo","time_signature","valence"
    ]
    keep = [c for c in keep if c in sp.columns]
    feats = sp[keep].copy()
    if "popularity" in feats.columns and "artist_popularity" not in feats.columns:
        feats = feats.rename(columns={"popularity":"artist_popularity"})
    if "genre" in feats.columns:
        feats["genre"] = feats["genre"].map(primary_genre_cell)

    agg = {}
    for col in ["track_id","key","mode","time_signature","genre"]:
        if col in feats.columns:
            agg[col] = (col, mode_or_first)
    for col in ["artist_popularity","acousticness","danceability","duration_ms","energy",
                "instrumentalness","liveness","loudness","speechiness","tempo","valence"]:
        if col in feats.columns:
            agg[col] = (col, "median" if col=="duration_ms" else "mean")
    feats_one = feats.groupby(["artist_norm","track_norm"], as_index=False).agg(**agg)
    n_sp_genre = feats_one["genre"].notna().sum() if "genre" in feats_one.columns else 0
    print(f"   Spotify dedup: {len(feats)} → {len(feats_one)} unique songs (Spotify genre non-null: {n_sp_genre})")

    # ---------- Merge peaks + features; apply weekly-preferred genre ----------
    df = peaks.merge(feats_one, on=["artist_norm","track_norm"], how="left")
    df = df.merge(genre_map, on=["artist_norm","track_norm"], how="left")
    if "genre" not in df.columns:
        df["genre"] = pd.NA
    df["genre"] = np.where(df["genre_weekly"].notna(), df["genre_weekly"], df["genre"])
    df = df.drop(columns=["genre_weekly"], errors="ignore")
    print(f"   after genre selection: rows={len(df)}, non-null genres={df['genre'].notna().sum()}")

    # ---------- Outcomes at the SONG level (will be repeated across quarters) ----------
    if "peak_position" not in df.columns:
        raise ValueError("peak_position missing in peaks; cannot compute outcomes.")
    df["inverse_rank"] = 101 - df["peak_position"]
    df["top10"]        = (df["peak_position"] <= 10).astype(int)

    # ---------- Build release quarter ----------
    if "first_week" not in df.columns:
        raise ValueError("first_week missing in peaks; cannot build release_q.")
    df["first_week"] = pd.to_datetime(df["first_week"], errors="coerce")
    df["release_q"]  = df["first_week"].dt.to_period("Q").astype(str)

    # ---------- Quarter calendar & index ----------
    cal_q = load_recession_calendar(RAW / "USRECQ.csv")  # expects observation_date + USRECQ (works with FRED)
    idx_to_quarter, quarter_to_idx = build_quarter_index(cal_q)

    # Map each song to its integer release index
    df["release_idx"] = df["release_q"].map(quarter_to_idx)
    missing_idx = df["release_idx"].isna().sum()
    if missing_idx:
        print(f"⚠️ {missing_idx} songs have release_q not in calendar; dropping them.")
        df = df[df["release_idx"].notna()].copy()
    df["release_idx"] = df["release_idx"].astype(int)

    # ---------- Expand to a true (song × quarter) PANEL ----------
    # Event-time support: K in [-8, +8]
    k_values = pd.DataFrame({"event_time_k": list(range(-8, 9))})
    df["__tmp__"] = 1
    k_values["__tmp__"] = 1
    panel = df.merge(k_values, on="__tmp__").drop(columns="__tmp__", errors="ignore")

    # Compute quarter index and map to quarter string
    panel["quarter_idx"] = panel["release_idx"] + panel["event_time_k"]
    # keep only valid quarters that exist in calendar
    max_idx = max(idx_to_quarter.keys())
    panel = panel[(panel["quarter_idx"] >= 0) & (panel["quarter_idx"] <= max_idx)].copy()
    panel["quarter"] = panel["quarter_idx"].map(idx_to_quarter)

    # Bring in recession_t (time-varying)
    panel = panel.merge(cal_q, on="quarter", how="left")  # adds recession_t

    # Keep tidy order
    # (artist_norm, track_norm, quarter, event_time_k, genre, top10, inverse_rank, features, etc.)
    lead_cols = [
        "artist_norm", "track_norm", "quarter", "event_time_k",
        "genre", "inverse_rank", "top10",
        "peak_position", "first_week", "release_q"
    ]
    lead_cols = [c for c in lead_cols if c in panel.columns]
    other_cols = [c for c in panel.columns if c not in lead_cols and c not in ["quarter_idx","release_idx"]]
    panel = panel[lead_cols + other_cols].sort_values(["artist_norm","track_norm","quarter"]).reset_index(drop=True)

    # Final sanity: genre coverage
    miss_pct = (1 - panel["genre"].notna().mean()) * 100
    print("   PANEL rows:", len(panel))
    print("   genre missing %:", round(miss_pct, 2))

    # Save
    # --- Save directly to data_final ---
    OUT_FINAL = Path("data_final")
    OUT_FINAL.mkdir(parents=True, exist_ok=True)

    out_pqt = OUT_FINAL / "panel_modelA_song_quarter.parquet"
    out_csv = OUT_FINAL / "panel_modelA_song_quarter.csv"

    # Save both formats
    panel.to_parquet(out_pqt, index=False)
    panel.to_csv(out_csv, index=False)

    print("✅ Saved directly to data_final:")
    print("   -", out_pqt.resolve())
    print("   -", out_csv.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        sys.exit(1)







