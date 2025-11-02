# src/02_make_panel_modelA.py
import pandas as pd
from pathlib import Path
from utils import load_recession_calendar, build_quarter_index, norm

IN  = Path("data_interim")
RAW = Path("data_raw")
OUT = Path("data_final")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("➡️ Loading intermediates...")
    peaks = pd.read_parquet(IN / "billboard_peaks.parquet")   # has artist_norm, track_norm, first_week, etc.
    sp    = pd.read_csv(RAW / "spotify_tracks.csv")

    # Build normalized keys from your Spotify columns
    # (your file has artist_name & track_name)
    sp["artist_norm"] = sp["artist_name"].map(norm)
    sp["track_norm"]  = sp["track_name"].map(norm)

    # Select available features from your schema
    keep = [
        "artist_norm","track_norm","genre","track_id","popularity",
        "acousticness","danceability","duration_ms","energy",
        "instrumentalness","key","liveness","loudness","mode",
        "speechiness","tempo","time_signature","valence"
    ]
    keep = [c for c in keep if c in sp.columns]
    feats = sp[keep].copy()
    if "popularity" in feats.columns:
        feats = feats.rename(columns={"popularity":"artist_popularity"})

    # Merge peaks (Billboard) with Spotify features on normalized keys
    df = peaks.merge(feats, on=["artist_norm","track_norm"], how="left")

    # Outcomes
    df["inverse_rank"] = 101 - df["peak_position"]
    df["top10"] = (df["peak_position"] <= 10).astype(int)

    # Quarter: FALLBACK to Billboard first_week because Spotify has no release_date
    df["first_week"] = pd.to_datetime(df["first_week"], errors="coerce")
    df["release_q"]  = df["first_week"].dt.to_period("Q").astype(str)

    # Recession calendar from USRECQ
    cal_q = load_recession_calendar(RAW / "USRECQ.csv")
    df = df.merge(cal_q, left_on="release_q", right_on="quarter", how="left").drop(columns=["quarter"])
    df["recession_t"] = df["recession_t"].fillna(0).astype(int)

    # Event time k relative to 2007Q4
    order = build_quarter_index(cal_q)
    k0 = order.get("2007Q4")
    df["event_time_k"] = df["release_q"].map(order) - k0

    # Keep a compact window for estimation
    df = df[(df["event_time_k"] >= -8) & (df["event_time_k"] <= 8)].copy()

    # Save
    df.to_parquet(OUT / "panel_modelA_song_quarter.parquet", index=False)
    df.to_csv(OUT / "panel_modelA_song_quarter.csv", index=False)
    print("✅ Saved:", OUT / "panel_modelA_song_quarter.parquet")

if __name__ == "__main__":
    main()
