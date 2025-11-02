# src/01_build_keys.py
import sys
import pandas as pd
from rapidfuzz import process, fuzz
from janitor import clean_names
from pathlib import Path
from utils import norm  # text normalization helper

RAW = Path("data_raw")
OUT = Path("data_interim")
OUT.mkdir(parents=True, exist_ok=True)


def pick(colnames, candidates, label):
    """Pick the first existing column from a list of candidates; raise a clear error if none found."""
    for c in candidates:
        if c in colnames:
            return c
    raise KeyError(f"[{label}] Expected one of {candidates}, found {list(colnames)}")


def main():
    print("➡️ Loading CSVs...")
    sp = pd.read_csv(RAW / "spotify_tracks.csv").pipe(clean_names)
    bb = pd.read_csv(RAW / "billboard_hot100_weekly.csv").pipe(clean_names)
    print(f"   spotify_tracks shape: {sp.shape}")
    print(f"   billboard shape      : {bb.shape}")

    # Detect column names across possible variants
    sp_artist_col = pick(sp.columns, ["artist", "artists", "artist_name"], "spotify artist")
    sp_track_col  = pick(sp.columns, ["track_name", "track_nam", "name", "track", "song"], "spotify track")
    bb_artist_col = pick(bb.columns, ["artist", "artists", "artist_name"], "billboard artist")
    bb_track_col  = pick(bb.columns, ["song", "track", "title", "track_name", "name"], "billboard track")
    bb_date_col   = pick(bb.columns, ["date", "week", "week_date"], "billboard date")
    bb_rank_col   = pick(bb.columns, ["rank", "position", "pos"], "billboard rank")
    print("   detected columns ✔")

    # Normalize keys used for matching
    print("➡️ Normalizing keys...")
    sp["artist_norm"] = sp[sp_artist_col].map(norm)
    sp["track_norm"]  = sp[sp_track_col].map(norm)
    bb["artist_norm"] = bb[bb_artist_col].map(norm)
    bb["track_norm"]  = bb[bb_track_col].map(norm)

    # Direct merge on normalized artist + track
    print("➡️ Direct merging...")
    cols_sp_keep = [c for c in sp.columns if c not in ["artist_norm", "track_norm"]]
    m = bb.merge(
        sp[["artist_norm", "track_norm"] + cols_sp_keep],
        on=["artist_norm", "track_norm"],
        how="left",
        indicator=True
    )
    direct_rate = 100 * (m["_merge"] != "left_only").mean()
    print(f"   direct match rate: {direct_rate:.1f}%")

    # ---------- Fuzzy match with prefix blocking (fast & robust) ----------
    unmatched = m[m["_merge"] == "left_only"].copy()
    if len(unmatched):
        print(f"➡️ Fuzzy matching with blocking for {len(unmatched):,} rows...")

        add_cols = [c for c in sp.columns if c not in ["artist_norm", "track_norm"]]

        # Build consistent 'key' and 'prefix' for Spotify candidates
        sp_map = sp.assign(
            key=(sp["artist_norm"].fillna("") + " - " + sp["track_norm"].fillna("")),
            prefix=sp["artist_norm"].fillna("").str[:2] + "|" + sp["track_norm"].fillna("").str[:2]
        )

        # Candidate strings per prefix (do not rely on indexing quirks)
        cand_keys = {}
        for p, grp in sp_map.groupby("prefix"):
            cand_keys[p] = (grp["artist_norm"].fillna("") + " - " + grp["track_norm"].fillna("")).tolist()

        # Payload map: key -> {artist_norm, track_norm, ...features}
        payload_cols = ["artist_norm", "track_norm"] + add_cols
        payload = {k: rec for k, rec in zip(
            sp_map["key"].tolist(),
            sp_map[payload_cols].to_dict("records")
        )}

        # Same 'key' and 'prefix' for the unmatched Billboard rows
        unmatched = unmatched.assign(
            key=(unmatched["artist_norm"].fillna("") + " - " + unmatched["track_norm"].fillna("")),
            prefix=unmatched["artist_norm"].fillna("").str[:2] + "|" + unmatched["track_norm"].fillna("").str[:2]
        )

        rows = []
        total = len(unmatched)
        for i, row in unmatched.iterrows():
            if i % 10000 == 0 and i > 0:
                print(f"   processed {i:,}/{total:,} ...")

            cands = cand_keys.get(row["prefix"], [])
            if not cands:
                continue

            match = process.extractOne(row["key"], cands, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 90:
                k = match[0]
                info = payload.get(k)
                if info:
                    rows.append(info)

        if rows:
            use_df = pd.DataFrame(rows)
            # Drop any prior spotify columns before merging back to avoid duplicates
            m = m.drop(columns=[c for c in add_cols if c in m.columns], errors="ignore")
            m = m.merge(use_df, on=["artist_norm", "track_norm"], how="left")
            print(f"   fuzzy matched (>=90): {len(use_df):,}")
        else:
            print("   no rows passed fuzzy threshold after blocking")
    else:
        print("   no fuzzy matching needed")
    # ----------------------------------------------------------------------

    # Compute peaks per song
    print("➡️ Computing peaks...")
    bbc = bb.copy()
    bbc[bb_date_col] = pd.to_datetime(bbc[bb_date_col], errors="coerce")
    peaks = (
        bbc.sort_values([bb_artist_col, bb_track_col, bb_date_col, bb_rank_col])
        .groupby(["artist_norm", "track_norm"], as_index=False)
        .agg(
            peak_position=(bb_rank_col, "min"),
            first_week=(bb_date_col, "min"),
            last_week=(bb_date_col, "max"),
            weeks_on=(bb_date_col, "nunique"),
        )
    )

    # Save outputs
    print("➡️ Saving to data_interim/ ...")
    m.to_parquet(OUT / "billboard_spotify_weeks.parquet", index=False)
    peaks.to_parquet(OUT / "billboard_peaks.parquet", index=False)
    print("✅ Done. Files saved:")
    print("   -", OUT / "billboard_spotify_weeks.parquet")
    print("   -", OUT / "billboard_peaks.parquet")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ Error:", e)
        sys.exit(1)


