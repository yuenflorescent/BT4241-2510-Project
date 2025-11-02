# src/99_utils.py
import re
import pandas as pd

def norm(s: str) -> str:
    """Chuẩn hoá chuỗi để match artist/track (hạ chữ, bỏ feat., ngoặc, ký tự lạ)."""
    if pd.isna(s): return ""
    s = str(s).lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", "", s)
    s = re.sub(r"\b(feat|ft)\.?\b.*", "", s)
    s = re.sub(r"[-–]\s*(remaster.*|radio edit.*|live.*)$", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_recession_calendar(usrecq_csv="data_raw/USRECQ.csv"):
    """Đọc USRECQ (FRED) -> bảng quý + cờ suy thoái."""
    rec = pd.read_csv(usrecq_csv)
    if "DATE" not in rec.columns:
        rec.rename(columns={rec.columns[0]:"DATE"}, inplace=True)
    if "USRECQ" not in rec.columns:
        rec.rename(columns={rec.columns[1]:"USRECQ"}, inplace=True)
    rec["quarter"] = pd.to_datetime(rec["DATE"]).dt.to_period("Q").astype(str)
    cal_q = rec[["quarter","USRECQ"]].rename(columns={"USRECQ":"recession_t"})
    return cal_q

def build_quarter_index(cal_q: pd.DataFrame):
    """Map quý -> chỉ số để tính event time."""
    return {q:i for i,q in enumerate(sorted(cal_q["quarter"].unique()))}
