import os
import requests
import pandas as pd
from io import StringIO

URL = "https://etherscan.io/chart/blocksize?output=csv"
OUT = "./data_preprocessing/data/metrics/eth_avg_block_size.csv"

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(URL, headers=headers, timeout=30)
r.raise_for_status()

df = pd.read_csv(StringIO(r.text))
print("Columns:", list(df.columns))

if "UnixTimeStamp" in df.columns:
    df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="raise").astype("int64")
else:
    dt = pd.to_datetime(df["Date(UTC)"], utc=True, errors="raise")
    df["timestamp"] = (dt.view("int64") // 10**9).astype("int64")

if "Date(UTC)" not in df.columns:
    df["Date(UTC)"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime(
        "%-m/%-d/%Y"
    )

exclude = {"Date(UTC)", "UnixTimeStamp", "DateTime", "timestamp"}
value_candidates = [c for c in df.columns if c not in exclude]
if len(value_candidates) != 1:
    raise RuntimeError(f"Unexpected value columns: {value_candidates}")
value_col = value_candidates[0]

df["avg_block_size_bytes"] = pd.to_numeric(df[value_col], errors="raise")
df = df.drop(columns=[value_col])

for c in ("UnixTimeStamp", "DateTime"):
    if c in df.columns:
        df = df.drop(columns=[c])

df = df.sort_values("timestamp")
out_cols = ["timestamp", "Date(UTC)"] + [
    c for c in df.columns if c not in ("timestamp", "Date(UTC)")
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df[out_cols].to_csv(OUT, index=False)

print("Saved:", OUT)
print(df[out_cols].head())
