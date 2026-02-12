import os
import requests
import pandas as pd
from io import StringIO

URL = "https://etherscan.io/chart/ethersupplygrowth?output=csv"
OUT = "./data_preprocessing/data/metrics/eth_supply_growth.csv"

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(URL, headers=headers, timeout=30)
r.raise_for_status()

df = pd.read_csv(StringIO(r.text))
print("Columns:", list(df.columns))

if "UnixTimeStamp" in df.columns:
    df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="raise").astype("int64")
else:
    if "Date(UTC)" not in df.columns:
        raise RuntimeError(
            f"Missing both UnixTimeStamp and Date(UTC). Columns={list(df.columns)}"
        )
    dt = pd.to_datetime(df["Date(UTC)"], utc=True, errors="raise")
    df["timestamp"] = (dt.view("int64") // 10**9).astype("int64")

if "Date(UTC)" not in df.columns:
    df["Date(UTC)"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime(
        "%-m/%-d/%Y"
    )

for c in ("UnixTimeStamp", "DateTime"):
    if c in df.columns:
        df = df.drop(columns=[c])

rename_map = {}
for col in df.columns:
    if col in ("timestamp", "Date(UTC)"):
        continue
    new = (
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("/", "_")
    )
    while "__" in new:
        new = new.replace("__", "_")
    rename_map[col] = new

df = df.rename(columns=rename_map)

df = df.sort_values("timestamp")
out_cols = ["timestamp", "Date(UTC)"] + [
    c for c in df.columns if c not in ("timestamp", "Date(UTC)")
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df[out_cols].to_csv(OUT, index=False)

print("Saved:", OUT)
print(df[out_cols].head())
