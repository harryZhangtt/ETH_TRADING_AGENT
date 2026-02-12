import os
import requests
import pandas as pd
from io import StringIO

URL = "https://etherscan.io/chart/avg-txfee-usd?output=csv"
OUT = "./data_preprocessing/data/metrics/eth_avg_txfee.csv"

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(URL, headers=headers, timeout=30)
r.raise_for_status()

df = pd.read_csv(StringIO(r.text))
print("Columns:", list(df.columns))

df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="raise").astype("int64")

rename_map = {}
if "Average Txn Fee (USD)" in df.columns:
    rename_map["Average Txn Fee (USD)"] = "avg_txfee_usd"
if "Average Txn Fee (Ether)" in df.columns:
    rename_map["Average Txn Fee (Ether)"] = "avg_txfee_eth"

if len(rename_map) == 0:
    raise RuntimeError("Expected fee columns not found.")

df = df.rename(columns=rename_map)

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
