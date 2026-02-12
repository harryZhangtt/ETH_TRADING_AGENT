import requests
import pandas as pd
from io import StringIO

URL = "https://etherscan.io/chart/address?output=csv"
OUT = "./data_preprocessing/data/metrics/eth_unique_addresses.csv"

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(URL, headers=headers, timeout=30)
r.raise_for_status()

df = pd.read_csv(StringIO(r.text))
print("Columns:", list(df.columns))

df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="raise").astype("int64")
df["total_unique_addresses"] = pd.to_numeric(df["Value"], errors="raise").astype("int64")

df = df.sort_values("timestamp")
df["daily_increase"] = df["total_unique_addresses"].diff().fillna(0).astype("int64")

df[["timestamp", "Date(UTC)","total_unique_addresses", "daily_increase"]].to_csv(OUT, index=False)
print("Saved:", OUT)
