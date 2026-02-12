import requests
import pandas as pd
from io import StringIO

URL = "https://etherscan.io/chart/tokenerc-20txns?output=csv"
OUT = "./data_preprocessing/data/metrics/eth_erc20_daily_token_transfers.csv"

headers = {"User-Agent": "Mozilla/5.0"}

r = requests.get(URL, headers=headers, timeout=30)
r.raise_for_status()

df = pd.read_csv(StringIO(r.text))
print("Columns:", list(df.columns))

value_candidates = [c for c in df.columns if c not in ("Date(UTC)", "UnixTimeStamp")]
if len(value_candidates) != 1:
    raise RuntimeError(f"Unexpected value columns: {value_candidates}")
value_col = value_candidates[0]

df["timestamp"] = pd.to_numeric(df["UnixTimeStamp"], errors="raise").astype("int64")
df["erc20_daily_token_transfers"] = pd.to_numeric(df[value_col], errors="raise")

df = df.sort_values("timestamp")
df[["timestamp", "Date(UTC)", "erc20_daily_token_transfers"]].to_csv(OUT, index=False)

print("Saved:", OUT)
print(df.head())
