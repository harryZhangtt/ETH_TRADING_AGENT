import io
from typing import Optional

import pandas as pd
import requests


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def fetch_chart_csv(
    url: str,
    api_key: Optional[str] = None,
    debug: bool = False,
) -> pd.DataFrame:
    params = {"output": "csv"}
    if api_key:
        params["apikey"] = api_key

    try:
        response = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
        response.raise_for_status()
    except (requests.RequestException, ValueError) as exc:
        if debug:
            print(f"[DEBUG] Etherscan chart request failed: {exc}")
        return pd.DataFrame()

    text = response.text.strip()
    if not text:
        if debug:
            print("[DEBUG] Etherscan chart response empty")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        if debug:
            print(f"[DEBUG] Etherscan chart CSV parse failed: {exc}")
        return pd.DataFrame()

    return df
