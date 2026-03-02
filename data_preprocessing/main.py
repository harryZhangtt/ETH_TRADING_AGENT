from pathlib import Path

from analytics import AnalyticsConfig, analyze_metrics
from metrics import PipelineConfig, build_universal_metrics
from metrics.universal_caller import append_upon_universal_metrics

_COMBINED_CSV = "data/metrics/eth_metrics_combined.csv"
_MACRO_DIR = "data/Macro_econ_data"
_MACRO_FIELDS = {
    "sp500":     "sp500_price_2015.csv",
    "nasdaq":    "nasdaq_price_2015.csv",
    "dow30":     "dow30_price_2015.csv",
    "vix":       "vix_price_2015.csv",
    "gold":      "gold_price_2015.csv",
    "crude_oil": "crude_oil_price_2015.csv",
    "nikkei225": "nikkei225_price_2015.csv",
    "ftse100":   "ftse100_price_2015.csv",
    "sse":       "sse_price_2015.csv",
    "eurostoxx": "eurostoxx_price_2015.csv",
}


def main() -> None:
    pipeline_config = PipelineConfig(etherscan_api_key="7K814DY5AXIQCHEH9VKWBBIP1AAMHU2VIS", debug=False)

    # ── Step 1: build or reuse the hourly ETH metrics CSV ─────────────────
    if not Path(_COMBINED_CSV).exists():
        build_universal_metrics(
            start="2015-08-08", end="2026-02-11",
            caller="main", config=pipeline_config, save=True,
        )

    # ── Step 2: enrich with macro-economic data ────────────────────────────
    append_upon_universal_metrics(
        datas=[f"{_MACRO_DIR}/{v}" for v in _MACRO_FIELDS.values()],
        fields_to_append=list(_MACRO_FIELDS.keys()),
        combined_csv=_COMBINED_CSV,
        save=True,
        output_dir="data/metrics",
        output_filename="eth_metrics_combined_macro.csv",
    )
    macro_csv = "data/metrics/eth_metrics_combined_macro.csv"

    analytics_config = AnalyticsConfig(
        input_csv=macro_csv,
        output_dir="=data/analytics",
        debug=False,
    )
    analyze_metrics(df=None, config=analytics_config, save=True)


if __name__ == "__main__":
    main()
