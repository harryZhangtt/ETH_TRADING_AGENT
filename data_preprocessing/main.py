from pathlib import Path

from analytics import AnalyticsConfig, analyze_metrics
from further_preprocessing import build_midway_dataset
from metrics import PipelineConfig,build_universal_metrics


def main() -> None:
    pipeline_config = PipelineConfig(etherscan_api_key="7K814DY5AXIQCHEH9VKWBBIP1AAMHU2VIS", debug=False)


    ##2015-08-08
    metrics_df = build_universal_metrics(start="2015-08-08", end="2026-02-11", caller="main", config=pipeline_config, save=True)



    metrics_csv = f"{pipeline_config.output_dir}/eth_metrics_combined.csv"
    
    if not Path(metrics_csv).exists():
        raise FileNotFoundError(
            f"Metrics CSV not found at {metrics_csv}. Run the metrics pipeline first."
        )

    build_midway_dataset(
        input_csv=Path(metrics_csv),
        output_dir=Path("data_preprocessing/data/midway"),
        output_name="eth_metrics_midway.csv",
        multi_day_lag=7,
    )

    analytics_config = AnalyticsConfig(
        input_csv=metrics_csv,
        output_dir="data_preprocessing/data/analytics",
        debug=False,
    )
    analyze_metrics(df=None, config=analytics_config, save=True)


if __name__ == "__main__":
    main()
