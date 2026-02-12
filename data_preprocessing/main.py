from metrics import PipelineConfig, build_universal_metrics


def main() -> None:
    config = PipelineConfig(etherscan_api_key="7K814DY5AXIQCHEH9VKWBBIP1AAMHU2VIS")
    df = build_universal_metrics(start="2015-08-08", end="2026-02-11", caller="main", config=config, save=True)
    print(df.head())


if __name__ == "__main__":
    main()
