from metrics import PipelineConfig, build_universal_metrics


def main() -> None:
    config = PipelineConfig(etherscan_api_key="7K814DY5AXIQCHEH9VKWBBIP1AAMHU2VIS")
    df = build_universal_metrics(period="30d", caller="main", config=config, save=True)
    print(df.head())


if __name__ == "__main__":
    main()
