from typing import Optional

import pandas as pd

from .common.config import AnalyticsConfig
from .common.io_utils import ensure_dir, load_csv, save_csv, save_json
from .common.preprocess import (
    add_return_features,
    cast_numeric,
    ensure_timestamp_datetime,
    impute_missing,
    missingness_report,
    prepare_features,
    sort_dedup,
    validate_schema,
    winsorize,
)
from .linearity import linearity_report
from .pca import pca_denoise
from .visualize import (
    plot_correlation_heatmap,
    plot_pca_variance,
    plot_price_series,
    plot_returns,
)


def analyze_metrics(
    df: Optional[pd.DataFrame] = None,
    config: AnalyticsConfig = AnalyticsConfig(),
    save: bool = True,
) -> pd.DataFrame:
    if df is None:
        if not config.input_csv:
            raise ValueError("input_csv must be provided when df is None.")
        df = load_csv(config.input_csv)

    df = ensure_timestamp_datetime(df)
    df = sort_dedup(df)
    df = cast_numeric(df)
    df = add_return_features(df)
    df = winsorize(df, config.winsorize_limits)
    df = impute_missing(df, method=config.impute_method, limit=config.impute_limit)

    missing = validate_schema(df)
    if missing and config.debug:
        print(f"[DEBUG] Missing columns: {missing}")

    missing_report = missingness_report(df)

    features_df, feature_cols = prepare_features(df)
    denoised_df, pca_info = pca_denoise(
        features_df,
        variance_threshold=config.pca_variance_threshold,
    )

    lin_report = linearity_report(features_df)

    # Save artifacts
    if save:
        ensure_dir(config.output_dir)
        save_csv(df, config.output_dir, "metrics_cleaned.csv")
        save_csv(denoised_df, config.output_dir, "metrics_denoised.csv")
        save_csv(missing_report.reset_index().rename(columns={"index": "column"}), config.output_dir, "missingness_report.csv")

        save_csv(lin_report["pearson"].reset_index(), config.output_dir, "correlation_pearson.csv")
        save_csv(lin_report["spearman"].reset_index(), config.output_dir, "correlation_spearman.csv")
        save_csv(lin_report["vif"], config.output_dir, "vif_scores.csv")

        explained_ratio = pd.Series(
            pca_info.get("explained_ratio", []),
            name="explained_variance_ratio",
        )
        save_csv(explained_ratio.to_frame(), config.output_dir, "pca_explained_variance.csv")

        recon_error = pd.Series(
            pca_info.get("reconstruction_error", {}),
            name="reconstruction_mse",
        )
        save_csv(recon_error.to_frame().reset_index().rename(columns={"index": "feature"}), config.output_dir, "pca_reconstruction_error.csv")

        # Visuals
        plot_price_series(df, f"{config.output_dir}/price_series.png")
        plot_returns(df, f"{config.output_dir}/returns.png")
        if not lin_report["pearson"].empty:
            plot_correlation_heatmap(lin_report["pearson"], f"{config.output_dir}/correlation_heatmap.png")
        if len(explained_ratio) > 0:
            plot_pca_variance(explained_ratio, f"{config.output_dir}/pca_explained_variance.png")

        # Summary JSON
        summary = {
            "rows": int(len(df)),
            "features": feature_cols,
            "pca_components_used": int(pca_info.get("components_used", 0)),
        }
        save_json(summary, config.output_dir, "summary.json")

    return df
