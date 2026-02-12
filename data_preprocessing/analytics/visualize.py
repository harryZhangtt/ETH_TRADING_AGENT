from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_series(df: pd.DataFrame, output_path: str) -> None:
    if df.empty or "timestamp" not in df.columns or "close" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp"], df["close"], label="ETH Close")
    if "btc_close" in df.columns:
        ax.plot(df["timestamp"], df["btc_close"], label="BTC Close", alpha=0.6)
    ax.set_title("ETH/BTC Close Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_returns(df: pd.DataFrame, output_path: str) -> None:
    if df.empty or "timestamp" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    if "eth_return" in df.columns:
        ax.plot(df["timestamp"], df["eth_return"], label="ETH Log Return")
    if "btc_return" in df.columns:
        ax.plot(df["timestamp"], df["btc_return"], label="BTC Log Return", alpha=0.6)
    ax.set_title("Log Returns")
    ax.set_xlabel("Time")
    ax.set_ylabel("Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, output_path: str) -> None:
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr.values, interpolation="nearest", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    fig.colorbar(cax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pca_variance(explained_ratio: pd.Series, output_path: str) -> None:
    if explained_ratio.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(explained_ratio) + 1), explained_ratio.values, marker="o")
    ax.set_title("PCA Explained Variance Ratio")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained Variance Ratio")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
