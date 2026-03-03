"""
datasets_loader/main.py
=======================
Entry point: build the sliding-window dataset from the midway CSV and persist it.

Usage
-----
    # from project root
    python datasets_loader/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets_loader import DatasetConfig, EthDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = DatasetConfig(
        midway_csv="data/midway/eth_metrics_midway.csv",
        window_size=60,
        step_size=1,
        output_path="data/dataset_loader/eth_dataset.npz",
    )

    log.info("Building dataset from %s", cfg.midway_csv)
    dataset = EthDatasetLoader(cfg).build()
    log.info("Built: %s", dataset)

    dataset.save(cfg.output_path)
    log.info("Done.")
    


if __name__ == "__main__":
    main()
