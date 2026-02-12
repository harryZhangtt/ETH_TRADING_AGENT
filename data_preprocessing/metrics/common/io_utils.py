from pathlib import Path
from typing import Optional

import pandas as pd


def ensure_dir(path: str) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_csv(df: pd.DataFrame, output_dir: str, filename: str) -> Path:
    dir_path = ensure_dir(output_dir)
    file_path = dir_path / filename
    df.to_csv(file_path, index=False)
    return file_path


def maybe_save_csv(
    df: pd.DataFrame,
    output_dir: str,
    filename: str,
    enabled: bool = True,
) -> Optional[Path]:
    if not enabled:
        return None
    return save_csv(df, output_dir, filename)
