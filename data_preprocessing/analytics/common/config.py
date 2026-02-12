from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AnalyticsConfig:
    input_csv: Optional[str] = None
    output_dir: str = "data_preprocessing/data/analytics"
    pca_variance_threshold: float = 0.95
    winsorize_limits: float = 0.01
    impute_method: str = "ffill"
    impute_limit: Optional[int] = None
    debug: bool = False
