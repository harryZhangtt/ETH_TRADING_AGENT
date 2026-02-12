from typing import Dict

import numpy as np
import pandas as pd


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return df.corr(method=method)


def vif_scores(df: pd.DataFrame) -> pd.DataFrame:
    # VIF via correlation matrix inversion; robust to collinearity using pinv.
    if df.empty:
        return pd.DataFrame(columns=["feature", "vif"])

    corr = df.corr().fillna(0.0).to_numpy()
    inv_corr = np.linalg.pinv(corr)
    vifs = np.diag(inv_corr)
    return pd.DataFrame({"feature": df.columns, "vif": vifs})


def linearity_report(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {
        "pearson": correlation_matrix(df, method="pearson"),
        "spearman": correlation_matrix(df, method="spearman"),
        "vif": vif_scores(df),
    }
