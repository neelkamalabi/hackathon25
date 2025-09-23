import pandas as pd
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def calculate_iqr_bounds(df: pd.DataFrame, columns: List[str]) -> dict:
    """
    Calculates the IQR bounds (upper and lower) for specified columns in the DataFrame.
    :param df: DataFrame to calculate IQR bounds for
    :param columns: List of column names to calculate bounds for
    :return: Dictionary with column names as keys and (lower_bound, upper_bound) as values
    """
    # Check if all specified columns are present in the DataFrame
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"The following columns are not in the DataFrame: {missing_columns}"
        )

    bounds = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        bounds[col] = (lower_bound, upper_bound)

    return bounds


def cap_outliers(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    df_capped = df.copy()
    for col, (low, high) in bounds.items():
        df_capped[col] = df_capped[col].clip(lower=low, upper=high)
    return df_capped