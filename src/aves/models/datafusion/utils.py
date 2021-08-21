import pandas as pd
import numpy as np


def ensure_columns(df, columns, fill_value=np.nan):
    df = df.copy()

    if isinstance(columns, pd.Series):
        values = columns.values
    elif isinstance(columns, pd.DataFrame):
        values = columns.columns.values
    else:
        values = list(columns)

    for col in values:
        if not col in df.columns:
            df[col] = fill_value

    df = df[values].copy()
    return df


def ensure_index(df, index):
    if isinstance(index, pd.Series):
        values = index.values
    elif isinstance(index, pd.DataFrame):
        values = index.index.values
    else:
        values = list(index)

    return pd.DataFrame(index=values).join(df, how="left").copy()
