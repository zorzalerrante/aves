import pandas as pd
from sklearn.preprocessing import normalize, minmax_scale, quantile_transform
from sklearn.feature_extraction.text import TfidfTransformer

def normalize_rows(df):
    df = pd.DataFrame(normalize(df, norm='l1'), index=df.index, columns=df.columns)
    return df

def normalize_columns(df):
    df = pd.DataFrame(normalize(df, norm='l1', axis=0), index=df.index, columns=df.columns)
    return df

def standardize_columns(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

def minmax_columns(df):
    return pd.DataFrame(minmax_scale(df, axis=0), index=df.index, columns=df.columns)

def quantile_transform_columns(df, n_quantiles=10, output_distribution='uniform'):
    return pd.DataFrame(quantile_transform(df, axis=0, n_quantiles=n_quantiles, output_distribution=output_distribution, copy=True), index=df.index, columns=df.columns)

def tfidf(df, norm='l1', smooth_idf=True):
    return pd.DataFrame(TfidfTransformer(norm=norm, smooth_idf=smooth_idf).fit_transform(df).todense(), index=df.index, columns=df.columns)