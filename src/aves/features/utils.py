import pandas as pd
import scattertext
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import minmax_scale, normalize, quantile_transform


def normalize_rows(df):
    return df.div(df.sum(axis=1), axis=0)


def normalize_columns(df):
    return normalize_rows(df.T).T


def standardize_columns(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)


def standardize_rows(df):
    return df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)


def minmax_columns(df):
    return pd.DataFrame(minmax_scale(df, axis=0), index=df.index, columns=df.columns)


def quantile_transform_columns(df, n_quantiles=10, output_distribution="uniform"):
    return pd.DataFrame(
        quantile_transform(
            df,
            axis=0,
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            copy=True,
        ),
        index=df.index,
        columns=df.columns,
    )


def tfidf(df, norm="l1", smooth_idf=False):
    return pd.DataFrame(
        TfidfTransformer(norm=norm, smooth_idf=smooth_idf).fit_transform(df).todense(),
        index=df.index,
        columns=df.columns,
    )


def weighted_mean(df, value_column, weighs_column):
    weighted_sum = (df[value_column] * df[weighs_column]).sum()
    return weighted_sum / df[weighs_column].sum()


def logodds_ratio_with_uninformative_dirichlet_prior(df, alpha_w=0.0001):
    scorer = scattertext.LogOddsRatioUninformativeDirichletPrior(alpha_w=alpha_w)

    top_words = {}

    frequencies = df.sum(axis=0)

    for idx, row in df.iterrows():
        positive = row
        negative = frequencies - positive
        group_scores = scorer.get_scores(positive, negative)
        top_words[idx] = group_scores

    return pd.DataFrame(top_words).T
