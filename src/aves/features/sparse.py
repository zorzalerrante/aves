from scipy.sparse import dok_matrix
import pandas as pd
from cytoolz import itemmap


def long_dataframe_to_sparse_matrix(
    df, index, vars, values, id_to_row=None, var_to_column=None
):
    if id_to_row is None:
        unique_index_values = df[index].unique()
        id_to_row = dict(zip(unique_index_values, range(len(unique_index_values))))

    n_rows = len(id_to_row)

    if var_to_column is None:
        unique_vars = df[vars].unique()
        var_to_column = dict(zip(unique_vars, range(len(unique_vars))))

    n_cols = len(var_to_column)

    dtm = dok_matrix((n_rows, n_cols), dtype=df[values].dtype)

    for i, tup in enumerate(df.itertuples()):
        elem_row, elem_col, elem_val = (
            getattr(tup, index),
            getattr(tup, vars),
            getattr(tup, values),
        )
        if elem_row in id_to_row:
            row_id = id_to_row[elem_row]
        else:
            continue

        if elem_col in var_to_column:
            col_id = var_to_column[elem_col]
        else:
            continue

        dtm[row_id, col_id] = elem_val

    return dtm.tocsr(), id_to_row, var_to_column


def sparse_matrix_to_long_dataframe(
    matrix,
    index_name="index",
    var_name="column",
    value_name="value",
    index_map=None,
    var_map=None,
    reverse_maps=False,
):

    matrix = matrix.todok()
    df = pd.DataFrame.from_records(
        list(map(lambda x: (x[0][0], x[0][1], x[1]), matrix.items()))
    )

    df.columns = [index_name, var_name, value_name]

    if index_map:
        if reverse_maps:
            index_map = itemmap(reversed, index_map)
        df[index_name] = df[index_name].map(index_map)

    if var_map:
        if reverse_maps:
            var_map = itemmap(reversed, var_map)
        df[var_name] = df[var_name].map(var_map)

    return df
