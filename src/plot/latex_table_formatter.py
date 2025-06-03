import pandas as pd


def highlight_and_color_cell(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns[1:]:
        col_vals = df[col]
        min_val = col_vals.min()
        sorted_vals = col_vals.sort_values().to_list()
        # median_val = col_vals.median()

        def format_cell(val):
            # val_int = int(round(val, 0))
            print_val = round(val, 1)
            color = "green!15" if val in sorted_vals[:10] else "red!15"
            # color = "green!15" if val <= median_val else "red!15"
            cell = rf"\cellcolor{{{color}}}"
            if val == min_val:
                return cell + rf"\underline{{\textbf{{{print_val}}}}}"
            else:
                return cell + str(print_val)

        df[col] = col_vals.apply(format_cell)

    # Ensure 'method' stays unchanged
    df["method"] = df["method"]

    return df


def delta_kappa_format(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns[1:]:
        col_vals = df[col] * 100
        _max = col_vals.max()
        # sorted_vals = col_vals.sort_values().to_list()
        # median_val = col_vals.median()

        def format_cell(val):
            color = "green!15" if val >= 0 else "red!15"
            cell = rf"\cellcolor{{{color}}}"
            if val == _max:
                return cell + rf"\underline{{\textbf{{{round(val, 1)}}}}}"
            else:
                return cell + str(round(val, 1))

        df[col] = col_vals.apply(format_cell)

    # Ensure 'method' stays unchanged
    df["method"] = df["method"]
    return df
