import numpy as np
import pandas as pd


def intersection(first_curve: pd.Series, second_curve: pd.Series, strategy: str):
    sign = f"sign_{strategy}"
    name = f"intersection_{strategy}"
    if not first_curve.name:
        first_curve.name = name
    curve_difference = np.diff(np.sign(first_curve - second_curve))
    difference_sign = np.sign(curve_difference)
    data = pd.concat(
        [
            pd.DataFrame(
                first_curve.iloc[np.argwhere(curve_difference).flatten()],
                index=first_curve.index,
            ),
            pd.DataFrame(difference_sign, index=first_curve.index[:-1], columns=[sign]),
        ],
        axis=1,
    )
    data = data.rename(columns={first_curve.name: name})
    return data


#
def difference(data: pd.DataFrame, strategy_name: str):
    sign_name = f"sign_{strategy_name}"
    intersection_name = f"intersection_{strategy_name}"
    gain = (
        data[[intersection_name, sign_name]]
        .dropna()
        .diff()
        .rename(
            columns={
                intersection_name: f"gain_{strategy_name}",
                sign_name: f"type_{strategy_name}",
            }
        )
    )
    percentage_data = data[[intersection_name]].dropna()
    if not percentage_data.empty:
        percentage = (percentage_data.pct_change() * 100).rename(
            columns={intersection_name: f"percentage_{strategy_name}"}
        )
    else:
        percentage = pd.DataFrame(
            columns=[f"percentage_{strategy_name}"], index=gain.index
        )
    data_ = pd.concat([gain, percentage], axis=1)
    data_ = data_.where(data_[f"type_{strategy_name}"] == -2)
    return data_


def rsi(data, window=14):
    close_diff = data.price.diff()
    up_days = (
        close_diff.where(close_diff > 0).rolling(window=window, min_periods=1).mean()
    )
    down_days = (
        -close_diff.where(close_diff < 0).rolling(window=window, min_periods=1).mean()
    )

    rs = up_days / down_days
    rsi = 100 - (100 / (1 + rs))
    return rsi
