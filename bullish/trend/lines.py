from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from numpy import ndarray
from pandas.core.arrays import ExtensionArray


def _get_min_low(data: pd.DataFrame) -> Any:  # noqa : ANN401
    return data.low == data.low.min()


def _get_max_high(data: pd.DataFrame) -> Any:  # noqa : ANN401
    return data.high == data.high.max()


def _get_low_values(data: pd.DataFrame) -> ExtensionArray | ndarray[Any, Any]:
    return data.low.to_numpy()


def _get_high_values(data: pd.DataFrame) -> ExtensionArray | ndarray[Any, Any]:
    return data.high.to_numpy()


def resistance(
    data: pd.DataFrame, max_date: int
) -> Tuple[Optional[float], Optional[float], pd.DataFrame]:
    return _line(
        data, _get_max_high, _get_min_low, _get_high_values, max, "resistance", max_date
    )


def support(
    data: pd.DataFrame, max_date: int
) -> Tuple[Optional[float], Optional[float], pd.DataFrame]:
    return _line(
        data, _get_min_low, _get_max_high, _get_low_values, min, "support", max_date
    )


def _line(  # noqa : PLR0913
    data: pd.DataFrame,
    get_extreme: Callable[[pd.DataFrame], "pd.Series[float]"],
    get_opposite: Callable[[pd.DataFrame], "pd.Series[float]"],
    get_values: Callable[[pd.DataFrame], ExtensionArray | ndarray[Any, Any]],
    slope_function: Callable[[List[float]], float],
    name: str,
    max_date: int,
) -> Tuple[Optional[float], Optional[float], pd.DataFrame]:
    data_extreme = data.where(get_extreme(data)).dropna()
    extreme_index = data_extreme.index
    data_from_extreme = data.loc[extreme_index[0] :]
    data_to_opposite = data_from_extreme.where(get_opposite(data_from_extreme)).dropna()
    base_data = data_from_extreme.loc[extreme_index[0] : data_to_opposite.index[0]]
    ys = get_values(base_data)
    xs = base_data.index.astype("int64") // 10**9
    xs_ = xs[1:]
    ys_ = ys[1:]
    slopes = [(y - ys[0]) / (x - xs[0]) for x, y in zip(xs_, ys_)]
    if not slopes:
        return (None, None, pd.DataFrame())

    _slope = slope_function(slopes)
    min_slope_index = slopes.index(_slope)
    intercept = ys_[min_slope_index] - _slope * xs_[min_slope_index]
    return (
        _slope,
        intercept,
        pd.DataFrame(
            [
                _slope * x + intercept
                for x in pd.DatetimeIndex([*list(base_data.index), max_date]).astype(
                    "int64"
                )
                // 10**9
            ],
            index=pd.DatetimeIndex([*list(base_data.index), max_date]),
            columns=[name],
        ),
    )
