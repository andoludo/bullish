import pandas as pd
import plotly.graph_objects as go


def _get_min_low(data):
    return data.low == data.low.min()


def _get_max_high(data):
    return data.high == data.high.max()


def _get_low_values(data):
    return data.low.values


def _get_high_values(data):
    return data.high.values


def resistance(data: pd.DataFrame, max_date):
    return _line(
        data, _get_max_high, _get_min_low, _get_high_values, max, "resistance", max_date
    )


def support(data: pd.DataFrame, max_date):
    return _line(
        data, _get_min_low, _get_max_high, _get_low_values, min, "support", max_date
    )


def _line(data, get_extreme, get_opposite, get_values, slope_function, name, max_date):
    data_extreme = data.where(get_extreme(data)).dropna()
    extreme_index = data_extreme.index
    data_from_extreme = data.loc[extreme_index[0] :]
    data_to_opposite = data_from_extreme.where(get_opposite(data_from_extreme)).dropna()
    base_data = data_from_extreme.loc[extreme_index[0] : data_to_opposite.index[0]]
    ys = get_values(base_data)
    xs = base_data.index.astype("int64") // 10 ** 9
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
                for x in pd.DatetimeIndex(list(base_data.index) + [max_date]).astype(
                    "int64"
                )
                // 10 ** 9
            ],
            index=pd.DatetimeIndex(list(base_data.index) + [max_date]),
            columns=[name],
        ),
    )

def plot_support(fig, data):
    plot_trend(data, fig, support,"support", "black")


def plot_resistance(fig, data):
    plot_trend(data, fig, resistance,"resistance", "red")

def plot_trend(data, fig, trend_function, name, color):
    lines = set()
    for i in range(len(data.index)):
        res = trend_function(data[: i + 1], data.index[-1])
        if res is None:
            continue
        (min_slope, intercept, support_) = res
        if (min_slope, intercept) not in lines:
            lines.add((min_slope, intercept))
        else:
            continue
        if support_ is not None:
            fig.add_trace(
                go.Scatter(
                    x=support_.index,
                    y=support_[name],
                    mode="lines",
                    marker=dict(color=color),
                    # opacity=0.6,
                    line={"width": 0.4}

                )
            )


def support_line(data: pd.DataFrame):
    data_lowest = data.where(data.low == data.low.min()).dropna()
    lowest_index = data_lowest.index
    data_from_lowest = data.loc[lowest_index[0] :]
    data_to_max = data_from_lowest.where(
        data_from_lowest.high == data_from_lowest.high.max()
    ).dropna()
    base_data = data_from_lowest.loc[lowest_index[0] : data_to_max.index[0]]
    ys = base_data.low.values
    xs = range(len(base_data.index))
    xs_ = xs[1:]
    ys_ = ys[1:]
    slopes = [(y - ys[0]) / (x - xs[0]) for x, y in zip(xs_, ys_)]
    if not slopes:
        return

    min_slope = min(slopes)
    min_slope_index = slopes.index(min_slope)
    intercept = ys_[min_slope_index] - min_slope * xs_[min_slope_index]
    length = (
        4 * len(data_from_lowest)
        if len(data_from_lowest) < 4
        else 6 * len(data_from_lowest)
    )
    return (
        min_slope,
        intercept,
        pd.DataFrame(
            [min_slope * x + intercept for x in range(length)],
            index=pd.date_range(
                start=lowest_index[0],
                end=lowest_index[0] + pd.Timedelta(days=length - 1),
            ),
            columns=["support"],
        ),
    )
