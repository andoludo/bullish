import pandas as pd


def _get_min_low(data):
    return data.low == data.low.min()


def _get_max_high(data):
    return data.high == data.high.max()


def _get_low_values(data):
    return data.low.values


def _get_high_values(data):
    return data.high.values


def resistance(data: pd.DataFrame):
    return _line(data, _get_max_high, _get_min_low, _get_high_values, "Resistance")


def support(data: pd.DataFrame):
    return _line(data, _get_min_low, _get_max_high, _get_low_values, "Support")


def _line(data, get_extreme, get_opposite, get_values, name):
    data_extreme = data.where(get_extreme(data)).dropna()
    extreme_index = data_extreme.index
    data_from_extreme = data.loc[extreme_index[0] :]
    data_to_opposite = data_from_extreme.where(get_opposite(data_from_extreme)).dropna()

    base_data = data_from_extreme.loc[extreme_index[0] : data_to_opposite.index[0]]
    ys = get_values(base_data)
    xs = range(len(base_data.index))
    xs_ = xs[1:]
    ys_ = ys[1:]
    slopes = [(y - ys[0]) / (x - xs[0]) for x, y in zip(xs_, ys_)]
    if not slopes:
        return

    min_slope = max(slopes)
    min_slope_index = slopes.index(min_slope)
    intercept = ys_[min_slope_index] - min_slope * xs_[min_slope_index]
    length = (
        4 * len(data_from_extreme)
        if len(data_from_extreme) < 4
        else 6 * len(data_from_extreme)
    )
    return (
        min_slope,
        intercept,
        pd.DataFrame(
            [min_slope * x + intercept for x in range(length)],
            index=pd.date_range(
                start=extreme_index[0],
                end=extreme_index[0] + pd.Timedelta(days=length - 1),
            ),
            columns=[name],
        ),
    )
