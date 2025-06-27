import pandas as pd

from bullish.figures.figures import plot


def test_plot(data_aapl: pd.DataFrame) -> None:
    fig = plot(
        data_aapl,
        "AAPL",
    )
    fig.show()
