import datetime
from typing import Optional, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bullish.analysis.functions import add_indicators
from datetime import date


def plot(
    data: pd.DataFrame,
    symbol: str,
    name: Optional[str] = None,
    dates: Optional[List[date]] = None,
    industry_data: Optional[pd.DataFrame] = None,
) -> go.Figure:
    data = add_indicators(data)
    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[
            [{"rowspan": 2}],  # Row 1: main chart
            [None],  # Row 2: skipped (part of row 1)
            [{}],  # Row 3: RSI
            [{}],  # Row 4: MACD
            [{}],  # Row 5: ADX
            [{}],  # Row 6: OBV
            [{}],  # Row 7: ATR
        ],
        subplot_titles=(
            f"Price + SMAs ({symbol} [{name}])",
            f"RSI ({symbol} [{name}])",
            f"MACD ({symbol} [{name}])",
            f"ADX ({symbol} [{name}])",
            f"OBV ({symbol} [{name}])",
            f"Industry ({symbol} [{name}])",
        ),
    )
    # Row 1: Candlestick + SMAs
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data.open,
            high=data.high,
            low=data.low,
            close=data.close,
            name="Candlestick",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(rangeslider_thickness=0.04, row=1, col=1)
    fig.add_trace(
        go.Scatter(x=data.index, y=data.SMA_50, name="SMA 50", mode="lines"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data.SMA_200, name="SMA 200", mode="lines"),
        row=1,
        col=1,
    )

    # Row 2: RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data.RSI, name="RSI 14", mode="lines"),
        row=3,
        col=1,
    )

    # Row 3: MACD
    fig.add_trace(
        go.Scatter(x=data.index, y=data.MACD_12_26_9, name="MACD", mode="lines"),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=data.index, y=data.MACD_12_26_9_SIGNAL, name="MACD Signal", mode="lines"
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=data.index, y=data.MACD_12_26_9_HIST, name="MACD Histogram", opacity=0.5
        ),
        row=4,
        col=1,
    )

    # Row 4: ADX
    fig.add_trace(
        go.Scatter(x=data.index, y=data.ADX_14, name="ADX_14", mode="lines"),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data.MINUS_DI, name="-DI", mode="lines"),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data.PLUS_DI, name="+DI", mode="lines"),
        row=5,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data.OBV, name="OBV", mode="lines"),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data.ADOSC, name="ADOSC", mode="lines"),
        row=6,
        col=1,
    )
    if industry_data is not None and not industry_data.empty:
        for c in industry_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=industry_data.index,
                    y=industry_data[c],
                    name=c,
                    mode="lines",
                    opacity=0.5 if c != "symbol" else 1.0,
                ),
                row=7,
                col=1,
            )

    if dates is not None and dates:
        for date in dates:
            if (
                data.first_valid_index().date() > date  # type: ignore
                or data.last_valid_index().date() + datetime.timedelta(days=31 * 3)  # type: ignore
                < date
            ):
                continue
            fig.add_vline(
                x=date,
                line_dash="dashdot",
                line_color="MediumPurple",
                line_width=1,
                row=1,
                col=1,
            )

    # Layout tweaks
    fig.update_layout(
        height=1500,
        showlegend=True,
        title="Technical Indicator Dashboard",
        margin={"t": 60, "b": 40},
    )

    # Optional: Add horizontal lines for RSI (e.g., 70/30 levels)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="red", row=5, col=1)

    return fig
