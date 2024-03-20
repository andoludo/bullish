from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_strategy(data, strategies, name: str = "test", show:bool=True):
    fig = make_subplots(

        rows=2,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
        ],
    )
    for x in [
        "movingaverage_5",
        "movingaverage_10",
        "movingaverage_4",
        "movingaverage_18",
        "movingaverage_20",
        "movingaverage_200",
        "price",
        "support",
        "resistance"
    ]:
        if x not in data:
            continue
        fig.add_trace(go.Line(x=data.index, y=data[x], name=x), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data[f"rsi_14"],
            name="rsi",
        ),
        row=2,
        col=1,
    )
    for s in strategies:
        for bs in ["buy", "sell"]:
            y_name = f"{bs}_{s.name}"
            if y_name not in data:
                continue
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[y_name],
                    name=y_name,
                    mode="markers",
                    marker={"size": 10, "color": {"buy": "green", "sell": "red"}[bs]},
                ),
                row=1,
                col=1,
            )
        if f"percentage_{s.name}" not in data:
            continue
        fig.add_trace(
            go.Box(y=data[f"percentage_{s.name}"], name=s.name),
            row=2,
            col=2,
        )
    fig.update_layout(title_text=name)
    if show:
        fig.show()
    return fig.to_json()
