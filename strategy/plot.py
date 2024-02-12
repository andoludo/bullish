from plotly.subplots import make_subplots
import plotly.graph_objects as go


def plot_strategy(data, strategies, name: str = "test"):
    fig = make_subplots(

        rows=2,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{}, {}],
        ],
    )
    for x in [
        "moving_average_5",
        "moving_average_20",
        "price",
    ]:
        fig.add_trace(go.Line(x=data.index, y=data[x], name=x), row=1, col=1)
    for s in strategies:
        for bs in ["buy", "sell"]:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[f"{bs}_{s.name}"],
                    name=f"{bs}_{s.name}",
                    mode="markers",
                    marker={"size": 10, "color": {"buy": "green", "sell": "red"}[bs]},
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[f"percentage_{s.name}"],
                mode="markers",
                marker={"size": 5},
                name=s.name,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Box(y=data[f"percentage_{s.name}"], name=s.name),
            row=2,
            col=2,
        )
    fig.update_layout(title_text=name)
    fig.show()
