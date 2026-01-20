"""
Charts Module

Pure Plotly functions for creating dashboard charts.
No Streamlit dependencies - these functions are testable and reusable.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_rul_chart(df):
    """
    Create a chart showing RUL trend over time.

    Args:
        df: DataFrame with 'cycle' and 'rul_prediction' columns.

    Returns:
        Plotly Figure with RUL trend line.
    """
    fig = go.Figure()

    # Add RUL prediction line
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["rul_prediction"],
        mode="lines+markers",
        name="Predicted RUL",
        line=dict(color="firebrick", width=2),
        marker=dict(size=4)
    ))

    # Configure layout
    fig.update_layout(
        title="Remaining Useful Life (RUL) Trend",
        xaxis_title="Operating Cycles",
        yaxis_title="RUL (Cycles)",
        template="plotly_white",
        height=350,
        showlegend=True,
        hovermode="x unified"
    )

    # Add gridlines
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")

    return fig


def create_sensor_grid(df, sensors_to_plot=None, cols=4):
    """
    Create a grid of charts for sensor data.

    Args:
        df: DataFrame with sensor readings and 'cycle' column.
        sensors_to_plot: List of sensor names to plot. If None, plot all.
        cols: Number of columns in the grid.

    Returns:
        Plotly Figure with sensor grid.
    """
    # Handle empty data
    if df is None or df.empty:
        return go.Figure()

    # Determine which sensors to plot
    if sensors_to_plot is None:
        excluded = {"cycle", "rul_prediction"}
        sensors_to_plot = [c for c in df.columns if c not in excluded]

    # Filter to sensors that exist in dataframe
    sensors_to_plot = [s for s in sensors_to_plot if s in df.columns]
    if not sensors_to_plot:
        return go.Figure()

    # Calculate grid dimensions
    cols = max(1, int(cols))
    rows = (len(sensors_to_plot) + cols - 1) // cols

    # Create subplot grid
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=sensors_to_plot,
        vertical_spacing=0.12
    )

    # Add a trace for each sensor
    for i, sensor_name in enumerate(sensors_to_plot):
        row = (i // cols) + 1
        col = (i % cols) + 1

        fig.add_trace(
            go.Scatter(
                x=df["cycle"],
                y=df[sensor_name],
                mode="lines",
                name=sensor_name,
                line=dict(width=1.5),
            ),
            row=row,
            col=col,
        )

        # Update axes
        fig.update_xaxes(title_text="Cycles", row=row, col=col, showgrid=True)
        fig.update_yaxes(showgrid=True, row=row, col=col)

    # Update layout
    fig.update_layout(
        height=max(350, 220 * rows),
        showlegend=False,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig
