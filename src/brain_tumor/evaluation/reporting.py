"""Chart rendering for comparison results, decoupled from Streamlit.

The original ``compare_model`` called ``st.pyplot`` directly, which meant the
comparison logic could not run outside the Streamlit app. Here the plots are
returned as plain :class:`matplotlib.figure.Figure` objects that callers can
pass to ``st.pyplot(fig)``, save with ``fig.savefig(...)``, or embed in a
report.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["model_name"], df[metric], color="skyblue")
    ax.set_title(f"So sanh {metric} giua cac mo hinh")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05 * df[metric].max())
    for i, value in enumerate(df[metric]):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_all_metrics(df: pd.DataFrame) -> list[plt.Figure]:
    metrics = [col for col in df.columns if col != "model_name"]
    return [plot_metric(df, metric) for metric in metrics if df[metric].notnull().any()]
