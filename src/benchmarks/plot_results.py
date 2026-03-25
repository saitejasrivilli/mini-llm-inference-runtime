import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def plot_metric(df: pd.DataFrame, metric: str, title: str, out_path: str):
    ensure_dir("outputs/plots")
    plt.figure(figsize=(8, 4))
    plt.bar(df["mode"], df[metric])
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()