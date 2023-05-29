from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams as z_rcParams
from .analysis_functions import between_labels, within_labels

def get_colors_lines():
    # Set colors and line styles
    colors = {
        "bl_ja": "red",
        "bl_info": "black",
        "bl_tp": "blue",
        "tr_ja": "red",
        "tr_info": "black",
        "tr_tp": "blue",
    }

    line_styles = {
        "bl_ja": "--",
        "bl_info": "--",
        "bl_tp": "--",
        "tr_ja": "-",
        "tr_info": "-",
        "tr_tp": "-",
    }
    return colors, line_styles

def set_params(large=False):
    z_rcParams["font.weight"] = "bold"
    z_rcParams["axes.labelweight"] = "bold"
    z_rcParams["axes.titleweight"] = "bold"
    z_rcParams["font.size"] = 15
    z_rcParams["font.family"] = "sans-serif"
    if large:
        z_rcParams["axes.labelsize"] = 34
        z_rcParams["axes.titlesize"] = 34
        z_rcParams["xtick.labelsize"] = 34
        z_rcParams["ytick.labelsize"] = 34

def line_style(column):
    if column.startswith("bl"):
        return "--"
    elif column.startswith("tr"):
        return "-"
    else:
        return None

def confusion_matrix(
    arr: np.ndarray | pd.DataFrame,
    labels: list | np.ndarray,
    title: Optional[str] = None,
    save_name: Optional[str] = "confusion_matrix.png",
    save_path: Optional[str] = "C:\\Users\\Flynn\\OneDrive\\Desktop",
    best=False,
) -> np.array:
    # TODO: check for whole integer float values
    if isinstance(arr, pd.DataFrame):
        arr = arr.values
    if isinstance(arr, np.ndarray):
        arr = arr.astype(int)

    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_subplot(111)

    lines = between_labels(labels)
    mid, ev = within_labels(labels)
    midloc = [i - 0.5 for i in mid]
    for i in lines:
        ax.hlines(y=i, xmin=0, xmax=arr.shape[0], color="white", linewidth=2)
        ax.vlines(x=i, ymin=0, ymax=arr.shape[0], color="white", linewidth=2)

    # TODO: Equation for when the sizes of the labels are different
    ax.hlines(y=6, xmin=0, xmax=arr.shape[0], color="white", linewidth=8)
    ax.vlines(x=6, ymin=0, ymax=arr.shape[0], color="white", linewidth=8)

    ax.set_title(title, pad=20)

    ax = sns.heatmap(
        arr,
        square=True,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=ev,
        yticklabels=ev,
        ax=ax,
        annot_kws={"size": 38},
    )

    ax.xaxis.set_ticks(midloc, ev)
    ax.yaxis.set_ticks(midloc, ev)

    plt.xlabel("true label", labelpad=20)
    plt.ylabel("predicted label", labelpad=20)

    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.tick_params(axis="x", which="both", length=0, labelrotation=0)
    ax.tick_params(axis="y", which="both", length=0)

    for i, row in enumerate(arr):
        count = sum(row)
        ax.text(arr.shape[1], i + 0.5, str(count), ha="left", va="center", fontsize=34)

    plt.tight_layout()
    if save_name:
        if best:
            plt.savefig(f"{save_path}_best.png", dpi=300)
        plt.savefig(f"{save_path}.png", dpi=300)
        print("Figure Saved.")
    else:
        plt.show()
    return ax

def plot_information(stat):
    # Plotting
    df = stat.drop(columns=["bl_acc", "tr_acc", "q"])
    x_label = stat["q"]

    colors, line_styles = get_colors_lines()
    fig, ax = plt.subplots(figsize=(20, 10))

    plt.xticks(stat["q"].index, stat["q"], rotation=45)
    plt.xlabel(
        "q-values", labelpad=10
    )  # Adjust the labelpad to increase the distance from the x-axis
    plt.ylabel("Information (h)", labelpad=15)
    # Plotting the lines
    for column in df.columns:
        line_style = line_styles[column]
        color = colors[column]
        ax.plot(df[column], linestyle=line_style, color=color, label=column)
    plt.title("Information as a function of q")

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], color="black", linestyle="--", label="Previous vs Current"),
        plt.Line2D([0], [0], color="black", linestyle="-", label="Current vs Previous"),
        plt.Line2D([0], [0], marker="s", color="red", label="Jacknife Bias Corrected", linestyle="None"),
        plt.Line2D([0], [0], marker="s", color="blue", label="Treves-Panzeri Corrected", linestyle="None"),
    ]

    ax.legend(handles=legend_elements, loc="best", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

def plot_cm(stat, labels, conf_bottom_left, conf_top_right, bestbl, besttr):
    # Plotting confusion matrices for each value of q
    for i in range(stat['q'].shape[0]):

        savepath = f'C:\\Users\\Flynn\\OneDrive\\Desktop\\temp\\confusion\\sfn\\LxL'
        q_data_bl = conf_bottom_left[:, :, i]
        q_data_tr = conf_top_right[:, :, i]
        bl_infobest = False
        tr_infobest = False
        if i == bestbl:
            bl_infobest = True
        if i == besttr:
            tr_infobest = True

        bl_conf_title = f"Yesterday x Today: \n" \
                        f" q = {np.round(stat['q'][i], 2)}" \
                        f" | h={np.round(stat['bl_info'][i], 2)} | ja={np.round(stat['bl_ja'][i], 2)} | tp={np.round(stat['bl_tp'][i], 2)}"

        tr_conf_title = f"Today x Yesterday: \n" \
                        f" q = {np.round(stat['q'][i], 2)} " \
                        f"| h={np.round(stat['tr_info'][i], 2)} | ja={np.round(stat['tr_ja'][i], 2)} | tp={np.round(stat['tr_tp'][i], 2)}"

        confusion_matrix(q_data_bl, labels, title=bl_conf_title, save_path=f'{savepath}\\bl\\bl_{i}', best=bl_infobest)
        confusion_matrix(q_data_tr, labels, title=tr_conf_title,
                         save_path=f'{savepath}\\tr\\tr_{np.round(stat["q"][i], 2)}', best=tr_infobest)
