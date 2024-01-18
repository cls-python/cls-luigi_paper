import pandas as pd
from os.path import join as pjoin
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def collect_and_save_summaries(results_path="results", save_to_path="scores_and_plots", out_name="scores.csv"):
    summaries_dict = {
        "problem_name": [],
        "regressor": [],
        "training_size": [],
        "degree": [],
        "noise": [],
        "seed": [],
        "regret": [],
        "mse": []
    }

    for r in os.listdir(results_path):
        problem_name, training_size, degree, noise, seed = r.split("-")

        sub_dir = pjoin(results_path, r)

        for file in os.listdir(sub_dir):
            if file.endswith("json") and "Evaluation" in file:
                evaluation_file = load_json(pjoin(sub_dir, file))

                summaries_dict["problem_name"].append(problem_name)
                summaries_dict["regressor"].append(evaluation_file["regressor"])
                summaries_dict["training_size"].append(int(training_size.split("_")[-1]))
                summaries_dict["degree"].append(int(degree.split("_")[-1]))
                summaries_dict["noise"].append(float(noise.split("_")[-1]))
                summaries_dict["seed"].append(int(seed.split("_")[-1]))
                summaries_dict["regret"].append(evaluation_file["regret"])
                summaries_dict["mse"].append(evaluation_file["mse"])

    summaries_df = pd.DataFrame.from_dict(summaries_dict)

    os.makedirs(save_to_path, exist_ok=True)

    summaries_df.to_csv(pjoin(save_to_path, out_name), index=False)


def legend_name(regressor):
    if regressor == "LinearRegressionModel":
        return "LR"
    elif regressor == "RandomForestModel":
        return "RF"
    elif regressor == "SPOPlus":
        return "SPO+"
    elif regressor == "LightGBMModel":
        return "LGBM"
    elif regressor == "LightGBMModelLinearTree":
        return "LGBM (LT)"


def draw_comparison_boxplot(summaries_df, training_sizes=[100, 1000, 5000], noises=[0.0, 0.5], save_to_path="scores_and_plots", out_name="comparison_boxplot.png", show=True):
    boxes, labels = [], []

    fig, axis = plt.subplots(3, 2, figsize=(26, 18))
    axis = axis.flatten()

    models_and_colors = {
        "LinearRegressionModel": "#57C2A3",
        "RandomForestModel": "#FFB2C2",
        "SPOPlus": "#F08D3C",
        "LightGBMModel": "#A13687",
        "LightGBMModelLinearTree": "#568AC5"}

    dfs = []

    for ts in training_sizes:
        for n in noises:
            sliced_df = summaries_df[
                (summaries_df["training_size"] == ts) &
                (summaries_df["noise"] == n)]
            dfs.append(sliced_df)

    for ix, df in enumerate(dfs):
        y_pos_factor = -.285
        for model, color in models_and_colors.items():
            pivoted = df[df["regressor"] == model].pivot(index="seed", columns="degree", values="regret")

            bp = axis[ix].boxplot(
                pivoted,
                boxprops=dict(facecolor=color, color=color, linewidth=4),
                medianprops=dict(color="w", linewidth=2),
                whiskerprops=dict(color=color, linewidth=2), capprops=dict(color=color, linewidth=2),
                flierprops=dict(markeredgecolor=color, marker="o", markersize=5, markeredgewidth=2),
                patch_artist=True,
                positions=np.arange(pivoted.shape[1]) + y_pos_factor,
                widths=0.11
            )

            boxes.append(bp["boxes"][0])
            if ix == 0:
                labels.append(legend_name(model))
            y_pos_factor += 0.14

        # grid
        axis[ix].grid(color="grey", alpha=0.5, linewidth=0.5, which="major", axis="y")

        # axis labels
        if ix % 2 == 0:
            axis[ix].set_ylabel('Normalized Regret', fontsize=24)

        if (ix == 4) or (ix == 5):
            axis[ix].set_xlabel('Polynomial Degree', fontsize=24)

        # x ticks
        axis[ix].set_xticks(ticks=[0, 1, 2, 3], labels=[1, 2, 4, 6], fontsize=20)

        # y ticks
        axis[ix].set_xlim(-0.5, 3.5)
        y_ticks = [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
        axis[ix].set_yticks(ticks=y_ticks, labels=y_ticks, fontsize=20)
        axis[ix].set_ylim(-0.02, 0.29)
        axis[ix].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # subplot title
        axis[ix].set_title(f"Training Set Size = {ts}, Noise Half−width = {n}", fontsize=24)

        # legend
        leg = axis[ix].legend(boxes, labels, fontsize=22, loc=2, labelspacing=0.2, handlelength=1, ncol=3, )
        leg.get_frame().set_linewidth(0.0)
        leg.get_frame().set_facecolor('none')

        # vertical lines
        axis[ix].axvline(x=.5, color="k", linestyle="--", linewidth=1.5)
        axis[ix].axvline(x=1.5, color="k", linestyle="--", linewidth=1.5)
        axis[ix].axvline(x=2.5, color="k", linestyle="--", linewidth=1.5)

    plt.suptitle("Shortest Path Regret Comparison\n\n", fontsize=30)
    plt.tight_layout()
    if show:
        plt.show()
    fig.savefig(pjoin(save_to_path, out_name), dpi=400)


if __name__ == "__main__":
    summaries_df = collect_and_save_summaries()
    # summaries_df = pd.read_csv("scores_and_plots/scores.csv")
    draw_comparison_boxplot(summaries_df)
