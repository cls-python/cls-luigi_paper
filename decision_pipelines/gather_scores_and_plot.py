import seaborn as sns
import pandas as pd
from os.path import join as pjoin
import json
import os


def draw_regret_facet_grid(summaries_df, save_to_path="scores_and_plots", title="",
                           out_name="regret_shortest_path.png"):
    sns.set_style("whitegrid")

    summaries_df = summaries_df.rename(columns={
        "degree": "Polynomial Degree",
        "regressor": "Regressor",
        "regret": "Normalized Regret",
        "noise": "Noise Half−width",
        "training_size": "Training Set Size"
    })

    g = sns.FacetGrid(summaries_df, col="Noise Half−width", row="Training Set Size", aspect=1.4)
    g.map_dataframe(sns.boxplot, x="Polynomial Degree", y="Normalized Regret", hue="Regressor",
                    palette=sns.color_palette("Set2")[:summaries_df["Regressor"].nunique()], fliersize=2, fill=True)
    g.add_legend()
    for ax in g.axes.flatten():
        ax.tick_params(labelbottom=True)

    g.set(ylim=(-0.01, 0.26))
    g.fig.suptitle(title)
    g.tight_layout()
    g.savefig(pjoin(save_to_path, out_name), dpi=500)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def collect_and_save_summaries(results_path="results", save_to_path="scores_and_plots", out_name="scores.csv"):
    summaries_dict = {
        "problem_name": [],
        # "optimizer": [],
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
                # summaries_dict["optimizer"].append(evaluation_file["optimizer"])
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

    return summaries_df


if __name__ == "__main__":
    summaries_df = collect_and_save_summaries()
    draw_regret_facet_grid(summaries_df, title="Regret for the Shortest Path Problems\n\n")
