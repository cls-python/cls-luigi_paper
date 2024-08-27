from os.path import join as pjoin
import pandas as pd
from utils.io_methods import load_json


def load_split_dataset(ds_name, root):
    path = pjoin(root, "binary_classfication_pipelines", "datasets", ds_name, "test_phase")

    x_train = pd.read_csv(pjoin(path, "x_train.csv"))
    x_test = pd.read_csv(pjoin(path, "x_test.csv"))
    y_train = pd.read_csv(pjoin(path, "y_train.csv"))
    y_test = pd.read_csv(pjoin(path, "y_test.csv"))

    return x_train, x_test, y_train, y_test


def get_task_seconds(ds_name, root, factor):
    path = pjoin(root, "binary_classfication_pipelines/logs/", f"{ds_name}_train_time.json")

    return int(load_json(path)["total_seconds"] * factor)


def get_best_askl_pipeline(pipeline_id, ds_name, seed, askl1=True):
    if askl1:
        run_history_path = pjoin(f"askl1_results/{ds_name}/smac3-output/run_{seed}/runhistory.json")
    else:
        run_history_path = pjoin(f"askl2_results/{ds_name}/smac3-output/run_{seed}/runhistory.json")

    run_history =load_json(run_history_path)
    best_pipeline_raw = run_history["configs"][str(pipeline_id - 1)]

    best_pipeline = {
        "id_in_leaderboard": pipeline_id,
        "id_in_run_history": pipeline_id - 1,
        "classifier": best_pipeline_raw["classifier:__choice__"],
        "feature_preprocessor": best_pipeline_raw["feature_preprocessor:__choice__"],
        "scaler": best_pipeline_raw["data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__"]
    }

    return best_pipeline
