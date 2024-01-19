import pandas as pd
from os import listdir, makedirs
from os.path import join as pjoin
import json
from cls_luigi import RESULTS_PATH

OUTDIR = "run_histories"


def generate_and_save_run_history(ds_name, results_dir=RESULTS_PATH, out_dir=OUTDIR, metric="balanced_accuracy"):
    makedirs(out_dir, exist_ok=True)

    imputer_col = []
    scaler_col = []
    feature_preprocessor_col = []
    classifier_col = []
    train_accuracy = []
    test_accuracy = []
    train_balanced_accuracy = []
    test_balanced_accuracy = []
    last_task_col = []
    status = []

    dataset_dir = pjoin(results_dir, ds_name)
    for file in listdir(dataset_dir):

        if file.endswith("json"):
            if "run_summary" in file or "FAILURE" in file or "TIMEOUT" in file:

                result_json = load_json(pjoin(dataset_dir, file))

                if "run_summary" in file:
                    status.append("success")
                    pipeline = result_json["pipeline"]
                    imputer_col.append(pipeline["imputer"])
                    scaler_col.append(pipeline["scaler"])
                    feature_preprocessor_col.append(pipeline["feature_preprocessor"])
                    classifier_col.append(pipeline["classifier"])
                    train_accuracy.append(result_json["accuracy"]["train"])
                    test_accuracy.append(result_json["accuracy"]["test"])
                    train_balanced_accuracy.append(result_json["balanced_accuracy"]["train"])
                    test_balanced_accuracy.append(result_json["balanced_accuracy"]["test"])
                    last_task_col.append(result_json["last_task"])

                elif "FAILURE" in file:
                    last_task_col.append(result_json["task_id"])
                    status.append("failed")
                    imputer_col.append(None)
                    scaler_col.append(None)
                    feature_preprocessor_col.append(None)
                    classifier_col.append(None)
                    train_accuracy.append(None)
                    test_accuracy.append(None)
                    train_balanced_accuracy.append(None)
                    test_balanced_accuracy.append(None)

                elif "TIMEOUT" in file:
                    last_task_col.append(result_json["task_id"])
                    status.append("timeout")
                    imputer_col.append(None)
                    scaler_col.append(None)
                    feature_preprocessor_col.append(None)
                    classifier_col.append(None)
                    train_accuracy.append(None)
                    test_accuracy.append(None)
                    train_balanced_accuracy.append(None)
                    test_balanced_accuracy.append(None)

    run_history_df = pd.DataFrame()

    run_history_df["imputer"] = imputer_col
    run_history_df["scaler"] = scaler_col
    run_history_df["feature_preprocessor"] = feature_preprocessor_col
    run_history_df["classifier"] = classifier_col
    run_history_df["train_accuracy"] = train_accuracy
    run_history_df["test_accuracy"] = test_accuracy
    run_history_df["train_balanced_accuracy"] = train_balanced_accuracy
    run_history_df["test_balanced_accuracy"] = test_balanced_accuracy
    run_history_df["last_task"] = last_task_col
    run_history_df["status"] = status

    run_history_df.sort_values(by=f"test_{metric}", ascending=False, inplace=True)
    run_history_df.to_csv(pjoin(out_dir, f"{ds_name}_train_run_history.csv"), index=False)

    return run_history_df


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def train_summary_stats_str(lb_df):
    stats_string = "\n========== Training Stats ==========\n"
    stats_string += f"Number of runs: {len(lb_df)}\n"
    stats_string += f"Successful runs: {len(lb_df[lb_df['status'] == 'success'])}\n"
    stats_string += f"Failed runs: {len(lb_df[lb_df['status'] == 'failed'])}\n"
    stats_string += f"Timeout runs: {len(lb_df[lb_df['status'] == 'timeout'])}\n"
    stats_string += f"Best pipeline components: "
    stats_string += f"Imputer: {lb_df.iloc[0]['imputer']}, "
    stats_string += f"Scaler: {lb_df.iloc[0]['scaler']}, "
    stats_string += f"Feature Preprocessor: {lb_df.iloc[0]['feature_preprocessor']}, "
    stats_string += f"Classifier: {lb_df.iloc[0]['classifier']}\n"
    stats_string += f"Validation balanced accuracy: {lb_df.iloc[0]['test_balanced_accuracy']:.2f}\n"
    stats_string += f"Validation accuracy: {lb_df.iloc[0]['test_accuracy']:.2f}\n"
    stats_string += "===================================\n"
    return stats_string


def test_summary_stats_str(ds_name, results_dir=RESULTS_PATH):
    stats_string = "\n========== Testing Stats ==========\n"

    for file in listdir(pjoin(results_dir, ds_name + "_incumbent")):
        if file.endswith("json") and "run_summary" in file:
            result_json = load_json(pjoin(results_dir, ds_name, file))
            stats_string += f"Train accuracy: {result_json['accuracy']['train']:.2f}\n"
            stats_string += f"Test accuracy: {result_json['accuracy']['test']:.2f}\n"

            stats_string += f"Train balanced accuracy: {result_json['balanced_accuracy']['train']:.2f}\n"
            stats_string += f"Test balanced accuracy: {result_json['balanced_accuracy']['test']:.2f}\n"

            return stats_string
