import pandas as pd
from os import listdir, makedirs
from os.path import join as pjoin
from cls_luigi import RESULTS_PATH

OUTDIR = "run_histories"

from utils.io_methods import load_json, dump_json


def generate_and_save_run_history(ds_name, results_dir=RESULTS_PATH, out_dir=OUTDIR, sort_by_metric="accuracy"):
    makedirs(out_dir, exist_ok=True)

    imputer_col = []
    scaler_col = []
    feature_preprocessor_col = []
    classifier_col = []
    train_accuracy = []
    valid_accuracy = []
    train_balanced_accuracy = []
    valid_balanced_accuracy = []
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
                    valid_accuracy.append(result_json["accuracy"]["test"])
                    train_balanced_accuracy.append(result_json["balanced_accuracy"]["train"])
                    valid_balanced_accuracy.append(result_json["balanced_accuracy"]["test"])
                    last_task_col.append(result_json["last_task"])

                elif "FAILURE" in file:
                    last_task_col.append(result_json["task_id"])
                    status.append("failed")
                    imputer_col.append(None)
                    scaler_col.append(None)
                    feature_preprocessor_col.append(None)
                    classifier_col.append(None)
                    train_accuracy.append(None)
                    valid_accuracy.append(None)
                    train_balanced_accuracy.append(None)
                    valid_balanced_accuracy.append(None)

                elif "TIMEOUT" in file:
                    last_task_col.append(result_json["task_id"])
                    status.append("timeout")
                    imputer_col.append(None)
                    scaler_col.append(None)
                    feature_preprocessor_col.append(None)
                    classifier_col.append(None)
                    train_accuracy.append(None)
                    valid_accuracy.append(None)
                    train_balanced_accuracy.append(None)
                    valid_balanced_accuracy.append(None)

    run_history_df = pd.DataFrame()

    run_history_df["imputer"] = imputer_col
    run_history_df["scaler"] = scaler_col
    run_history_df["feature_preprocessor"] = feature_preprocessor_col
    run_history_df["classifier"] = classifier_col
    run_history_df["train_accuracy"] = train_accuracy
    run_history_df["valid_accuracy"] = valid_accuracy
    run_history_df["train_balanced_accuracy"] = train_balanced_accuracy
    run_history_df["valid_balanced_accuracy"] = valid_balanced_accuracy
    run_history_df["last_task"] = last_task_col
    run_history_df["status"] = status

    run_history_df.sort_values(by=f"valid_{sort_by_metric}", ascending=False, inplace=True)
    run_history_df.to_csv(pjoin(out_dir, f"{ds_name}_train_run_history.csv"), index=False)

    return run_history_df


def train_summary_stats_str(run_history):
    stats_string = "\n========== Training Stats ==========\n"
    stats_string += f"Number of runs: {len(run_history)}\n"
    stats_string += f"Successful runs: {len(run_history[run_history['status'] == 'success'])}\n"
    stats_string += f"Failed runs: {len(run_history[run_history['status'] == 'failed'])}\n"
    stats_string += f"Timeout runs: {len(run_history[run_history['status'] == 'timeout'])}\n"
    stats_string += f"Best pipeline components: "
    stats_string += f"Imputer: {run_history.iloc[0]['imputer']}, "
    stats_string += f"Scaler: {run_history.iloc[0]['scaler']}, "
    stats_string += f"Feature Preprocessor: {run_history.iloc[0]['feature_preprocessor']}, "
    stats_string += f"Classifier: {run_history.iloc[0]['classifier']}\n"

    # stats_string += f"Validation balanced accuracy: {lb_df.iloc[0]['test_balanced_accuracy']:.2f}\n"
    stats_string += f"Validation accuracy: {run_history.iloc[0]['valid_accuracy']:.2f}\n"
    stats_string += "===================================\n"
    return stats_string


def save_train_summary(dataset, run_history, metric="accuracy", out_path="logs"):
    summary = {
        "n_runs": len(run_history),
        "successful": len(run_history[run_history['status'] == 'success']),
        "failed": len(run_history[run_history['status'] == 'failed']),
        "timeout": len(run_history[run_history['status'] == 'timeout']),
        "best_pipeline": [run_history.iloc[0]['imputer'], run_history.iloc[0]['scaler'],
                          run_history.iloc[0]['feature_preprocessor'], run_history.iloc[0]['classifier']],
        f"train_{metric}": run_history.iloc[0][f'train_{metric}'],
        f"validation_{metric}": run_history.iloc[0][f'valid_{metric}']
    }

    path = pjoin(out_path, f"{dataset}_train_summary.json")
    dump_json(summary, path)


def test_summary_stats_str(ds_name, metric="accuracy", results_dir=RESULTS_PATH):
    stats_string = "\n========== Testing Stats ==========\n"
    inc_dir = pjoin(results_dir, ds_name + "_incumbent")

    run_summary = [file for file in listdir(inc_dir) if file.endswith("run_summary.json")]
    assert len(run_summary) == 1, f"There exists more than one run_summary in the incumbet folder of {ds_name}"
    run_summary = run_summary[0]
    run_summary = load_json(pjoin(inc_dir, run_summary))

    stats_string += f"Train {metric}: {run_summary[metric]['train']:.2f}\n"
    stats_string += f"Test {metric}]: {run_summary[metric]['test']:.2f}\n"

    return stats_string


def save_test_summary(ds_name, metric="accuracy", out_path="logs", results_dir=RESULTS_PATH):
    inc_dir = pjoin(results_dir, ds_name + "_incumbent")

    run_summary = [file for file in listdir(inc_dir) if file.endswith("run_summary.json")]
    assert len(run_summary) == 1, f"There exists more than one run_summary in the incumbet folder of {ds_name}"
    run_summary = run_summary[0]
    run_summary = load_json(pjoin(inc_dir, run_summary))

    summary = {
        "pipeline": run_summary["pipeline"],
        f"train_{metric}": run_summary[metric]["train"],
        f"test_{metric}": run_summary[metric]["test"]
    }
    path = pjoin(out_path, f"{ds_name}_test_summary.json")
    dump_json(summary, path)




def save_test_scores_and_pipelines_for_all_datasets(results_dir=RESULTS_PATH, metric="accuracy",
                                                    out_path="logs/test_summary.csv"):
    dataset, classifier, feature_preprocessor, scaler, imputer, test_score = [], [], [], [], [], []
    for _dir in listdir(results_dir):
        if "_incumbent" in _dir:
            ds = _dir[:-10]
            # for ds in datasets:
            #     if f"{ds}_incumbent" in _dir:
            inc_dir = pjoin(results_dir, _dir)
            for file in listdir(inc_dir):
                if file.endswith("_run_summary.json"):
                    run_summary = load_json(pjoin(inc_dir, file))
                    dataset.append(ds)
                    classifier.append(run_summary["pipeline"]["classifier"])
                    feature_preprocessor.append(run_summary["pipeline"]["feature_preprocessor"])
                    scaler.append(run_summary["pipeline"]["scaler"])
                    imputer.append(run_summary["pipeline"]["imputer"])
                    test_score.append(run_summary[metric]["test"])
                    break

    test_summary = pd.DataFrame()
    test_summary["dataset"] = dataset
    test_summary["classifier"] = classifier
    test_summary["feature_preprocessor"] = feature_preprocessor
    test_summary["scaler"] = scaler
    test_summary["imputer"] = imputer
    test_summary[f"test_{metric}"] = test_score

    test_summary.to_csv(out_path, index=False)
    # return test_summary
