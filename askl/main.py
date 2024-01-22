import pickle
import sys
import os

from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import json

import pandas as pd
from os.path import join as pjoin
import autosklearn.classification
from utils.io_methods import load_json, dump_pickle, dump_json


def datasets():
    datasets_names = [
        'sylvine',
        'bank-marketing',
        'kc1',
        'phoneme',

        'pc4', # exists in autosklearn
        'wilt', # exists in autosklearn
        'qsar-biodeg', # exists in autosklearn
        'mozilla4', # exists in autosklearn
        'steel-plates-fault', # exists in autosklearn
        'ozone-level-8hr', # exists in autosklearn
        'eeg-eye-state', # exists in autosklearn

        'madelon',  # exists in autosklearn
        'numerai28.6',
        'higgs',
        'APSFailure',
        'riccardo',

    ]

    return datasets_names


def load_split_dataset(ds_name):
    path = pjoin(ROOT, "automl_pipelines", "datasets", ds_name, "test_phase")

    x_train = pd.read_csv(pjoin(path, "x_train.csv"))
    x_test = pd.read_csv(pjoin(path, "x_test.csv"))
    y_train = pd.read_csv(pjoin(path, "y_train.csv"))
    y_test = pd.read_csv(pjoin(path, "y_test.csv"))

    return x_train, x_test, y_train, y_test


def get_task_seconds(ds_name, factor=1):
    path = pjoin(ROOT, "automl_pipelines/logs/", f"{ds_name}_time.json")

    return int(load_json(path)["total_seconds"] * factor)


def get_best_pipeline(pipeline_id, ds_name):
    run_history_path = pjoin(f"results/{ds_name}/smac3-output/run_{seed}/runhistory.json")
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


def main():
    for ds in datasets():
        print(f"Running AutoSklearn with dataset {ds}...")
        x_train, x_test, y_train, y_test = load_split_dataset(ds)
        task_time = get_task_seconds(ds, factor=time_factor)
        print(f"time_left_for_this_task is set to {task_time} seconds")

        run_dir = pjoin("results", ds)
        print("Starting training...")
        automl = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=task_time,
            ensemble_class=None,
            max_models_on_disc=1000000,  # to save all the models
            seed=seed,
            tmp_folder=run_dir,
            delete_tmp_folder_after_terminate=False,
            n_jobs=1,
            memory_limit=100000
        )
        automl.fit(x_train, y_train, dataset_name=ds)
        print("Finished training...")

        dump_pickle(automl, f"results/{ds}/askl_obj.pkl")

        print("Saved automl object...")

        best_pipeline_id = int(automl.leaderboard().index[0])

        print("Predicting with best pipeline...")
        train_prediction = automl.predict(x_train)
        train_accuracy = accuracy_score(y_train, train_prediction)

        test_prediction = automl.predict(x_test)
        test_accuracy = accuracy_score(y_test, test_prediction)

        best_pipeline_summary = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy
        }

        best_pipeline_summary.update(
            get_best_pipeline(best_pipeline_id, ds))

        print("Saving best pipeline summary...")

        dump_json(best_pipeline_summary, f"results/{ds}/best_pipeline_summary.json")

        print("Done!\n==================================================================================================\n\n")

if __name__ == "__main__":
    seed = 42
    time_factor = 2
    main()
