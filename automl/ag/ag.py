import sys
sys.path.append("..")
sys.path.append("../..")

import pandas as pd
from sklearn.metrics import accuracy_score

from os.path import join as pjoin
from automl.utils import get_luigi_enumeration_time, dump_pickle, dump_json, set_seed


from autogluon.tabular import TabularDataset, TabularPredictor


def main(ds, time_factor, seed, n_jobs, presets, datasets_dir, cls_luigi_dir, working_dir):

    print(f"Running AutoGluon with dataset {ds}...")
    set_seed(seed)

    x_train = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}",  "test_phase", "x_train.csv"))
    x_test = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}", "test_phase", "x_test.csv"))
    y_train = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}", "test_phase", "y_train.csv"))
    y_test = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}", "test_phase", "y_test.csv"))

    label = y_train.columns[0]

    train = x_train.copy()
    train[label] = y_train[label].tolist()

    test = x_test.copy()
    test[label] = y_test[label].tolist()

    task_time = get_luigi_enumeration_time(cls_luigi_dir, ds, seed) * time_factor
    task_time = int(task_time)

    print(f"time_left_for_this_task is set to {task_time} seconds")

    train_data = TabularDataset(train)
    test_data = TabularDataset(test)

    run_dir = pjoin(working_dir, "ag_results", ds, f"seed-{seed}")
    print("Starting training...")

    automl = TabularPredictor(
        label=label,
        path=run_dir,
        verbosity=2,
    ).fit(
        train_data=train_data,
        time_limit=float(task_time),
        presets=presets,
        num_cpus=n_jobs,
    )
    dump_pickle(automl, f"ag_results/{ds}/askl_obj.pkl")
    print("Saved automl object...")
    print("Predicting...")

    train_prediction = automl.predict(train_data.drop(columns=[label]))
    train_accuracy = accuracy_score(y_train, train_prediction)

    test_prediction = automl.predict(test_data.drop(columns=[label]))
    test_accuracy = accuracy_score(y_test, test_prediction)

    best_pipeline_summary = {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }
    dump_json(best_pipeline_summary, f"ag_results/{ds}/best_pipeline_summary.json")

    print(
        "Done!\n==================================================================================================\n\n")


if __name__ == "__main__":

    import argparse
    import pathlib
    import os
    CWD = os.getcwd()
    DATASETS_DEFAULT_DIR = pjoin(pathlib.Path(CWD).parent, "datasets")
    CLS_LUIGI_DIR = pjoin(pathlib.Path(CWD).parent, "cls_luigi_automl")

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Seed for reproducibility")

    parser.add_argument("--time_factor",
                        type=float,
                        default=1,
                        help="allowed time for running AutoML systems = CLS-Luigi time * time_factor")

    parser.add_argument('--ds_name',
                        type=str,
                        help='Dataset name')

    parser.add_argument("--n_jobs",
                        type=int,
                        default=1,
                        help="Number of jobs ")

    parser.add_argument("--datasets_dir",
                        type=str,
                        default=DATASETS_DEFAULT_DIR,
                        help="Path to the datasets directory")

    parser.add_argument("--working_dir",
                        type=str,
                        default=CWD,
                        help="Path to the working directory where the results will be saved"
                        )
    parser.add_argument("--presets",
                        type=str,
                        default="best_quality",
                        help="Preset to use in AutoGluon")
    config = parser.parse_args()

    main(
        config.ds_name,
        config.time_factor,
        config.seed,
        config.n_jobs,
        config.presets,
        config.datasets_dir,
        CLS_LUIGI_DIR,
        config.working_dir

    )


# python ag.py --ds_name kc1
