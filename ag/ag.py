import sys
import os

from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from os.path import join as pjoin
from utils.io_methods import dump_pickle, dump_json
from utils.helpers import load_split_dataset, get_task_seconds, get_best_askl_pipeline, set_seed
from autogluon.tabular import TabularDataset, TabularPredictor


def main(ds, time_factor, seed):
    print(f"Running AutoGluon with dataset {ds}...")
    set_seed(seed)

    x_train, x_test, y_train, y_test = load_split_dataset(ds, root=ROOT)

    label = y_train.columns[0]

    train = x_train.copy()
    train[label] = y_train[label].tolist()

    test = x_test.copy()
    test[label] = y_test[label].tolist()

    task_time = get_task_seconds(ds_name=ds, root=ROOT, factor=time_factor)
    print(f"time_left_for_this_task is set to {task_time} seconds")

    train_data = TabularDataset(train)
    test_data = TabularDataset(test)

    run_dir = pjoin(ROOT, "ag", "ag_results", ds)
    print("Starting training...")

    automl = TabularPredictor(
        label=label,
        path=run_dir,
        verbosity=2,
    ).fit(
        train_data=train_data,
        time_limit=float(task_time),
        presets="best_quality",
        num_cpus=1,
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


    seed = 42
    time_factor = 2

    datasets = [
        'spambase',  # exists in autosklearn
        'sylvine',
        'bank-marketing',
        'phoneme',
        'kc1',  # exists in autosklearn
        'pc4',  # exists in autosklearn
        'wilt',  # exists in autosklearn
        'qsar-biodeg',  # exists in autosklearn
        'mozilla4',  # exists in autosklearn
        'steel-plates-fault',  # exists in autosklearn
        'ozone-level-8hr',  # exists in autosklearn
        'eeg-eye-state',  # exists in autosklearn
        'madelon',
        'numerai28.6',
        'higgs',
    ]

    for ds in datasets:
        try:
            main(ds, time_factor, seed)
        except:
            print(f"Failed to run AutoSklearn with dataset {ds}!")
            print("Make sure you downloaded the dataset and it exists in the datasets folder")
