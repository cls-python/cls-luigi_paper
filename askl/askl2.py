import sys
import os

from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from os.path import join as pjoin
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from utils.io_methods import dump_pickle, dump_json
from utils.helpers import load_split_dataset, get_task_seconds, get_best_askl_pipeline


def main(ds, time_factor, seed):
    print(f"Running AutoSklearn with dataset {ds}...")
    x_train, x_test, y_train, y_test = load_split_dataset(ds, root=ROOT)
    task_time = get_task_seconds(ds_name=ds, root=ROOT, factor=time_factor)
    print(f"time_left_for_this_task is set to {task_time} seconds")

    run_dir = pjoin(ROOT, "askl", "askl2_results", ds)
    print("Starting training...")

    automl = AutoSklearn2Classifier(
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

    dump_pickle(automl, f"askl2_results/{ds}/askl_obj.pkl")

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
        get_best_askl_pipeline(best_pipeline_id, ds, seed, askl1=False))

    print("Saving best pipeline summary...")

    dump_json(best_pipeline_summary, f"askl2_results/{ds}/best_pipeline_summary.json")

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
