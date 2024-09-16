import sys
import os

from sklearn.metrics import accuracy_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from os.path import join as pjoin
from utils.io_methods import dump_pickle, dump_json
from utils.helpers import load_split_dataset, get_task_seconds, get_best_askl_pipeline
import logging


def main(
    ds,
    time_factor,
    seed,
    askl_version,
    apply_ensemble,
    n_jobs,
    memory_limit
    ):
    
    
    print(f"Running AutoSklearn {askl_version} with dataset {ds}...")
    x_train, x_test, y_train, y_test = load_split_dataset(ds, root=ROOT)
    task_time = get_task_seconds(ds_name=ds, root=ROOT, factor=time_factor)
    print(f"time_left_for_this_task is set to {task_time} seconds")
    
    
    logging_config = logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
    
    askl_config = {
        "time_left_for_this_task": task_time,
        "max_models_on_disc":1000000,  # try to save all save all the models
        "seed":seed,
        "delete_tmp_folder_after_terminate":False,
        "n_jobs":n_jobs,
        "memory_limit":memory_limit,
        "logging_config": logging_config
        
        }
    
    
    if apply_ensemble:
        print("Running with ensembling")
        run_dir = pjoin(ROOT, "askl", f"askl{askl_version}_ens_results", ds)
        
    else:
        print("Running without ensembling")

        run_dir = pjoin(ROOT, "askl", f"askl{askl_version}_no_ens_results", ds)
        askl_config["ensemble_class"] = None
                
    askl_config["tmp_folder"] = run_dir

    print("Starting training...")


    if askl_version == 1:
        import autosklearn.classification

        automl = autosklearn.classification.AutoSklearnClassifier(
            **askl_config
        )
    
    elif askl_version == 2:
        from autosklearn.experimental.askl2 import AutoSklearn2Classifier
        
        automl = AutoSklearn2Classifier(**askl_config)
        
        
    automl.fit(x_train, y_train, dataset_name=ds)
    print("Finished training...")

    dump_pickle(automl, pjoin(run_dir, "askl_obj.pkl"))

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
        get_best_askl_pipeline(
            pjoin(run_dir, f"smac3-output/run_{seed}/runhistory.json"),
            best_pipeline_id))

    print("Saving best pipeline summary...")

    dump_json(best_pipeline_summary, pjoin(run_dir, "best_pipeline_summary.json"))

    print(
        "Done!\n==================================================================================================\n\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed",
                    type=int,
                    default=42,
                    help="Seed for reproducibility")
        
    parser.add_argument("--time_factor",
                    type=int,
                    default=2,
                    help="allowed time for running AutoML systems = CLS-Luigi time * time_factor")
    
    parser.add_argument("--askl_version",
                type=int,
                default=1,
                choices=[1, 2],
                help="Whether to use AutoSklearn 1.0 or 2.0")
    
    parser.add_argument('--ensemble',
                        action='store_true',
                        help="Whether to activate ensembling in AutoSklearn")
    
    parser.add_argument("--n_jobs",
        type=int,
        default=1,
        help="Number of jobs ")
    
    parser.add_argument("--memory_limit",
        type=int,
        default=100000,
        help="Memory limit in the AutoML procedure")
    
    config = parser.parse_args()

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
            main(
                ds,
                config.time_factor,
                config.seed,
                config.askl_version,
                True if config.ensemble else False,
                config.n_jobs,
                config.memory_limit
                )
        except:
            print(f"Failed to run AutoSklearn with dataset {ds}!")
            print("Make sure you downloaded the dataset and it exists in the datasets folder")


# python askl.py --time_factor 1 --askl_version 1 --apply_ensemble True 
# python askl.py --time_factor 1 --askl_version 1 --apply_ensemble False
# python askl.py --time_factor 1 --askl_version 2 --apply_ensemble True
# python askl.py --time_factor 1 --askl_version 2 --apply_ensemble False