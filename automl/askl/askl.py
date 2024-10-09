import sys
sys.path.append("..")
sys.path.append("../..")

from sklearn.metrics import accuracy_score
from os.path import join as pjoin
import logging
import pandas as pd

from automl.utils import dump_pickle, load_json, dump_json, get_best_askl_pipeline


def get_luigi_enumeration_time(cls_luigi_dir, ds_name, seed):

    return load_json(
        pjoin(
            cls_luigi_dir,
            "enumeration_outputs",
            f"seed-{seed}",
            f"{ds_name}",
            "logs",
            "train_time.json")
    )["total_seconds"]


def main(
        ds,
        time_factor,
        seed,
        askl_version,
        apply_ensemble,
        n_jobs,
        memory_limit,
        datasets_dir,
        working_dir,
        task_time
):
    print(f"Running AutoSklearn {askl_version} with dataset {ds}...")

    x_train = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}",  "test_phase", "x_train.csv"))
    x_test = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}", "test_phase", "x_test.csv"))
    y_train = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}", "test_phase", "y_train.csv"))
    y_test = pd.read_csv(pjoin(datasets_dir, ds, f"seed-{seed}", "test_phase", "y_test.csv"))


    print(f"time_left_for_this_task is set to {task_time} seconds")

    logging_config = logging.basicConfig(level=logging.DEBUG,
                                         format='%(asctime)s - %(levelname)s - %(message)s',
                                         handlers=[logging.StreamHandler()])

    askl_config = {
        "time_left_for_this_task": task_time,
        "max_models_on_disc": 1000000,  # try to save all save all the models
        "seed": seed,
        "delete_tmp_folder_after_terminate": False,
        "n_jobs": n_jobs,
        "memory_limit": memory_limit,
        "logging_config": logging_config

    }

    if apply_ensemble:
        print("Running with ensembling")
        run_dir = pjoin(working_dir, "askl_results", f"askl{askl_version}_ens_results", ds, f"time_factor-{time_factor}")

    else:
        print("Running without ensembling")

        run_dir = pjoin(working_dir, "askl_results", f"askl{askl_version}_no_ens_results", ds, f"time_factor-{time_factor}")
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

    lb = automl.leaderboard()
    lb.to_csv(pjoin(run_dir, "leaderboard.csv"))

    best_pipeline_id = int(lb.index[0])
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
    
    
    dump_json(
        obj=best_pipeline_summary, 
        path=pjoin(run_dir, "best_pipeline_summary.json")
        )

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
                        default=2.0,
                        help="allowed time for running AutoML systems = CLS-Luigi time * time_factor")

        
    parser.add_argument("--max_sec",
                        type=int,
                        default=0,
                        help="Max seconds to run the MCTS, If this time is 0, the enumeration time will be used"
                        )
    
    parser.add_argument("--askl_version",
                        type=int,
                        choices=[1, 2],
                        help="Whether to use AutoSklearn 1.0 or 2.0")

    parser.add_argument('--ensemble',
                        action='store_true',
                        help="Whether to activate ensembling in AutoSklearn")

    parser.add_argument('--ds_name',
                        type=str,
                        help='Dataset name')

    parser.add_argument("--n_jobs",
                        type=int,
                        default=1,
                        help="Number of jobs ")

    parser.add_argument("--memory_limit",
                        type=int,
                        default=100000,
                        help="Memory limit in the AutoML procedure")

    parser.add_argument("--datasets_dir",
                        type=str,
                        default=DATASETS_DEFAULT_DIR,
                        help="Path to the datasets directory")

    parser.add_argument("--working_dir",
                        type=str,
                        default=CWD,
                        help="Path to the working directory where the results will be saved"
                        )




    config = parser.parse_args()
    
    task_time = config.max_sec if config.max_sec > 0 else get_luigi_enumeration_time(
        CLS_LUIGI_DIR,
        config.ds_name,
        config.seed) * config.time_factor
    task_time = int(task_time)

    main(
        config.ds_name,
        config.time_factor,
        config.seed,
        config.askl_version,
        True if config.ensemble else False,
        config.n_jobs,
        config.memory_limit,
        config.datasets_dir,
        config.working_dir,
        task_time
    )
    # except:
    #     print(f"Failed to run AutoSklearn with dataset {config.ds_name}!")
    #     print("Make sure you downloaded the dataset and it exists in the datasets folder")

# python askl.py --ds_name kc1 --time_factor 1 --askl_version 1 --ensemble
# python askl.py --time_factor 1 --askl_version 1 --ensemble
# python askl.py --time_factor 1 --askl_version 2 --ensemble
# python askl.py --time_factor 1 --askl_version 2 --ensemble
