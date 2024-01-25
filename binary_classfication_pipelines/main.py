import sys
import time

from tqdm import tqdm

sys.path.append("..")

import logging
import os

# CLS-Luigi imports
from cls_luigi.inhabitation_task import RepoMeta

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator

# Global Parameters and AutoML validator
from implementations.global_parameters import GlobalParameters
from validators.not_forbidden_validator import NotForbiddenValidator

# template
from implementations.template import *

from download_and_save_openml_datasets import download_and_save_openml_dataset
from import_pipeline_components import import_pipeline_components

from utils.luigi_daemon import LuigiDaemon
from utils.time_recorder import TimeRecorder

from utils.automl_scores_utils import generate_and_save_run_history, train_summary_stats_str, test_summary_stats_str, \
    save_train_summary, save_test_scores_and_pipelines_for_all_datasets

loggers = [logging.getLogger("luigi-root"), logging.getLogger("luigi-interface")]


# def datasets():
#     datasets_list = [
#         # 9967,  # steel-plates-fault
#         # 9957,  # qsar-biodeg
#         # 9952,  # phoneme
#         # 9978,  # ozone-level-8hr
#         # 145847,  # hill-valley
#         # 146820,  # wilt
#         # 3899,  # mozilla4
#         # 9983,  # eeg-eye-state
#         359962,  # kc1 classification
#         # 359958,  # pc4 classification
#         # 361066,  # bank-marketing classification
#         # 359972,  # sylvin classification
#         # 167120,  # numerai28.6
#         # 9976,  # Madelon
#         # 146606,  # higgs
#         # 168868,  # APSFailure
#         # 168338,  # riccardo


#     ]
#     return datasets_list


def generate_and_filter_pipelines():
    target = Classifier.return_type()
    print("Collecting repo...")
    repository = RepoMeta.repository
    print("Building...")

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Building grammar tree and inhabiting pipelines...")

    inhabitation_result = fcl.inhabit(target)
    print("Enumerating pipelines...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if actual > 0:
        max_results = actual

    print("Filtering using UniqueTaskPipelineValidator...")
    validator = UniqueTaskPipelineValidator(
        [LoadSplitData, NumericalImputer, Scaler, FeaturePreprocessor,
         Classifier])
    pipelines = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]

    print("Filtering using NotForbiddenValidator...")
    automl_validator = NotForbiddenValidator()
    pipelines = [t for t in pipelines if automl_validator.validate(t)]
    print("Generated {} pipelines".format(max_results))
    print("Number of pipelines after filtering:", len(pipelines))
    return pipelines


def set_global_parameters(x_train, x_test, y_train, y_test, ds_name, seed) -> None:
    global_parameters = GlobalParameters()
    global_parameters.x_train_path = x_train
    global_parameters.x_test_path = x_test
    global_parameters.y_train_path = y_train
    global_parameters.y_test_path = y_test
    global_parameters.dataset_name = ds_name
    global_parameters.seed = seed


def run_train_phase(paths, pipelines, ds_name, seed, worker_timeout=100):
    set_global_parameters(
        paths["train_phase"]["x_train_path"],
        paths["train_phase"]["x_valid_path"],
        paths["train_phase"]["y_train_path"],
        paths["train_phase"]["y_valid_path"],
        ds_name,
        seed)

    print(f"Running training phase (all pipelines) for dataset {ds_name} using the training and validation datasets...")
    with TimeRecorder(f"logs/{ds_name}_time.json"):
        with LuigiDaemon():
            for pipeline in tqdm(pipelines):
                # pipeline.set_worker_timeout_for_all_tasks(worker_timeout=worker_timeout)  # 100 seconds by default
                luigi.build(
                    [pipeline],
                    local_scheduler=False,
                    logging_conf_file="logging.conf",
                    detailed_summary=True,
                    workers=1
                )

    loggers[1].warning("\n{}\n{} This was dataset: {} {}\n{}\n".format(
        "*" * 150,
        "*" * 65,
        ds_name,
        "*" * (65 - len(str(paths))),
        "*" * 150))


def run_test_phase(paths, best_pipeline, ds_name, seed, worker_timeout=None):
    set_global_parameters(
        paths["test_phase"]["x_train_path"],
        paths["test_phase"]["x_test_path"],
        paths["test_phase"]["y_train_path"],
        paths["test_phase"]["y_test_path"],
        ds_name + "_incumbent",
        seed)

    print(f"Running testing phase (best pipeline) for dataset {ds_name} using the training and testing datasets...")
    best_pipeline.set_worker_timeout_for_all_tasks(worker_timeout=worker_timeout)

    tick = time.time()
    with LuigiDaemon():
        luigi.build(
            [best_pipeline],
            local_scheduler=False,
            detailed_summary=True,
            workers=1
        )

    tock = time.time()
    print(f"Testing phase took: {tock - tick:.1f} seconds")


def main(pipelines, seed, metric):
    os.makedirs("logs", exist_ok=True)
    # import_pipeline_components()
    # pipelines = generate_and_filter_pipelines()

    if pipelines:
        # for ds_id in datasets():
        print(f"Downloading dataset with ID {ds_id}")
        ds_name, paths = download_and_save_openml_dataset(ds_id, seed)

        print("=============================================================================================")
        print(f"                    Training and Testing on dataset: {ds_name}")
        print("=============================================================================================")

        run_train_phase(paths, pipelines, ds_name, seed)

        run_history_df = generate_and_save_run_history(ds_name=ds_name, sort_by_metric=metric)
        print("Generated and saved training run history for dataset:", ds_name)
        print(train_summary_stats_str(run_history_df))
        save_train_summary(ds_name, run_history_df)

        best_pipeline_id = run_history_df.iloc[0]["last_task"][0]
        best_pipeline = [p for p in pipelines if p.task_id == best_pipeline_id][0]
        run_test_phase(paths, best_pipeline, ds_name, seed)

        print(test_summary_stats_str(ds_name))
        print("=============================================================================================\n\n")
        save_test_scores_and_pipelines_for_all_datasets(metric=metric)
    else:
        print("No results!")


if __name__ == "__main__":
    seed = 42
    metric = "accuracy"
    datasets_ids = [
        # 9967,  # steel-plates-fault
        # 9957,  # qsar-biodeg
        # 9952,  # phoneme
        # 9978,  # ozone-level-8hr
        # 145847,  # hill-valley
        # 146820,  # wilt
        # 3899,  # mozilla4
        # 9983,  # eeg-eye-state
        359962,  # kc1 classification
        # 359958,  # pc4 classification
        # 361066,  # bank-marketing classification
        # 359972,  # sylvin classification
        # 167120,  # numerai28.6
        # 9976,  # Madelonsave_test_scores_and_pipelines_for_all_datasets
        # 146606,  # higgs
        # 168868,  # APSFailure
        # 168338,  # riccardo
    ]
    
    
    import_pipeline_components()
    pipelines = generate_and_filter_pipelines()
    
    for ds_id in datasets_ids:
    
        main(pipelines, seed, metric)
    
    
    save_test_scores_and_pipelines_for_all_datasets(metric=metric)

