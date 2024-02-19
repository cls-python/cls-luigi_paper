import sys

sys.path.append("..")

import logging
import os

# CLS-Luigi imports
from cls_luigi.inhabitation_task import RepoMeta

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls.debug_util import deep_str
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
    save_train_summary, save_test_summary

from utils.automl_scores_utils import save_test_scores_and_pipelines_for_all_datasets
from utils.io_methods import dump_txt

loggers = [logging.getLogger("luigi-root"), logging.getLogger("luigi-interface")]


def generate_and_filter_pipelines():
    target = Classifier.return_type()
    print("Collecting repo...")
    repository = RepoMeta.repository
    print("Building...")

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Building grammar tree and inhabiting pipelines...")

    inhabitation_result = fcl.inhabit(target)
    rules_string = deep_str(inhabitation_result.rules)
    dump_txt(rules_string, "inhabitation_rules.txt")
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


def set_luigi_worker_configs(timeout_sec=None):
    luigi.configuration.get_config().remove_section("worker")

    if timeout_sec:
        luigi.configuration.get_config().set('worker', 'timeout', str(timeout_sec))


def run_train_phase(paths, pipelines, ds_name, seed, worker_timeout, workers=1):
    set_global_parameters(
        paths["train_phase"]["x_train_path"],
        paths["train_phase"]["x_valid_path"],
        paths["train_phase"]["y_train_path"],
        paths["train_phase"]["y_valid_path"],
        ds_name,
        seed)

    print(f"Running training phase (all pipelines) for dataset {ds_name} using the training and validation datasets...")
    with TimeRecorder(f"logs/{ds_name}_train_time.json"):
        set_luigi_worker_configs(timeout_sec=worker_timeout)
        with LuigiDaemon():
            luigi.build(
                pipelines,
                local_scheduler=False,
                logging_conf_file="logging.conf",
                detailed_summary=True,
                workers=workers,
            )

    loggers[1].warning("\n{}\n{} This was dataset: {} {} training phase\n{}\n".format(
        "*" * 150,
        "*" * 65,
        ds_name,
        "*" * (65 - len(str(paths))),
        "*" * 150))


def run_test_phase(paths, best_pipeline, ds_name, seed, worker_timeout, workers=1):
    set_global_parameters(
        paths["test_phase"]["x_train_path"],
        paths["test_phase"]["x_test_path"],
        paths["test_phase"]["y_train_path"],
        paths["test_phase"]["y_test_path"],
        ds_name + "_incumbent",
        seed)

    print(f"Running testing phase (best pipeline) for dataset {ds_name} using the training and testing datasets...")
    with TimeRecorder(f"logs/{ds_name}_test_time.json"):
        set_luigi_worker_configs(timeout_sec=worker_timeout)
        with LuigiDaemon():
            luigi.build(
                [best_pipeline],
                local_scheduler=False,
                detailed_summary=True,
                workers=workers,
            )
    loggers[1].warning("\n{}\n{} This was dataset: {} {} testing phase\n{}\n".format(
        "*" * 150,
        "*" * 65,
        ds_name,
        "*" * (65 - len(str(paths))),
        "*" * 150))


def main(pipelines, seed, metric, ds_id, train_worker_timeout=100, test_worker_timeout=None, workers=1):
    os.makedirs("logs/luigi_logs", exist_ok=True)

    print(f"Downloading dataset with ID {ds_id}")
    ds_name, paths = download_and_save_openml_dataset(ds_id, seed)

    if pipelines:

        print("=============================================================================================")
        print(f"                    Training and Testing on dataset: {ds_name}")
        print("=============================================================================================")

        run_train_phase(paths, pipelines, ds_name, seed, train_worker_timeout, workers)

        run_history_df = generate_and_save_run_history(ds_name=ds_name, sort_by_metric=metric)
        print("Generated and saved training run history for dataset:", ds_name)
        print(train_summary_stats_str(run_history_df))
        save_train_summary(ds_name, run_history_df)

        best_pipeline_id = run_history_df.iloc[0]["last_task"][0]
        best_pipeline = [p for p in pipelines if p.task_id == best_pipeline_id][0]
        run_test_phase(paths, best_pipeline, ds_name, seed, test_worker_timeout, workers)

        print(test_summary_stats_str(ds_name))
        save_test_summary(ds_name)
        print("=============================================================================================\n\n")

    else:
        print("No results!")


if __name__ == "__main__":
    seed = 123
    metric = "accuracy"

    datasets_ids = [
        9957,  # qsar-biodeg
        359958,  # pc4 classification
        9967,  # steel-plates-fault
        359962,  # kc1 classification
        9978,  # ozone-level-8hr
        146820,  # wilt
        43,  # spambase

        359972,  # sylvin classification

        9952,  # phoneme
        361066,  # bank-marketing classification
        9983,  # eeg-eye-state
        3899,  # mozilla4
        9976,  # madelon

        146606,  # higgs
        167120,  # numerai28.6

    ]

    os.mkdir("logs")
    with TimeRecorder(f"logs/synthesis_time.json"):
        import_pipeline_components()
        pipelines = generate_and_filter_pipelines()

    for ds_id in datasets_ids:
        main(
            pipelines=pipelines,
            seed=seed,
            metric=metric,
            ds_id=ds_id,
            train_worker_timeout=100,
            test_worker_timeout=None,
            workers=1)

    save_test_scores_and_pipelines_for_all_datasets()

