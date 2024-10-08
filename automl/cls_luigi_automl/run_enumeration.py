import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")



import logging
import os
from os.path import join as pjoin
import luigi

# CLS-Luigi imports
from cls_luigi.inhabitation_task import RepoMeta, CLSLuigiDecoder, CLSLugiEncoder

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls.debug_util import deep_str
from tqdm import tqdm

# Global Parameters and AutoML validator
from implementations.global_parameters import GlobalPipelineParameters
from validators.not_forbidden_validator import NotForbiddenValidator

# template
from implementations.template import *

from automl.download_and_save_openml_datasets import download_and_save_openml_dataset
from automl.cls_luigi_automl.import_pipeline_components import import_pipeline_components

from cls_luigi.tools.luigi_daemon import LinuxLuigiDaemonHandler as LuigiDaemon
from cls_luigi.tools.time_recorder import TimeRecorder

from automl.utils import generate_and_save_run_history, train_summary_stats_str, \
    save_train_summary, save_test_summary

from automl.utils import save_test_scores_and_pipelines_for_all_datasets
from automl.utils import dump_txt, load_json, dump_json

loggers = [logging.getLogger("luigi-root"), logging.getLogger("luigi-interface")]

def generate_and_filter_pipelines(logs_dir):
    target = Classifier.return_type()
    print("Collecting repo...")
    repository = RepoMeta.repository
    print("Building...")

    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    print("Building grammar tree and inhabiting pipelines...")

    inhabitation_result = fcl.inhabit(target)
    rules_string = deep_str(inhabitation_result.rules)
    dump_txt(rules_string, pjoin(logs_dir, "inhabitation_rules.txt"))
    print("Enumerating pipelines...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if actual > 0:
        max_results = actual

    pipelines = [t for t in inhabitation_result.evaluated[0:max_results]]
    print("Filtering using NotForbiddenValidator...")
    automl_validator = NotForbiddenValidator()
    pipelines = [t for t in pipelines if automl_validator.validate(t())]

    print("Generated {} pipelines".format(max_results))
    print("Number of pipelines after filtering:", len(pipelines))
    return pipelines


def get_and_save_pipelines(logs_dir, pipelines_dir="json_pipelines"):

    with TimeRecorder(pjoin(logs_dir, "pipeline_synthesis_and_encoding_time.json")):
        import_pipeline_components()
        pipelines = generate_and_filter_pipelines(logs_dir)
        for p in pipelines:
            p_path = os.path.join(pipelines_dir, f"{p().task_id}.json")
            dump_json(
                obj=p,
                path=p_path,
                encoder_cls=CLSLugiEncoder)

    pipelines = instantiate_pipelines(pipelines)
    return pipelines


def instantiate_pipelines(pipelines):
    return [pipeline() for pipeline in pipelines]


def set_global_parameters(x_train, x_test, y_train, y_test, pipelines_outputs_dir, seed) -> None:
    global_parameters = GlobalPipelineParameters()
    global_parameters.x_train_path = x_train
    global_parameters.x_test_path = x_test
    global_parameters.y_train_path = y_train
    global_parameters.y_test_path = y_test
    global_parameters.seed = seed
    global_parameters.pipelines_outputs_dir = pipelines_outputs_dir


def set_luigi_worker_configs(timeout_sec=None):
    luigi.configuration.get_config().remove_section("worker")

    if timeout_sec:
        luigi.configuration.get_config().set('worker', 'timeout', str(timeout_sec))


def run_train_phase(paths, pipelines, pipeline_outputs_dir, ds_name, seed, worker_timeout, logs_dir,  workers=1):
    set_global_parameters(
        paths["train_phase"]["x_train_path"],
        paths["train_phase"]["x_valid_path"],
        paths["train_phase"]["y_train_path"],
        paths["train_phase"]["y_valid_path"],
        pipeline_outputs_dir,
        seed)

    print(f"Running training phase (all pipelines) for dataset {ds_name} using the training and validation datasets...")
    with TimeRecorder(pjoin(logs_dir, "train_time.json")):
        set_luigi_worker_configs(timeout_sec=worker_timeout)
        with LuigiDaemon():
            for p in tqdm(pipelines):
                luigi.build(
                    [p],
                    local_scheduler=False,
                    # logging_conf_file="logging.conf",
                    detailed_summary=True,
                    workers=workers,
                )

    loggers[1].warning("\n{}\n{} This was dataset: {} {} training phase\n{}\n".format(
        "*" * 150,
        "*" * 65,
        ds_name,
        "*" * (65 - len(str(paths))),
        "*" * 150))


def run_test_phase(paths, best_pipeline, pipeline_outputs_dir, ds_name, seed, worker_timeout, logs_dir, workers=1):
    set_global_parameters(
        paths["test_phase"]["x_train_path"],
        paths["test_phase"]["x_test_path"],
        paths["test_phase"]["y_train_path"],
        paths["test_phase"]["y_test_path"],
        pipeline_outputs_dir,
        seed)

    print(f"Running testing phase (best pipeline) for dataset {ds_name} using the training and testing datasets...")
    with TimeRecorder(pjoin(logs_dir, "test_time.json")):
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


def main(pipelines, seed, metric, ds_name, ds_paths,
         train_outputs_dir, test_output_dir, logs_dir,
         train_worker_timeout=100,
         test_worker_timeout=None, workers=1):

    if pipelines:

        print("=============================================================================================")
        print(f"                    Training and Testing on dataset: {ds_name}")
        print("=============================================================================================")

        run_train_phase(ds_paths, pipelines, train_outputs_dir, ds_name, seed, train_worker_timeout,logs_dir, workers)

        run_history_df = generate_and_save_run_history(ds_name=ds_name,
                                                       results_dir=train_outputs_dir,
                                                       sort_by_metric=metric,
                                                       out_dir=logs_dir
                                                       )
        print("Generated and saved training run history for dataset:", ds_name)
        print(train_summary_stats_str(run_history_df))
        save_train_summary(run_history=run_history_df, out_path=logs_dir)

        best_pipeline_id = run_history_df.iloc[0]["last_task"][0]
        best_pipeline = [p for p in pipelines if p.task_id == best_pipeline_id][0]
        run_test_phase(ds_paths, best_pipeline, test_output_dir, ds_name, seed, test_worker_timeout, logs_dir,workers)

        save_test_summary(pipeline=best_pipeline, out_path=logs_dir)
        print("=============================================================================================\n\n")

    else:
        print("No results!")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    CWD = os.getcwd()
    DEFAULT_OUTPUT_DIR = pjoin(CWD, "enumeration_outputs")

    DATASETS_DIR = pjoin(
        Path(CWD).parent.absolute(),
        "datasets"
    )

    parser.add_argument("--outputs_dir",
                        type=str,
                        default=DEFAULT_OUTPUT_DIR,
                        help="Directory for all pipeline enumeration outputs"
                        )

    parser.add_argument("--datasets_dir",
                        type=str,
                        default=DATASETS_DIR,
                        help="Directory for all datasets"
                        )

    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Seed for reproducibility")

    parser.add_argument("--metric",
                        type=str,
                        default="accuracy",
                        help="Metric for sorting the results")

    parser.add_argument("--ds_id",
                        type=int,
                        default=359962,
                        help="OpenML dataset ID. Default is kc1 classification dataset.")

    parser.add_argument("--train_worker_timeout",
                        type=int,
                        default=100,
                        help="Luigi worker timeout for the training phase")

    parser.add_argument("--test_worker_timeout",
                        type=int,
                        default=None,
                        help="Luigi worker timeout for the testing phase")

    parser.add_argument("--workers",
                        type=int,
                        default=1,
                        help="Number of luigi workers")

    config = parser.parse_args()

    print(f"Downloading dataset with ID {config.ds_id}")
    ds_name, paths = download_and_save_openml_dataset(config.datasets_dir,
                                                      config.ds_id, config.seed)

    DS_OUTPUT_DIR = pjoin(DEFAULT_OUTPUT_DIR, f"seed-{config.seed}", ds_name)
    LOGS_DIR = pjoin(DS_OUTPUT_DIR, "logs")
    PIPELINES_DIR = pjoin(DS_OUTPUT_DIR, "pipelines")
    PIPELINE_OUTPUTS = pjoin(DS_OUTPUT_DIR, "pipelines_outputs")
    INC_OUTPUT_DIR = pjoin(DS_OUTPUT_DIR, "incumbent_outputs")

    os.makedirs(DS_OUTPUT_DIR, exist_ok=False)
    os.makedirs(LOGS_DIR, exist_ok=False)
    os.makedirs(PIPELINES_DIR, exist_ok=False)
    os.makedirs(PIPELINE_OUTPUTS, exist_ok=False)
    os.makedirs(INC_OUTPUT_DIR, exist_ok=False)


    pipelines = get_and_save_pipelines(pipelines_dir=PIPELINES_DIR, logs_dir=LOGS_DIR)

    main(
        pipelines=pipelines,
        seed=config.seed,
        metric=config.metric,
        ds_name=ds_name,
        ds_paths=paths,
        train_worker_timeout=config.train_worker_timeout,
        test_worker_timeout=config.test_worker_timeout,
        workers=config.workers,
        train_outputs_dir=PIPELINE_OUTPUTS,
        test_output_dir=INC_OUTPUT_DIR,
        logs_dir=LOGS_DIR
    )

    save_test_scores_and_pipelines_for_all_datasets(
        results_dir=PIPELINE_OUTPUTS, out_path=pjoin(LOGS_DIR, "test_summary.csv"))


    # python run_enumeration.py --ds_id 359962


    # datasets_ids = [
    #     # 9957,  # qsar-biodeg
    #     # 359958,  # pc4 classification
    #     # 9967,  # steel-plates-fault
    #     359962,  # kc1 classification
    #     #     9978,  # ozone-level-8hr
    #     #     146820,  # wilt
    #     #     43,  # spambase
    #     #     359972,  # sylvin classification
    #     #     9952,  # phoneme
    #     #     361066,  # bank-marketing classification
    #     #     9983,  # eeg-eye-state
    #     #     3899,  # mozilla4
    #     #     9976,  # madelon
    #     #     146606,  # higgs
    #     #     167120,  # numerai28.6
    # ]
