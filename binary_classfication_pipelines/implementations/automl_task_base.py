import traceback
from logging import Logger, getLogger
from os import makedirs
from os.path import exists
from os.path import join as pjoin
from os import listdir
import luigi
from cls_luigi import RESULTS_PATH
from cls_luigi.inhabitation_task import LuigiCombinator
from luigi.task import flatten

from utils.io_methods import dump_json
from .global_parameters import GlobalParameters


class AutoMLTaskBase(luigi.Task, LuigiCombinator):
    worker_timeout = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not exists("logs"):
            makedirs("logs")

        self.global_params = GlobalParameters()
        self.logger = self.get_luigi_logger()

    @staticmethod
    def makedirs_in_not_exist(path: str) -> None:
        if not exists(path):
            makedirs(path)

    def _make_and_get_output_folder(self,
                                    output_folder: str = RESULTS_PATH,
                                    dataset_name: int | str = None
                                    ) -> str:

        if dataset_name is None:
            dataset_name = self.global_params.dataset_name

        dataset_name = self._check_if_numeric_and_cast_to_str(dataset_name)
        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)
        return dataset_outputs_folder

    def get_luigi_local_target_with_task_id(self,
                                            outfile: str,
                                            output_folder: str = RESULTS_PATH,
                                            dataset_name: int | str = None
                                            ) -> luigi.LocalTarget:

        dataset_outputs_folder = self._make_and_get_output_folder(output_folder, dataset_name)
        return luigi.LocalTarget(pjoin(dataset_outputs_folder, self.task_id + "_" + outfile))

    def get_luigi_local_target_without_task_id(self,
                                               outfile, output_folder=RESULTS_PATH,
                                               dataset_name: str | int = None
                                               ) -> luigi.LocalTarget:

        if dataset_name is None:
            dataset_name = self.global_params.dataset_name

        dataset_name = self._check_if_numeric_and_cast_to_str(dataset_name)

        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)

        return luigi.LocalTarget(pjoin(dataset_outputs_folder, outfile))

    @staticmethod
    def _check_if_numeric_and_cast_to_str(dataset_name: str | int) -> str:
        if isinstance(dataset_name, (int, float)):
            return str(dataset_name)
        return dataset_name

    def _log_warnings(self, warning_list: list) -> None:
        if len(warning_list) > 0:
            luigi_logger = self.get_luigi_logger()
            for w in warning_list:
                luigi_logger.warning("{}: {}".format(self.task_id, w.message))

    @staticmethod
    def get_luigi_logger() -> Logger:
        return getLogger('luigi-root')

    def _get_pipeline_list(self):
        pipeline = list()

        def collect_pipeline(task):
            if task not in pipeline:
                pipeline.append(task)

            children = flatten(task.requires())
            for child in children:
                collect_pipeline(child)

        collect_pipeline(self)
        return pipeline

    def set_worker_timeout_for_all_tasks(self, worker_timeout):
        pipeline_tasks = self._get_pipeline_list()
        for t in pipeline_tasks:
            t.worker_timeout = worker_timeout

    @luigi.Task.event_handler(luigi.Event.TIMEOUT)
    def on_timeout(self, *args):
        self.logger.warning("TIMEOUT handler for task: {}".format(self.task_id))

        try:
            dataset_outputs_folder = self._make_and_get_output_folder()
            pipeline_tasks = self._get_pipeline_list()

            timeout_report = {
                "task_id": self.task_id,
                "pipeline": [task.task_family for task in pipeline_tasks],
            }
            out_path = f"{dataset_outputs_folder}/{self.task_id}_TIMEOUT_1.json"

            if exists(out_path):
                n_time_out = len(
                    [timeout for timeout in listdir(dataset_outputs_folder) if f"{self.task_id}_TIMEOUT" in timeout])
                self.logger.warning(f"This task has timed out before: {n_time_out} times")
                self.logger.warning("Adjusting file name to include the timeout count")
                out_path = f"{dataset_outputs_folder}/{self.task_id}_TIMEOUT_{n_time_out + 1}.json"

            dump_json(timeout_report, out_path)

        except Exception as e:
            self.logger.warning("Error for task: {}".format(self.task_id))
            self.logger.warning(e)

    @luigi.Task.event_handler(luigi.Event.FAILURE)
    def on_failure(self, *args):
        self.logger.warning("FAILURE handler for task: {}".format(self.task_id))

        try:
            dataset_outputs_folder = self._make_and_get_output_folder()
            pipeline_tasks = self._get_pipeline_list()

            traceback_string = traceback.format_exc()

            error_message = "Runtime error:\n%s" % traceback_string

            failure_report = {
                "task_id": self.task_id,
                "error": error_message,
                "pipeline": [task.task_family for task in pipeline_tasks],
            }

            out_path = f"{dataset_outputs_folder}/{self.task_id}_FAILURE_1.json"
            if exists(out_path):
                n_failed = len(
                    [timeout for timeout in listdir(dataset_outputs_folder) if f"{self.task_id}_FAILURE" in timeout])
                self.logger.warning(f"This task has failed before: {n_failed} times")
                self.logger.warning("Adjusting file name to include the failure count")
                out_path = f"{dataset_outputs_folder}/{self.task_id}_FAILURE_{n_failed + 1}.json"

            dump_json(failure_report, out_path)

        except Exception as e:
            self.logger.warning("Error for task: {}".format(self.task_id))
            self.logger.warning(e)
