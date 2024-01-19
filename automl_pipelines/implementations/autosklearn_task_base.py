import json
import traceback
from logging import Logger, getLogger
from os import makedirs
from os.path import exists
from os.path import join as pjoin
import luigi
from cls_luigi import RESULTS_PATH
from cls_luigi.inhabitation_task import LuigiCombinator

from .global_parameters import GlobalParameters


class AutoSklearnTask(luigi.Task, LuigiCombinator):
    worker_timeout = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not exists("logs"):
            makedirs("logs")

        self.global_params = GlobalParameters()

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

        dataset_name = self._check_if_int_and_cast_to_str(dataset_name)
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

        dataset_name = self._check_if_int_and_cast_to_str(dataset_name)

        dataset_outputs_folder = pjoin(output_folder, dataset_name)
        self.makedirs_in_not_exist(dataset_outputs_folder)

        return luigi.LocalTarget(pjoin(dataset_outputs_folder, outfile))

    @staticmethod
    def _check_if_int_and_cast_to_str(dataset_name: str | int) -> str:
        if isinstance(dataset_name, int):
            dataset_name = str(dataset_name)
        return dataset_name

    def _log_warnings(self, warning_list: list) -> None:
        if len(warning_list) > 0:
            luigi_logger = self.get_luigi_logger()
            for w in warning_list:
                luigi_logger.warning("{}: {}".format(self.task_id, w.message))

    @staticmethod
    def get_luigi_logger() -> Logger:
        return getLogger('luigi-root')

    def _get_upstream_tasks(self):
        def _get_upstream_tasks_recursively(task, upstream_list=None):
            if upstream_list is None:
                upstream_list = []

            if task not in upstream_list:
                upstream_list.append(task)

            requires = task.requires()
            if requires:
                if isinstance(requires, luigi.Task):
                    if requires not in upstream_list:
                        upstream_list.append(requires)
                    _get_upstream_tasks_recursively(requires, upstream_list)
                elif isinstance(requires, dict):
                    for key, value in requires.items():
                        if value not in upstream_list:
                            upstream_list.append(value)
                        _get_upstream_tasks_recursively(value, upstream_list)
            return upstream_list

        return _get_upstream_tasks_recursively(self)

    def on_failure(self, exception):
        dataset_outputs_folder = self._make_and_get_output_folder()
        upstream_tasks = self._get_upstream_tasks()

        traceback_string = traceback.format_exc()

        error_message = "Runtime error:\n%s" % traceback_string

        failure_report = {
            "task_id": self.task_id,
            "error": error_message,
            "upstream_tasks": [task.task_family for task in upstream_tasks],
        }

        with open(f"{dataset_outputs_folder}/{self.task_id}_FAILURE.json", "w") as f:
            json.dump(failure_report, f, indent=4)

        return error_message


class TaskTimeOutHandler(object):
    @luigi.Task.event_handler(luigi.Event.TIMEOUT)
    def on_timeout(self, *args):
        dataset_outputs_folder = self._make_and_get_output_folder()
        upstream_tasks = self._get_upstream_tasks()

        timeout_report = {
            "task_id": self.task_id,
            "upstream_tasks": [task.task_family for task in upstream_tasks],
        }
        with open(f"{dataset_outputs_folder}/{self.task_id}_TIMEOUT.json", "w") as f:
            json.dump(timeout_report, f, indent=4)
