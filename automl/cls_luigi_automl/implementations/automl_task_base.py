import traceback
from logging import Logger, getLogger
from os.path import exists
from os.path import join as pjoin
from os import listdir
import luigi
from cls_luigi.inhabitation_task import LuigiCombinator
from luigi.task import flatten

from cls_luigi.tools.io_functions import dump_json
from .global_parameters import GlobalPipelineParameters


class AutoMLTaskBase(luigi.Task, LuigiCombinator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.global_params = GlobalPipelineParameters()
        self.logger = self.get_luigi_logger()

    def get_luigi_local_target_with_task_id(self,
                                            outfile: str,
                                            ) -> luigi.LocalTarget:

        return luigi.LocalTarget(pjoin(self.global_params.pipelines_outputs_dir, self.task_id + "_" + outfile))

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

    @luigi.Task.event_handler(luigi.Event.TIMEOUT)
    def handle_timeout_task(self, *args):
        self.logger.warning("TIMEOUT handler for task: {}".format(self.task_id))

        try:
            pipeline_tasks = self._get_pipeline_list()

            timeout_report = {
                "task_id": self.task_id,
                "pipeline": [task.task_family for task in pipeline_tasks],
            }
            out_path = pjoin(self.global_params.pipelines_outputs_dir, self.task_id + "_TIMEOUT_1.json")

            if exists(out_path):
                n_time_out = len(
                    [timeout for timeout in listdir(self.global_params.pipelines_outputs_dir) if
                     f"{self.task_id}_TIMEOUT" in timeout])
                self.logger.warning(f"This task has timed out before: {n_time_out} times")
                self.logger.warning("Adjusting file name to include the timeout count")
                out_path = f"{self.global_params.pipelines_outputs_dir}/{self.task_id}_TIMEOUT_{n_time_out + 1}.json"

            dump_json(timeout_report, out_path)

        except Exception as e:
            self.logger.warning("Error for task: {}".format(self.task_id))
            self.logger.warning(e)

    @luigi.Task.event_handler(luigi.Event.FAILURE)
    def handle_failed_task(self, *args):
        self.logger.warning("FAILURE handler for task: {}".format(self.task_id))

        try:
            pipeline_tasks = self._get_pipeline_list()

            traceback_string = traceback.format_exc()

            error_message = "Runtime error:\n%s" % traceback_string

            failure_report = {
                "task_id": self.task_id,
                "error": error_message,
                "pipeline": [task.task_family for task in pipeline_tasks],
            }

            out_path = pjoin(self.global_params.pipelines_outputs_dir, self.task_id + "_FAILURE_1.json")

            if exists(out_path):
                n_failed = len(
                    [timeout for timeout in listdir(self.global_params.pipelines_outputs_dir) if
                     f"{self.task_id}_FAILURE" in timeout])
                self.logger.warning(f"This task has failed before: {n_failed} times")
                self.logger.warning("Adjusting file name to include the failure count")
                out_path = f"{self.global_params.pipelines_outputs_dir}/{self.task_id}_FAILURE_{n_failed + 1}.json"

            dump_json(failure_report, out_path)

        except Exception as e:
            self.logger.warning("Error for task: {}".format(self.task_id))
            self.logger.warning(e)

    def get_score(self, metric):
        raise NotImplementedError("get_score method not implemented for task {}".format(self.task_id))
