import sys


sys.path.append("..")

from shortest_path_pipelines.shortest_path_template import*

import luigi

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.inhabitation_task import RepoMeta
from cls_luigi.unique_task_pipeline_validator import UniqueTaskPipelineValidator
from global_parameters import GlobalParameters

import utils.luigi_daemon
from utils.time_recorder import TimeRecorder

from shortest_path_pipelines.gather_scores_and_plot import collect_and_save_summaries, draw_comparison_boxplot


luigi.interface.core.log_level = "WARNING"


def generate_and_filter_pipelines():
    target = Evaluation.return_type()
    print("Collecting repo...")
    repository = RepoMeta.repository
    print("Building...")
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
    print("Building grammar tree and inhabiting pipelines...")
    inhabitation_result = fcl.inhabit(target)
    print("Enumerating pipelines...")
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if not actual is None or actual == 0:
        max_results = actual

    print("Filtering using UniqueTaskPipelineValidator...")
    validator = UniqueTaskPipelineValidator(
        [SolutionApproach, TwoStageSolution, SKLMultiOutputRegressionModel,
         EndToEndLearning])
    pipelines = [t() for t in inhabitation_result.evaluated[0:max_results] if validator.validate(t())]
    p1 = [t() for t in inhabitation_result.evaluated[0:max_results]]
    if pipelines:
        print("Number of pipelines", max_results)
        print("Number of pipelines after filtering", len(pipelines))
    return pipelines


def main(pipelines, training_size, deg, noise, seed):

    if pipelines:
        print("Running Pipelines...")

        gp = GlobalParameters()
        gp.num_data = training_size
        gp.deg = deg
        gp.noise_width = noise
        gp.seed = seed
        gp.grid = (5, 5)
        gp.num_features = 5

        gp.dataset_name = "shortest_path-" + "ts_" + str(training_size) + "-deg_" + str(
            deg) + "-noise_" + str(
            noise) + "-seed_" + str(seed)

        gp.batch = 32
        gp.optimizer = "adam"
        gp.epochs = 10
        with utils.luigi_daemon.LuigiDaemon():
            luigi.build(pipelines, local_scheduler=False, detailed_summary=True)


    else:
        print("No pipelines!")


if __name__ == "__main__":
    pipelines = generate_and_filter_pipelines()

    with TimeRecorder("time.json") as tr:

        for training_size in [100, 1000, 5000]:
            for deg in [1, 2, 4, 6]:
                for noise in [0, .5]:
                    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                        main(pipelines, training_size, deg, noise, seed)
                        tr.checkpoint(f"ts:{training_size}-deg:{deg}-noise:{noise}-seed:{seed}")
                    
                    
    summaries_df = collect_and_save_summaries()
    draw_comparison_boxplot(summaries_df)