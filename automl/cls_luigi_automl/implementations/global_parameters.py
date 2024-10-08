# import luigi


# class GlobalParameters(luigi.Config):
#     """
#     Global parameters for the pipeline, such as dataset id, name, seed, n_jobs, etc.
#     """

#     x_train_path = luigi.Parameter(default="None")
#     x_test_path = luigi.Parameter(default="None")
#     y_train_path = luigi.Parameter(default="None")
#     y_test_path = luigi.Parameter(default="None")
#     dataset_name = luigi.Parameter(default="None")

#     n_jobs = luigi.IntParameter(default=1)
#     seed = luigi.IntParameter(default=5)


from typing import Dict, Any

import luigi


class GlobalPipelineParameters(luigi.Config):
    """
    Global parameters for the pipeline, such as dataset id, name, seed, n_jobs, etc.
    """

    x_train_path = luigi.OptionalIntParameter(default=None)
    x_test_path = luigi.OptionalIntParameter(default=None)
    y_train_path = luigi.OptionalIntParameter(default=None)
    y_test_path = luigi.OptionalIntParameter(default=None)
    luigi_outputs_dir = luigi.OptionalIntParameter(default=None)
    pipelines_outputs_dir = luigi.OptionalIntParameter(default=None)

    n_jobs = luigi.IntParameter(default=1)
    seed = luigi.IntParameter(default=123456)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Set parameters from a dictionary.
        """
        self.x_train_path = params["x_train_path"]
        self.x_test_path = params["x_test_path"]
        self.y_train_path = params["y_train_path"]
        self.y_test_path = params["y_test_path"]
        self.luigi_outputs_dir = params["luigi_outputs_dir"]
        self.pipelines_outputs_dir = params["pipelines_outputs_dir"]
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
        if "seed" in params:
            self.seed = params["seed"]