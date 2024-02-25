import time
import luigi
import numpy as np
import pyepo

from cls_luigi.inhabitation_task import ClsParameter
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from torch.optim import SGD, Adam
from tqdm import tqdm
from xgboost import XGBRegressor
import lightgbm as lgb
from base_task import BaseTaskClass
from torch.utils.data import DataLoader
import torch

from models.pyepo_gurobi_shortest_path_solver import PyEPOGurobiShortestPathSolver
from torch_dataset import TorchDataset
from models.nn_regressor import fcNet
from utils.time_recorder import TimeRecorder

luigi.interface.core.log_level = "WARNING"


class SyntheticDataGenerator(BaseTaskClass):
    abstract = False

    def run(self):
        print("=====SyntheticDataGenerator=====")
        with TimeRecorder(self.output()["run_time"].path):
            np.random.seed(self.global_params.seed)

            features, costs = pyepo.data.shortestpath.genData(
                num_data=self.global_params.num_data + 1000,
                num_features=self.global_params.num_features,
                grid=self.global_params.grid,
                deg=self.global_params.deg,
                noise_width=self.global_params.noise_width,
                seed=self.global_params.seed
            )
            print("Generated shortest-path Dataset...")

            x_train, x_test, y_train, y_test = train_test_split(
                features,
                costs,
                test_size=1000,
                random_state=self.global_params.seed
            )

            self.dump_pickle(x_train, self.output()["x_train"].path)
            self.dump_pickle(x_test, self.output()["x_test"].path)
            self.dump_pickle(y_train, self.output()["y_train"].path)
            self.dump_pickle(y_test, self.output()["y_test"].path)
            print("Split and saved dataset Data...")

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl"),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }


class GenerateOptimalSolutionsAndObjectiveValues(BaseTaskClass):
    abstract = False
    split_dataset = ClsParameter(tpe=SyntheticDataGenerator.return_type())

    def requires(self):
        return {"split_dataset": self.split_dataset()}

    def run(self):
        print("=====GenerateOptimalSolutionsAndObjectiveValues=====")
        with TimeRecorder(self.output()["run_time"].path):
            solver = PyEPOGurobiShortestPathSolver(grid_size=self.global_params.grid)

            y_train = self.load_pickle(self.input()["split_dataset"]["y_train"].path)
            train_optimal_sols, train_optimal_objs = solver._get_solutions_and_objective_values(y_train)
            self.dump_pickle(train_optimal_sols, self.output()["train_optimal_sols"].path)
            self.dump_pickle(train_optimal_objs, self.output()["train_optimal_objs"].path)

            y_test = self.load_pickle(self.input()["split_dataset"]["y_test"].path)
            test_optimal_sols, test_optimal_objs = solver._get_solutions_and_objective_values(y_test)
            self.dump_pickle(test_optimal_sols, self.output()["test_optimal_sols"].path)
            self.dump_pickle(test_optimal_objs, self.output()["test_optimal_objs"].path)
            print("Saved optimal solutions and objective values...")

    def output(self):
        return {
            "train_optimal_sols": self.get_luigi_local_target_with_task_id("train_optimal_sols.pkl"),
            "train_optimal_objs": self.get_luigi_local_target_with_task_id("train_optimal_objs.pkl"),
            "test_optimal_sols": self.get_luigi_local_target_with_task_id("test_optimal_sols.pkl"),
            "test_optimal_objs": self.get_luigi_local_target_with_task_id("test_optimal_objs.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }


class SolutionApproach(BaseTaskClass):
    abstract = True
    split_dataset = ClsParameter(tpe=SyntheticDataGenerator.return_type())

    def requires(self):
        return {"split_dataset": self.split_dataset()}

    def output(self):
        return {
            "test_predictions": self.get_luigi_local_target_with_task_id("test_predictions.pkl"),
            "fitted_model": self.get_luigi_local_target_with_task_id("fitted_model.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }


class TwoStageSolution(SolutionApproach):
    abstract = True


class SKLMultiOutputRegressionModel(TwoStageSolution):
    abstract = True
    regressor = None

    def run(self):
        print(f"====={self.__class__.__name__}=====")

        with TimeRecorder(self.output()["run_time"].path):
            x_train, x_test, y_train, _ = self._load_split_dataset()

            self._init_multioutput_regressor()
            print(f"Training {self.__class__.__name__}...")
            self.regressor.fit(x_train, y_train)

            print(f"Predicting {self.__class__.__name__}...")
            test_predictions = self.regressor.predict(x_test)

            self.dump_pickle(test_predictions, self.output()["test_predictions"].path)
            self.dump_pickle(self.regressor, self.output()["fitted_model"].path)
            print(f"Saved predictions and model for {self.__class__.__name__}...")

    def _load_split_dataset(self):
        x_train = self.load_pickle(self.input()["split_dataset"]["x_train"].path)
        x_test = self.load_pickle(self.input()["split_dataset"]["x_test"].path)
        y_train = self.load_pickle(self.input()["split_dataset"]["y_train"].path)
        y_test = self.load_pickle(self.input()["split_dataset"]["y_test"].path)
        return x_train, x_test, y_train, y_test

    def _init_multioutput_regressor(self):
        return NotImplementedError


class RandomForestModel(SKLMultiOutputRegressionModel):
    abstract = False

    def _init_multioutput_regressor(self):
        self.regressor = MultiOutputRegressor(
            estimator=RandomForestRegressor(
                n_jobs=self.global_params.n_jobs,
                random_state=self.global_params.seed
            ),
            n_jobs=self.global_params.n_jobs,
        )


class LinearRegressionModel(SKLMultiOutputRegressionModel):
    abstract = False

    def _init_multioutput_regressor(self):
        self.regressor = MultiOutputRegressor(
            estimator=LinearRegression(
                n_jobs=self.global_params.n_jobs
            ),
            n_jobs=self.global_params.n_jobs,
        )


class LightGBMModel(SKLMultiOutputRegressionModel):
    abstract = False

    def _init_multioutput_regressor(self):
        self.regressor = MultiOutputRegressor(
            estimator=XGBRegressor(
                n_jobs=self.global_params.n_jobs,
                seed=self.global_params.seed
            ),
            n_jobs=self.global_params.n_jobs,
        )


class LightGBMModelLinearTree(SKLMultiOutputRegressionModel):
    abstract = False

    def _init_multioutput_regressor(self):
        self.regressor = MultiOutputRegressor(
            estimator=lgb.LGBMRegressor(
                n_jobs=self.global_params.n_jobs,
                random_state=self.global_params.seed,
                verbose=-1,
                linear_tree=True
            ),

            n_jobs=self.global_params.n_jobs,
        )


class EndToEndLearning(SolutionApproach):
    abstract = True


class SPOPlus(EndToEndLearning):
    abstract = False
    sols_and_objs = ClsParameter(tpe=GenerateOptimalSolutionsAndObjectiveValues.return_type())

    regressor = None
    optimizer = None
    train_loader = None
    test_loader = None

    test_predictions = None

    def requires(self):
        return {
            "split_dataset": self.split_dataset(),  # already defined in SolutionApproach
            "opt_sols_and_objs": self.sols_and_objs()
        }

    def run(self):
        print("=====SPOPlus=====")
        with TimeRecorder(self.output()["run_time"].path):
            torch.cuda.empty_cache()
            print("Cleared GPU cache...")
            torch.manual_seed(self.global_params.seed)
            torch.cuda.manual_seed(self.global_params.seed)

            device = self._get_device()

            self._init_data_loaders()
            self._init_regressor_and_optimizer(device)
            self._train(device)
            self.test_predictions = self._predict()

            self.dump_pickle(self.test_predictions, self.output()["test_predictions"].path)
            torch.save(self.regressor.state_dict(), self.output()["fitted_model"].path)
            print("Saved predictions and model state...")
            torch.cuda.empty_cache()
            print("Cleared GPU cache again...")

    def _init_data_loaders(self):
        x_train, x_test, y_train, y_test = self._load_split_dataset()
        train_opt_sols, train_opt_objs, test_opt_sols, test_opt_objs = self._load_solutions_and_objective_values()

        train_dataset = TorchDataset(X=x_train, y=y_train, sols=train_opt_sols, objs=train_opt_objs)
        test_dataset = TorchDataset(X=x_test, y=y_test, sols=test_opt_sols, objs=test_opt_objs)

        self.train_loader = DataLoader(train_dataset, batch_size=self.global_params.batch, shuffle=True,
                                       num_workers=self.global_params.n_jobs)
        self.test_loader = DataLoader(test_dataset, batch_size=self.global_params.batch, shuffle=False)
        print("Assigned data loaders...")

    def _init_regressor_and_optimizer(self, device):
        arch = [self.global_params.num_features] + []
        arch.append((self.global_params.grid[0] - 1) * self.global_params.grid[1] + \
                    (self.global_params.grid[1] - 1) * self.global_params.grid[0])

        self.regressor = fcNet(arch).to(device)

        if self.global_params.optimizer == "sgd":
            self.optimizer = SGD(self.regressor.parameters(), lr=self.global_params.learning_rate)
        if self.global_params.optimizer == "adam":
            self.optimizer = Adam(self.regressor.parameters(), lr=self.global_params.learning_rate)

        print(f"Assigned regressor and optimizer {self.global_params.optimizer}")

    def _train(self, device):

        self.regressor.train()
        solver = pyepo.model.grb.shortestPathModel(self.global_params.grid)
        spop = pyepo.func.SPOPlus(solver, processes=self.global_params.n_jobs)
        time.sleep(1)

        pbar = tqdm(range(self.global_params.epochs))
        print("Training SPO+...")
        for epoch in pbar:
            for feats, costs, sols, objs in self.train_loader:
                feats, costs, sols, objs = feats.to(device), costs.to(device), sols.to(device), objs.to(device)
                self.optimizer.zero_grad()
                predicted_costs = self.regressor(feats)
                loss = spop(predicted_costs, costs, sols, objs).mean()
                loss.backward()
                self.optimizer.step()

                desc = "Epoch {}, Loss: {:.4f}".format(epoch, loss.item())
                pbar.set_description(desc)

    def _predict(self):
        self.regressor.eval()

        predictions = None
        print("Predicting SPO+...")
        for feats, costs, sols, objs in tqdm(self.test_loader):
            if next(self.regressor.parameters()).is_cuda:
                feats, costs, sols, objs = feats.cuda(), costs.cuda(), sols.cuda(), objs.cuda()

            with torch.no_grad():
                if predictions is None:
                    predictions = self.regressor(feats).to("cpu").detach().numpy()
                else:
                    predictions = np.concatenate(
                        (
                            predictions,
                            self.regressor(feats).to(
                                "cpu").detach().numpy()), axis=0)
        return predictions

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Devices:")
            for i in range(torch.cuda.device_count()):
                print("    {}:".format(i), torch.cuda.get_device_name(i))
        else:
            device = torch.device("cpu")
            print("Device: CPU")
        return device

    def _load_split_dataset(self):
        x_train = self.load_pickle(self.input()["split_dataset"]["x_train"].path)
        x_test = self.load_pickle(self.input()["split_dataset"]["x_test"].path)
        y_train = self.load_pickle(self.input()["split_dataset"]["y_train"].path)
        y_test = self.load_pickle(self.input()["split_dataset"]["y_test"].path)
        return x_train, x_test, y_train, y_test

    def _load_solutions_and_objective_values(self):
        train_opt_sols = self.load_pickle(self.input()["opt_sols_and_objs"]["train_optimal_sols"].path)
        train_opt_objs = self.load_pickle(self.input()["opt_sols_and_objs"]["train_optimal_objs"].path)
        test_opt_sols = self.load_pickle(self.input()["opt_sols_and_objs"]["test_optimal_sols"].path)
        test_opt_objs = self.load_pickle(self.input()["opt_sols_and_objs"]["test_optimal_objs"].path)

        return train_opt_sols, train_opt_objs, test_opt_sols, test_opt_objs


class GenerateSolutionsForPredictedCosts(BaseTaskClass):
    abstract = False

    predictions = ClsParameter(tpe=SolutionApproach.return_type())
    optimizer = None

    def requires(self):
        return {"predictions": self.predictions()}

    def output(self):
        return {
            "test_prediction_solutions": self.get_luigi_local_target_with_task_id("test_solutions.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }

    def run(self):
        print("=====GenerateSolutionsForPredictedCosts=====")
        with TimeRecorder(self.output()["run_time"].path):
            test_predictions = self.load_pickle(self.input()["predictions"]["test_predictions"].path)
            solver = PyEPOGurobiShortestPathSolver(self.global_params.grid)
            prediction_sols, _ = solver._get_solutions_and_objective_values(test_predictions)

            self.dump_pickle(prediction_sols, self.output()["test_prediction_solutions"].path)
            print("Saved prediction solutions...")


class Evaluation(BaseTaskClass):
    abstract = False
    optimal_sols_and_objs = ClsParameter(tpe=GenerateOptimalSolutionsAndObjectiveValues.return_type())
    predicted_solution = ClsParameter(tpe=GenerateSolutionsForPredictedCosts.return_type())
    predictions = ClsParameter(tpe=SolutionApproach.return_type())
    split_dataset = ClsParameter(tpe=SyntheticDataGenerator.return_type())

    summary = {}
    mse = None
    regret = None

    def requires(self):
        return {
            "optimal_sols_and_objs": self.optimal_sols_and_objs(),
            "predicted_sols_and_objs": self.predicted_solution(),
            "predictions": self.predictions(),
            "split_dataset": self.split_dataset()
        }

    def run(self):
        print("=====Evaluation=====")

        with TimeRecorder(self.output()["run_time"].path):
            true_objective_values = self.load_pickle(self.input()["optimal_sols_and_objs"]["test_optimal_objs"].path)
            optimal_solutions = self.load_pickle(self.input()["optimal_sols_and_objs"]["test_optimal_sols"].path)
            prediction_solutions = self.load_pickle(
                self.input()["predicted_sols_and_objs"]["test_prediction_solutions"].path)

            predictions = self.load_pickle(self.input()["predictions"]["test_predictions"].path)
            y_test = self.load_pickle(self.input()["split_dataset"]["y_test"].path)

            self._compute_and_write_regret_in_summary(prediction_solutions, y_test, true_objective_values,
                                                      optimal_solutions)
            self._compute_and_write_mse_in_summary(predictions, y_test)
            self._write_pipeline_steps()
            self._save_outputs()

    def _compute_and_write_regret_in_summary(self, predicted_sols, true_costs, true_objs, optimal_solutions,
                                             minimization=True):
        self.regret = 0
        for index, predicted_sol in enumerate(predicted_sols):
            true_cost = true_costs[index]
            true_obj = true_objs[index]

            if minimization is True:
                _regret = np.dot(predicted_sol, true_cost) - true_obj

            elif minimization is False:
                _regret = true_obj - np.dot(predicted_sol, true_cost)

            if _regret < 0:
                if np.array_equal(predicted_sol, optimal_solutions[index]):
                    _regret = 0

            self.regret += _regret

        self.regret /= true_objs.sum()
        if self.regret < 0:
            print("This shouldn't happen: The Regret is negative")

        self.summary["regret"] = self.regret[0]
        print("Computed regret...")

    def _compute_and_write_mse_in_summary(self, predictions, true_costs):
        self.mse = ((predictions - true_costs) ** 2).mean()
        self.summary["mse"] = self.mse
        print("Computed MSE...")

    def _write_pipeline_steps(self):
        self.summary["regressor"] = self.requires()["predictions"].task_family

    def _save_outputs(self):
        self.dump_json(self.summary, self.output()["summary"].path)
        print("Saved summary json")

    def output(self):
        return {
            "summary": self.get_luigi_local_target_with_task_id("summary.json"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")

        }
