import json
import warnings

import joblib
import luigi
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from cls_luigi.inhabitation_task import ClsParameter
from .autosklearn_task_base import AutoSklearnTask


# class FeatureProvider(AutoSklearnTask):
#     abstract = True


class LoadAndSplitData(AutoSklearnTask):
    abstract = True

    def requires(self):
        return None

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "y_train": self.get_luigi_local_target_with_task_id("y_train.pkl"),
            "y_test": self.get_luigi_local_target_with_task_id("y_test.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }


class NumericalImputer(AutoSklearnTask):

    abstract = True
    split_dataset = ClsParameter(tpe=LoadAndSplitData.return_type())

    imputer = None
    x_train = None
    x_test = None

    def requires(self):
        return self.split_dataset()

    def _read_split_features(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }

    def _fit_transform_imputer(self):
        with warnings.catch_warnings(record=True) as w:
            self.imputer.fit(self.x_train)

            self.x_train = pd.DataFrame(
                columns=self.x_train.columns,
                data=self.imputer.transform(self.x_train)
            )

            self.x_test = pd.DataFrame(
                columns=self.x_test.columns,
                data=self.imputer.transform(self.x_test)
            )
            self._log_warnings(w)

    def _save_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.imputer, outfile)


# class Scaler(FeatureProvider2):
class Scaler(AutoSklearnTask):

    abstract = True
    imputed_feaatures = ClsParameter(tpe=NumericalImputer.return_type())

    scaler = None
    x_train = None
    x_test = None

    def requires(self):
        return self.imputed_feaatures()

    def _read_split_imputed_features(self):
        self.x_train = pd.read_pickle(self.input()["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["x_test"].path)

    def fit_transform_scaler(self):
        with warnings.catch_warnings(record=True) as w:
            self.scaler.fit(self.x_train)

            self.x_train = pd.DataFrame(
                columns=self.x_train.columns,
                data=self.scaler.transform(self.x_train)
            )
            self.x_test = pd.DataFrame(
                columns=self.x_test.columns,
                data=self.scaler.transform(self.x_test)
            )
            self._log_warnings(w)

    def sava_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.scaler, outfile)

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }


# class FeaturePreprocessor(FeatureProvider2):
class FeaturePreprocessor(AutoSklearnTask):

    abstract = True
    scaled_features = ClsParameter(tpe=Scaler.return_type())
    target_values = ClsParameter(tpe=LoadAndSplitData.return_type())

    feature_preprocessor = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def requires(self):
        return {
            "scaled_features": self.scaled_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "x_train": self.get_luigi_local_target_with_task_id("x_train.pkl"),
            "x_test": self.get_luigi_local_target_with_task_id("x_test.pkl"),
            "fitted_component": self.get_luigi_local_target_with_task_id("fitted_component.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }

    def _read_split_scaled_features(self):
        self.x_train = pd.read_pickle(self.input()["scaled_features"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["scaled_features"]["x_test"].path)

    def _read_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path).values.ravel()
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path).values.ravel()

    def sava_outputs(self):
        self.x_train.to_pickle(self.output()["x_train"].path)
        self.x_test.to_pickle(self.output()["x_test"].path)
        with open(self.output()["fitted_component"].path, 'wb') as outfile:
            joblib.dump(self.feature_preprocessor, outfile)

    def fit_transform_feature_preprocessor(self, x_and_y_required=False, handle_sparse_output=False):
        with warnings.catch_warnings(record=True) as w:
            if x_and_y_required is True:
                assert self.y_train is not None, "y_train is None!"
                self.feature_preprocessor.fit(self.x_train, self.y_train)
            else:
                self.feature_preprocessor.fit(self.x_train, self.y_train)

            if handle_sparse_output is True:
                self.x_train = pd.DataFrame.sparse.from_spmatrix(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_train)
                )

                self.x_test = pd.DataFrame.sparse.from_spmatrix(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_test)
                )

            elif handle_sparse_output is False:

                self.x_train = pd.DataFrame(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_train)
                )
                self.x_test = pd.DataFrame(
                    columns=self.feature_preprocessor.get_feature_names_out(),
                    data=self.feature_preprocessor.transform(self.x_test)
                )
            self._log_warnings(w)


class Classifier(AutoSklearnTask):
    abstract = True
    processed_features = ClsParameter(tpe=FeaturePreprocessor.return_type())
    target_values = ClsParameter(tpe=LoadAndSplitData.return_type())

    estimator = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    y_train_predict = None
    y_test_predict = None
    run_summary = {}

    def requires(self):
        return {
            "processed_features": self.processed_features(),
            "target_values": self.target_values()
        }

    def output(self):
        return {
            "prediction": self.get_luigi_local_target_with_task_id("prediction.pkl"),
            "run_summary": self.get_luigi_local_target_with_task_id("run_summary.json"),
            "fitted_classifier": self.get_luigi_local_target_with_task_id("fitted_component.pkl"),
            "run_time": self.get_luigi_local_target_with_task_id("run_time.json")
        }

    def _read_split_processed_features(self):
        self.x_train = pd.read_pickle(self.input()["processed_features"]["x_train"].path)
        self.x_test = pd.read_pickle(self.input()["processed_features"]["x_test"].path)

    def _read_split_target_values(self):
        self.y_train = pd.read_pickle(self.input()["target_values"]["y_train"].path).values.ravel()
        self.y_test = pd.read_pickle(self.input()["target_values"]["y_test"].path).values.ravel()

    def sava_outputs(self):
        self.y_test_predict.to_pickle(self.output()["prediction"].path)

        with open(self.output()["run_summary"].path, "w") as f:
            json.dump(self.run_summary, f, indent=4)

        with open(self.output()["fitted_classifier"].path, "wb") as outfile:
            joblib.dump(self.estimator, outfile)

    def fit_predict_estimator(self):
        with warnings.catch_warnings(record=True) as w:
            self.estimator.fit(self.x_train, self.y_train)
            self.y_test_predict = pd.DataFrame(
                columns=["y_predict"],
                data=self.estimator.predict(self.x_test))

            self.y_train_predict = pd.DataFrame(
                columns=["y_predict"],
                data=self.estimator.predict(self.x_train))

            self._log_warnings(w)

    def compute_accuracy(self):
        self.run_summary["last_task"] = self.task_id,
        self.run_summary["accuracy"] = {
            "train": round(float(accuracy_score(self.y_train, self.y_train_predict)), 5),
            "test": round(float(accuracy_score(self.y_test, self.y_test_predict)), 5)
        }

        self.run_summary["balanced_accuracy"] = {
            "train": round(float(balanced_accuracy_score(self.y_train, self.y_train_predict)), 5),
            "test": round(float(balanced_accuracy_score(self.y_test, self.y_test_predict)), 5)
        }

    def create_run_summary(self):
        upstream_tasks = self._get_upstream_tasks()

        self.run_summary["pipeline"] = {}

        imputer = list(filter(lambda task: isinstance(task, NumericalImputer), upstream_tasks))
        self._add_component_to_run_summary("imputer", imputer)

        scaler = list(filter(lambda task: isinstance(task, Scaler), upstream_tasks))
        self._add_component_to_run_summary("scaler", scaler)

        feature_preprocessor = list(filter(lambda task: isinstance(task, FeaturePreprocessor), upstream_tasks))
        self._add_component_to_run_summary("feature_preprocessor", feature_preprocessor)

        classifier = list(filter(lambda task: isinstance(task, Classifier), upstream_tasks))
        self._add_component_to_run_summary("classifier", classifier)

        self.compute_accuracy()

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

    def _add_component_to_run_summary(self, component_name, component):
        if component:
            self.run_summary["pipeline"][component_name] = component[0].task_family
        else:
            self.run_summary["pipeline"][component_name] = None
