import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append("..")

import pandas as pd
from os import listdir
from os.path import join as pjoin
import math
from utils.io_methods import load_json

def generate_and_save_askl_cls_comparison_csv(askl_results_path="askl/results/"):
    dataset, askl_run_time, cls_luigi_run_time, askl_n_pipelines, cls_luigi_n_pipelines, \
        askl_test_accuracy, cls_luigi_test_accuracy = [], [], [], [], [], [], []

    askl_clf, cls_luigi_clf = [], []
    askl_f_preproc, cls_luigi_f_preproc = [], []
    askl_scaler, cls_luigi_scaler = [], []

    for ds in listdir(askl_results_path):
        if ds == 'madelon':
            continue

        dataset.append(ds)

        # AutoSklearn results
        askl_ds_path = pjoin(askl_results_path, ds)
        askl_stats = load_json(pjoin(askl_ds_path, "smac3-output/run_42/stats.json"))
        askl_run_time.append(askl_stats["wallclock_time_used"])
        askl_n_pipelines.append(askl_stats["submitted_ta_runs"])
        askl_best_pipeline_summary = load_json(pjoin(askl_ds_path, "best_pipeline_summary.json"))
        askl_test_accuracy.append(askl_best_pipeline_summary["test_accuracy"])

        askl_scaler.append(map_component_name(askl_best_pipeline_summary["scaler"]))
        askl_f_preproc.append(map_component_name(askl_best_pipeline_summary["feature_preprocessor"]))
        askl_clf.append(map_component_name(askl_best_pipeline_summary["classifier"]))

        cls_luigi_run_time_json = load_json(pjoin(ROOT, f"automl_pipelines/logs/{ds}_time.json"))
        cls_luigi_run_time.append(cls_luigi_run_time_json["total_seconds"])

        cls_luigi_run_history = pd.read_csv(pjoin(ROOT, f"automl_pipelines/run_histories/{ds}_train_run_history.csv"))
        cls_luigi_n_pipelines.append(cls_luigi_run_history.shape[0])
        cls_test_summary = pd.read_csv(pjoin(ROOT, "automl_pipelines/logs/test_summary.csv"))

        ds_test_summary = cls_test_summary[cls_test_summary["dataset"] == ds]
        cls_luigi_test_accuracy.append(ds_test_summary["test_accuracy"].iloc[0])

        cls_luigi_scaler.append(handle_dtypes(ds_test_summary["scaler"].iloc[0]))
        cls_luigi_f_preproc.append(handle_dtypes(ds_test_summary["feature_preprocessor"].iloc[0]))
        cls_luigi_clf.append(handle_dtypes(ds_test_summary["classifier"].iloc[0]))

    df = pd.DataFrame()

    df["dataset"] = dataset
    df["askl_run_time"] = askl_run_time
    df["cls_luigi_run_time"] = cls_luigi_run_time
    df["askl_n_pipelines"] = askl_n_pipelines
    df["cls_luigi_n_pipelines"] = cls_luigi_n_pipelines

    df["askl_test_accuracy"] = askl_test_accuracy
    df["cls_luigi_test_accuracy"] = cls_luigi_test_accuracy

    df["winner"] = df.apply(return_winner, axis=1)

    df["askl_clf"] = askl_clf
    df["cls_luigi_clf"] = cls_luigi_clf

    df["cls_luigi_scaler"] = cls_luigi_scaler
    df["askl_scaler"] = askl_scaler

    df["askl_f_preproc"] = askl_f_preproc
    df["cls_luigi_f_preproc"] = cls_luigi_f_preproc

    df.sort_values("winner", inplace=True, ascending=False)

    df.to_csv("askl/askl_luigi_comparison.csv", index=False)


def return_winner(row):
    diff = abs(row["askl_test_accuracy"] - row["cls_luigi_test_accuracy"])

    if diff >= 0.01:

        if row["askl_test_accuracy"] > row["cls_luigi_test_accuracy"]:
            return "AutoSklearn"

        elif row["askl_test_accuracy"] < row["cls_luigi_test_accuracy"]:
            return "CLS-Luigi"

        elif row["askl_test_accuracy"] == row["cls_luigi_test_accuracy"]:
            return "Draw"

    return "~"


def map_component_name(c):
    if c not in name_map.keys():
        raise KeyError

    assert c in name_map.keys(), f"component {c} has no key in name map!"

    return name_map[c]


name_map = {
    "fast_ica": "FastICA",
    "feature_agglomeration": "FeatureAgglomeration",
    "kernel_pca": 'KernelPCA',
    "nystroem_sampler": "Nystroem",
    "pca": 'PCA',
    "polynomial": 'PolynomialFeatures',
    "random_trees_embedding": 'RandomTreesEmbedding',
    "kitchen_sinks": 'RBFSampler',
    "extra_trees_preproc_for_classification": 'SelectFromExtraTrees',
    "liblinear_svc_preprocessor": 'SelectFromLinearSVC',
    "select_percentile_classification": 'SelectPercentile',
    "select_rates_classification": 'SelectRates',
    "no_preprocessing": None,

    "minmax": 'MinMaxScaler',
    "normalize": 'Normalizer',
    "power_transformer": 'PowerTransformer',
    "quantile_transformer": 'QuantileTransformer',
    "robust_scaler": 'RobustScaler',
    "standardize": 'StandardScaler',
    "none": None,

    "adaboost": "AdaBoost",
    "bernoulli_nb": "BernoulliNB",
    "decision_tree": "DecisionTree",
    "extra_trees": "ExtraTrees",
    "gaussian_nb": "SKLGaussianNaiveBayes",
    "gradient_boosting": "GradientBoosting",
    "k_nearest_neighbors": "KNearestNeighbors",
    "lda": "SKLLinearDiscriminantAnalysis",
    "liblinear_svc": "LinearSVC",
    "libsvm_svc": "KernelSVC",
    "mlp": "SKLMultinomialNB",
    "passive_aggressive": "PassiveAggressive",
    "qda": "QuadraticDiscriminantAnalysis",
    "random_forest": "RandomForest",
    "sgd": "SGD"
}


def handle_dtypes(value):
    if isinstance(value, str):
        return value[3:]

    if isinstance(value, float):
        if math.isnan(value):
            return None

    if isinstance(value, None):
        return None


if __name__ == "__main__":
    df = generate_and_save_askl_cls_comparison_csv()
    print()
    print("")
