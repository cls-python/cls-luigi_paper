import warnings
import numpy as np
import pandas as pd
from openml import tasks
import os
from os.path import join as pjoin
from sklearn.model_selection import train_test_split


def download_and_save_openml_dataset(datasets_dir, dataset_id, seed):
    with warnings.catch_warnings(record=True) as w:
        paths = {}

        task = tasks.get_task(dataset_id)
        X, y = task.get_X_and_y(dataset_format='dataframe')
        y = _encode_classification_labels(y)
        X = _drop_unnamed_col(X)
        ds_name = task.get_dataset().name

        train_indices, test_indices = task.get_train_test_split_indices(
            repeat=0,
            fold=0,
            sample=0,
        )
        # train test split
        x_train = X.iloc[train_indices]
        x_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        test_dataset_dir = pjoin(datasets_dir, f"{ds_name}", f"seed-{seed}", "test_phase")
        os.makedirs(test_dataset_dir, exist_ok=True)

        x_train_path = os.path.join(test_dataset_dir, "x_train.csv")
        y_train_path = os.path.join(test_dataset_dir, "y_train.csv")
        x_test_path = os.path.join(test_dataset_dir, "x_test.csv")
        y_test_path = os.path.join(test_dataset_dir, "y_test.csv")

        x_train.to_csv(x_train_path, index=False)
        x_test.to_csv(x_test_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        paths["test_phase"] = {
            "x_train_path": x_train_path,
            "y_train_path": y_train_path,
            "x_test_path": x_test_path,
            "y_test_path": y_test_path,
        }

        # train_dataset_dir = f"datasets/{ds_name}/{seed}/train_phase"
        train_dataset_dir = pjoin(datasets_dir, f"{ds_name}", f"seed-{seed}", "train_phase")

        os.makedirs(train_dataset_dir, exist_ok=True)

        # train validation split
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.33, random_state=seed, shuffle=True)

        x_train_path = os.path.join(train_dataset_dir, "x_train.csv")
        y_train_path = os.path.join(train_dataset_dir, "y_train.csv")
        x_valid_path = os.path.join(train_dataset_dir, "x_valid.csv")
        y_valid_path = os.path.join(train_dataset_dir, "y_valid.csv")

        x_train.to_csv(x_train_path, index=False)
        x_valid.to_csv(x_valid_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        y_valid.to_csv(y_valid_path, index=False)

        paths["train_phase"] = {
            "x_train_path": x_train_path,
            "y_train_path": y_train_path,
            "x_valid_path": x_valid_path,
            "y_valid_path": y_valid_path,
        }

        return ds_name, paths


def _get_openml_dataset(task_id):
    task = tasks.get_task(task_id)
    X, y = task.get_X_and_y(dataset_format='dataframe')
    d_name = task.get_dataset().name

    return X, y, d_name


# todo
def _encode_classification_labels(y):
    classes = sorted(list(y.unique()))
    assert len(classes) == 2, "There exists more than two classes!"

    if isinstance(classes[0], (int, float, str, np.int64)) and isinstance(classes[1], (int, float, str, np.int64)):
        y = y.map(lambda x: 0 if x == classes[0] else 1)

    elif isinstance(classes[0], (bool, np.bool_)) and isinstance(classes[1], (bool, np.bool_)):
        y = y.map(lambda x: 0 if x == False else 1)

    else:
        raise TypeError("Label is not string, bool, or numeric")

    return y


def _drop_unnamed_col(df):
    unnamed_col = "Unnamed: 0"

    if unnamed_col in list(df.columns):
        return df.drop([unnamed_col], axis=1)
    return df


def get_dataset_information(task_id):
    x, _, ds_name = _get_openml_dataset(task_id)

    n_cols = x.shape[1]
    n_rows = x.shape[0]
    n_nans = np.count_nonzero(np.isnan(x))
    n_cols_with_nans = 0

    for c in x.columns:
        n_cols_with_nans += np.count_nonzero(np.isnan(x[c]))

    x["n_nans"] = x.apply(lambda row: row.isna().sum(), axis=1)

    n_rows_with_nans = x[x["n_nans"] > 0].shape[0]

    return ds_name, n_cols, n_rows, n_nans, n_cols_with_nans, n_rows_with_nans


def gather_dataset_information(dataset_ids, out_path="datasets/datasets_info.csv"):
    info_dict = {
        "ds_name": [],
        "n_cols": [],
        "n_rows": [],
        "n_nans": [],
        "n_cols_with_nans": [],
        "n_rows_with_nans": []
    }
    for ds in dataset_ids:
        _ds_name, _n_cols, _n_rows, _n_nans, _n_cols_with_nans, _n_rows_with_nans = get_dataset_information(ds)
        info_dict["ds_name"].append(_ds_name)
        info_dict["n_cols"].append(_n_cols)
        info_dict["n_rows"].append(_n_rows)
        info_dict["n_nans"].append(_n_nans)
        info_dict["n_cols_with_nans"].append(_n_cols_with_nans)
        info_dict["n_rows_with_nans"].append(_n_rows_with_nans)

    info_df = pd.DataFrame.from_dict(info_dict)
    info_df.to_csv(out_path, index=False)


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Seed for splitting the datasets")

    parser.add_argument("--download_to_dir",
                        type=str,
                        default=os.getcwd(),
                        help="Datasets are saved by default in /datasets in the current working directory")

    parser.add_argument('--datasets_ids',
                        type=list_of_ints,
                        default=[
                            9967,  # steel-plates-fault
                            9957,  # qsar-biodeg
                            9952,  # phoneme
                            9978,  # ozone-level-8hr
                            145847,  # hill-valley
                            146820,  # wilt
                            3899,  # mozilla4
                            9983,  # eeg-eye-state
                            359962,  # kc1 classification
                            359958,  # pc4 classification
                            361066,  # bank-marketing classification
                            359972,  # sylvin classification
                            167120,  # numerai28.6
                            9976,  # Madelon
                            146606,  # higgs
                            168868,  # APSFailure
                            168338,  # riccardo
                        ],
                        help="List of dataset IDs to be downloaded from OpenML")


    parsed = parser.parse_args()


    for dataset_id in parsed.datasets_ids:
        try:
            ds_name, _ = download_and_save_openml_dataset(
                datasets_dir=pjoin(parsed.download_to_dir, "datasets"),
                dataset_id=dataset_id,
                seed=parsed.seed)
            print(f"downloaded dataset {ds_name}")
        except:
            print(f"problem to download dataset with id {dataset_id}")
