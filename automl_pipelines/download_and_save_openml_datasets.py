import warnings

import numpy as np
from openml import tasks
import os

from sklearn.model_selection import train_test_split


def download_and_save_openml_dataset(dataset_id, seed):
    with warnings.catch_warnings(record=True) as w:

        paths = {}

        X, y, ds_name = _get_openml_dataset(dataset_id)
        y = _encode_classification_labels(y)
        X = _drop_unnamed_col(X)

        task = tasks.get_task(dataset_id)
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

        test_dataset_dir = f"datasets/{ds_name}/testing_phase"
        os.makedirs(test_dataset_dir, exist_ok=True)


        x_train_path = os.path.join(test_dataset_dir, "test_phase.csv")
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

        train_dataset_dir = f"datasets/{ds_name}/train_phase"
        os.makedirs(train_dataset_dir, exist_ok=True)

        # train validation split
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.33, random_state=seed)

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

    if isinstance(classes[0], str) and isinstance(classes[1], str):
        if classes[0].isnumeric() and classes[1].isnumeric():
            y = y.map(lambda x: 0 if x == classes[0] else 1)
        elif classes[0].isnumeric() is False and classes[1].isnumeric() is False:
            y = y.map(lambda x: 0 if x == "neg" else 1)
        else:
            raise TypeError("Label is string but neither numeric or neg/pos")
    elif (isinstance(classes[0], bool) and isinstance(classes[1], bool)) or \
            (isinstance(classes[0], np.bool_) and isinstance(classes[1], np.bool_)):
        y = y.map(lambda x: 0 if x == False else 1)

    else:
        raise TypeError("Label is not string nor bool")

    return y


def _drop_unnamed_col(df):
    unnamed_col = "Unnamed: 0"

    if unnamed_col in list(df.columns):
        return df.drop([unnamed_col], axis=1)
    return df


if __name__ == "__main__":

    task_list = [
        361066,  # bank-marketing classification
        # 146820,  # wilt classification
        # 168868,  # APSFailure classification
        # 168911,  # jasmine classification
        # 168350,  # phoneme classification contains negative values
        359958,  # pc4 classification
        359962,  # kc1 classification
        359972,  # sylvin classification
        # 359990,  # MiniBooNE classification
        # 146606,  #higgs
    ]

    for task_id in task_list:
        download_and_save_openml_dataset(task_id)