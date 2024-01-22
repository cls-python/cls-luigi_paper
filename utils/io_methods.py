import pickle
import json


def load_json(path, mode="r"):
    with open(path, mode) as f:
        return json.load(f)


def dump_json(obj, path, mode="w"):
    with open(path, mode) as f:
        try:
            json.dump(obj, f, indent=4)
        except AttributeError:
            print()


def load_pickle(path, mode="rb"):
    with open(path, mode) as f:
        pickle.load(f)


def dump_pickle(obj, path, mode="wb"):
    with open(path, mode) as f:
        pickle.dump(obj, f)