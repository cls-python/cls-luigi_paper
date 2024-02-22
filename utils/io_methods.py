import pickle
import json


def load_json(path, mode="r", decoder_cls=None):
    with open(path, mode) as f:
        return json.load(f, cls=decoder_cls)


def dump_json(obj, path, mode="w", encoder_cls=None):
    with open(path, mode) as f:
        json.dump(obj, f, indent=4, cls=encoder_cls)


def load_pickle(path, mode="rb"):
    with open(path, mode) as f:
        pickle.load(f)


def dump_pickle(obj, path, mode="wb"):
    with open(path, mode) as f:
        pickle.dump(obj, f)


def dump_txt(string, path, mode="w"):
    with open(path, mode) as f:
        f.write(string)
