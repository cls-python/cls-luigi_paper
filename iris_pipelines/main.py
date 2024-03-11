# *************************************************************************
# Step 1: Define LoadIris Component
# *************************************************************************

import luigi
from cls_luigi.inhabitation_task import LuigiCombinator


class LoadIris(luigi.Task, LuigiCombinator):

    def output(self):
        return luigi.LocalTarget("iris.csv")

    def run(self):
        from sklearn.datasets import load_iris
        ds = load_iris(as_frame=True).frame
        ds.to_csv(self.output().path, index=False)


# *************************************************************************
# Step 2 : Define Abstract Scaler Component
# *************************************************************************

from cls_luigi.inhabitation_task import ClsParameter
import pandas as pd


class Scaler(luigi.Task, LuigiCombinator):
    abstract = True  # placeholder
    iris = ClsParameter(tpe=LoadIris.return_type())

    def requires(self):
        return self.iris()  # from line 6

    def output(self):
        return luigi.LocalTarget(f"scaled_iris_{self.task_id}.csv")

    def run(self):
        # load dataset from previous component
        ds = pd.read_csv(self.input().path)
        feats = ds.drop(columns="target", axis=1)

        # transform features and save
        scaler = self.get_scaler()
        ds[feats.columns] = scaler.fit_transform(feats)
        ds.to_csv(self.output().path, index=False)

    def get_scaler(self):
        return NotImplementedError


# *************************************************************************
# Step 3: Define Two Concrete Scaler Components
# *************************************************************************

class MinMaxScaling(Scaler):
    abstract = False

    def get_scaler(self):
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()


class RobustScaling(Scaler):
    abstract = False

    def get_scaler(self):
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()


# *************************************************************************
# Step 4: Define Abstract & Concrete Classifier Components
# *************************************************************************


class Classifier(luigi.Task, LuigiCombinator):
    abstract = True  # placeholder
    scaled_ds = ClsParameter(tpe=Scaler.return_type())

    def requires(self):
        return self.scaled_ds()

    def output(self):
        return luigi.LocalTarget(f"y_pred-{self.task_id}.csv")

    def run(self):
        # load data from previous component
        ds = pd.read_csv(self.input().path)
        X = ds.drop(columns="target", axis=1)
        y = ds["target"]

        # fit classifier
        clf = self.get_classifier()
        clf.fit(X, y)

        # predict and save predictions
        y_pred = pd.DataFrame(data=clf.predict(X), columns=["y_pred"])
        y_pred.to_csv(self.output().path, index=False)

    def get_classifier(self):
        return NotImplementedError


class DecisionTree(Classifier):
    abstract = False

    def get_classifier(self):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier()


class RandomForest(Classifier):
    abstract = False

    def get_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()


# *************************************************************************
# Step 5: Synthesize and Run Pipelines
# *************************************************************************

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.inhabitation_task import RepoMeta

# collect components and set target
repository = RepoMeta.repository
target = Classifier.return_type()

# build tree-grammar and synthesize pipelines
fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
results = fcl.inhabit(target)

# restrict number of pipelines to 10 if infinite
num_pipes = results.size()  # returns -1 if infinite
num_pipes = 10 if num_pipes == -1 else num_pipes
pipes = [t() for t in results.evaluated[0:num_pipes]]

# handover pipelines to luigi scheduler
luigi.build(pipes, local_scheduler=True)
