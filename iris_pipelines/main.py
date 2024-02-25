# *************************************************************************
# Step 1: Define Loading Data Component
# *************************************************************************

import luigi
from cls_luigi.inhabitation_task import LuigiCombinator
import pandas as pd

class LoadIrisDataset(luigi.Task, LuigiCombinator):

    def output(self):
        return {
            "X": luigi.LocalTarget("X.csv"),
            "y": luigi.LocalTarget("y.csv")}

    def run(self):
        from sklearn.datasets import load_iris

        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target

        X.to_csv(self.output()["X"].path, index=False)
        y.to_csv(self.output()["y"].path, index=False)


# *************************************************************************
# Step 2 : Define Abstract Classifier Component
# *************************************************************************

from cls_luigi.inhabitation_task import ClsParameter

class FitPredictClassifier(luigi.Task, LuigiCombinator):
    abstract = True # place holder
    iris = ClsParameter(tpe=LoadIrisDataset.return_type())

    def requires(self):
        return self.iris() # from line 5

    def output(self):
        variant_label = self.task_id
        return {"y_pred": luigi.LocalTarget(f"y_pred-{variant_label}.csv")}

    def run(self):
        # load data from previous component
        X = pd.read_csv(self.input()["X"].path)
        y = pd.read_csv(self.input()["y"].path)

        #fit classifier
        clf = self.get_classifier()
        clf.fit(X, y)

        #predict and save predictions
        y_pred = pd.DataFrame(data=clf.predict(X), columns=["y_pred"])
        y_pred.to_csv(self.output()["y_pred"].path, index=False)

    def get_classifier(self):
        return NotImplementedError

# *************************************************************************
# Step 3: Define Two Concrete Classifier Components
# *************************************************************************


class FitPredictDecisionTree(FitPredictClassifier):
    abstract = False

    def get_classifier(self):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier()


class FitPredictRandomForest(FitPredictClassifier):
    abstract = False

    def get_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()


# *************************************************************************
# Step 4: Synthesize and Run Pipelines
# *************************************************************************

from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes
from cls_luigi.inhabitation_task import RepoMeta

# collect components and set target
repository = RepoMeta.repository
target = FitPredictClassifier.return_type()

# build tree-grammar and synthesize pipelines
fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes))
results = fcl.inhabit(target)

# restrict number of pipelines to 10 if infinite
num_pipes = results.size() # returns -1 if infinite
num_pipes = 10 if num_pipes == -1 else num_pipes
pipes = [t() for t in results.evaluated[0:num_pipes]]

# handover pipelines to luigi scheduler
luigi.build(pipes, local_scheduler=True)