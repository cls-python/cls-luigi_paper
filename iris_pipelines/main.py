from cls.fcl import FiniteCombinatoryLogic
from cls.subtypes import Subtypes

from cls_luigi.inhabitation_task import LuigiCombinator
from cls_luigi.inhabitation_task import ClsParameter
from cls_luigi.inhabitation_task import RepoMeta

import luigi
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# Step 1: Download and save iris dataset
class LoadIrisDataset(luigi.Task, LuigiCombinator):

    def requires(self):
        return None # no dependencies

    def output(self):
        return {
            "X": luigi.LocalTarget("X.csv"),
            "y": luigi.LocalTarget("y.csv")}

    def run(self):
        from sklearn.datasets import load_iris
        # get data 
        iris = load_iris(as_frame=True)
        X = iris.data
        y = iris.target
        #save data
        X.to_csv(self.output()["X"].path, index=False)
        y.to_csv(self.output()["y"].path, index=False)
        
        
        
# Step 2: Implement an abstract class for classifiers
class FitPredictClassifier(luigi.Task, LuigiCombinator):
    abstract = True # place holder
    iris = ClsParameter(tpe=LoadIrisDataset.return_type())

    def requires(self):
        return self.iris() # from line 5

    def output(self):
        variant_label = self.task_id
        return {
        "y_pred": luigi.LocalTarget(
            f"y_pred-{variant_label}.csv")}

    def run(self):
        # load data from previous component 
        X = pd.read_csv(self.input()["X"].path)
        y = pd.read_csv(self.input()["y"].path)

        #fit classifier
        clf = self.get_classifier()
        clf.fit(X, y)

        #predict and save predictions
        y_pred = pd.DataFrame(
            data=clf.predict(X),
            columns=["y_pred"])

        y_pred.to_csv(
            self.output()["y_pred"].path, index=False)

    def get_classifier(self):
        return NotImplementedError 
    
    

# Step 3: Implement concrete classifiers
class FitPredictDecisionTree(FitPredictClassifier):
    abstract = False
    
    def get_classifier(self):
        return DecisionTreeClassifier()


class FitPredictRandomForest(FitPredictClassifier):
    abstract = False
    
    def get_classifier(self):
        return RandomForestClassifier()
    
    
    

# Step 4: Synthesize and run pipelines
if __name__ == "__main__":

    # collect components
    target = FitPredictClassifier.return_type()
    repository = RepoMeta.repository

    # build repository
    fcl = FiniteCombinatoryLogic(
        repository,
        Subtypes(RepoMeta.subtypes),
        processes=1)

    # synthesize pipelines
    inhabitation_result = fcl.inhabit(target)
    
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if actual > 0:
        max_results = actual
    results = [t() for t in inhabitation_result.evaluated[0: max_results]]

    # run 
    luigi.build(results, local_scheduler=True)