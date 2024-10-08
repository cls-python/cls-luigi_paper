from ..template import Classifier
from sklearn.naive_bayes import GaussianNB
import warnings
from cls_luigi.tools.time_recorder import TimeRecorder

class SKLGaussianNaiveBayes(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = GaussianNB()

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
