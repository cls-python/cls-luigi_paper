from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from ..template import Classifier
import warnings
from utils.time_recorder import TimeRecorder

class SKLQuadraticDiscriminantAnalysis(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                self.estimator = QuadraticDiscriminantAnalysis(
                    reg_param=0.0,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
