from ..template import Classifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from utils.time_recorder import TimeRecorder

class SKLRandomForest(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()

                max_features = int(self.x_train.shape[1] ** 0.5)

                if max_features == 0:
                    max_features = "sqrt"

                self.estimator = RandomForestClassifier(
                    criterion="gini",
                    max_features=max_features,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    bootstrap=True,
                    random_state=self.global_params.seed,
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
