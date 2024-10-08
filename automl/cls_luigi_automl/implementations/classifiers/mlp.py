from sklearn.neural_network import MLPClassifier

from ..template import Classifier
import warnings
from cls_luigi.tools.time_recorder import TimeRecorder

class SKLMultiLayerPerceptron(Classifier):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self._read_split_original_target_values()
                self._read_split_processed_features()


                num_nodes_per_layer = 32
                hidden_layer_depth = 1
                
                hidden_layer_sizes = tuple(
                    num_nodes_per_layer for i in range(hidden_layer_depth)
                    )
                
                
                self.estimator = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation='relu',
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter = 512,
                    early_stopping=True,
                    n_iter_no_change=32,
                    validation_fraction=0.1,
                    tol=1e-4,
                    solver="adam",
                    batch_size="auto",
                    shuffle=True,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-8,
                    random_state=self.global_params.seed,
                    warm_start = True
                )

                self.fit_predict_estimator()
                self.create_run_summary()
                self.sava_outputs()
