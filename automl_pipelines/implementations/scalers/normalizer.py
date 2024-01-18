from ..template import Scaler
from sklearn.preprocessing import Normalizer
import warnings
from examples.automl.utils.time_recorder import TimeRecorder

class SKLNormalizer(Scaler):
    abstract = False

    def run(self):
        with warnings.catch_warnings(record=True) as w:
            with TimeRecorder(self.output()["run_time"].path) as time_recorder:
                self.scaler = Normalizer(
                    copy=False
                )
                self._read_split_imputed_features()
                self.fit_transform_scaler()
                self.sava_outputs()
