import numpy as np
import utils


class DecisionStump:

    def predict(self, X):
        if X[1] > 37.669007:
            if X[0] > -96.090109:
                return 1
            else:
                return 2
        else:
            # split value is -115.577574
            if X[0] > -115.577574:
                return 2
            else:
                return 1