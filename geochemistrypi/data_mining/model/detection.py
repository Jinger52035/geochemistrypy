# -*- coding: utf-8 -*-
# import sys
import pandas as pd
from ._base import WorkflowBase
from typing import Optional, Union, Dict
class DetectionWorkflowBase(WorkflowBase):

    def __init__(self) -> None:
        super().__init__()
        # These two attributes are used for the customized models of FLAML framework
        self.customized = False
        self.customized_name = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None) -> None:
        self.X = X
        self.model.fit(X)

class IsolationForest(DetectionWorkflowBase):
    pass
class LOF(DetectionWorkflowBase):
    pass

class OneClasSVM(DetectionWorkflowBase):
    pass

class SGDOneClassSVM(DetectionWorkflowBase):
    pass

class EllipticEnvelope(DetectionWorkflowBase):
    pass





