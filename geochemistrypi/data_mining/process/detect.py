# -*- coding: utf-8 -*-
import pandas as pd
from  ..model.detection import DetectionWorkflowBase
from typing import Optional




class DetectModelSelection(object):
    """"""

    def __init__(self, model):
        self.model = model
        self.dtc = DetectionWorkflowBase()

    def activate(self, X: pd.DataFrame, y: Optional[pd.DataFrame] = None, X_train: Optional[pd.DataFrame] = None,
                 X_test: Optional[pd.DataFrame] = None, y_train: Optional[pd.DataFrame] = None,
                 y_test: Optional[pd.DataFrame] = None) -> None:
        if(self.model == "IsolationForest"):
            print("IsolationForest")
        elif(self.model == "LOF"):
            print("LOF")
        elif(self.model == "OneClasSVM"):
            print("OneClasSVM")
        elif(self.model == "SGDOneClassSVM"):
            print("SGDOneClassSVM")
        elif(self.model == "EllipticEnvelope"):
            print("EllipticEnvelope")
        pass