# -*- coding: utf-8 -*-
import os
from typing import Optional

import pandas as pd

from ..model.decomposition import DecompositionWorkflowBase, PCADecomposition, TSNEDecomposition
from ._base import ModelSelectionBase


class DecompositionModelSelection(ModelSelectionBase):
    """Simulate the normal way of invoking scikit-learn decomposition algorithms."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.dcp_workflow = DecompositionWorkflowBase()

    def activate(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        X_train: Optional[pd.DataFrame] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.DataFrame] = None,
    ) -> None:
        """Train by Scikit-learn framework."""

        self.dcp_workflow.data_upload(X=X, y=y, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        if self.model_name == "Principal Component Analysis":
            hyper_parameters = PCADecomposition.manual_hyper_parameters()
            self.dcp_workflow = PCADecomposition(n_components=hyper_parameters["n_components"], svd_solver=hyper_parameters["svd_solver"])
        elif self.model_name == "T-SNE":
            hyper_parameters = TSNEDecomposition.manual_hyper_parameters()
            self.dcp_workflow = TSNEDecomposition(
                n_components=hyper_parameters["n_components"],
                perplexity=hyper_parameters["perplexity"],
                learning_rate=hyper_parameters["learning_rate"],
                n_iter=hyper_parameters["n_iter"],
                early_exaggeration=hyper_parameters["early_exaggeration"],
            )

        self.dcp_workflow.show_info()

        # Use Scikit-learn style API to process input data
        X_reduced = self.dcp_workflow.fit_transform(X)
        self.dcp_workflow.data_upload(X=X)

        # Save the model hyper-parameters
        self.dcp_workflow.save_hyper_parameters(hyper_parameters, self.model_name, os.getenv("GEOPI_OUTPUT_PARAMETERS_PATH"))

        # special components of different algorithms
        self.dcp_workflow.special_components(components_num=hyper_parameters["n_components"], reduced_data=X_reduced)

        # Save decomposition result
        # self.dcp_workflow.data_save(X_reduced, "X reduced", DATASET_OUTPUT_PATH, "Decomposition Result")

        # Save the trained model
        self.dcp_workflow.save_model()
