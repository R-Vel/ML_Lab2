from typing import Iterable, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class WhiteBox(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        n_estimators = 100,
        min_samples_split = 10,
        min_samples_leaf = 4,
        max_depth = 3):
        
        """
        Initialize class and random forest model

        Parameters:
        -----------
        random_state (int): Random state (default=42)
        n_estimators (int): Number of trees (default=100)
        min_samples_split (int): Minimum samples needed before splitting (default=10)
        min_samples_leaf (int): Minimum samples needed to be leaf node (default=4)
        max_depth (int): Maximum levels per tree (default=3):
        """

        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self._model_ = RandomForestClassifier(
            n_estimators=500,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            n_estimators=self.n_estimators 
            min_samples_split=self.min_samples_split
            min_samples_leaf=self.min_samples_leaf
            max_depth=self.max_depth
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[Iterable[float]] = None,
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        sample_weight (iterable): Sample class weights (default=None)
        """

        X_balanced, y_balanced = self._handle_imbalance(X, y)
        if sample_weight is None:
            self._model_.fit(X_balanced, y_balanced)
        else:
            self._model_.fit(X_balanced, y_balanced, sample_weight=sample_weight)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict the outcome from data X given threshold."""
        check_is_fitted(self, "_model_")

        probability = self.predict_proba(X)[:, 1]
        return (probability >= self.threshold).astype(int)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        "Return probabilities per class"
        return self._model_.predict_proba(X)
        
    def fit_predict(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)
    
    def _handle_imbalance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        k: int = 3
        ) -> np.ndarray :
        
        """
        Apply SMOTE with 13 nearest neighbors to balance class distribution

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        k (int): K nearest neighbors

        Returns
        -------
        X_train_smote (np.ndarray) : Resampled training data with synthetic minority class datapoints
        y_train_smote (np.ndarray) : The corresponding targets
        """

        smote = SMOTE(random_state=self.random_state, 
                      k_neighbors=k)
        X_train_smote, y_train_smote = smote.fit_resample(X, y)

        return X_train_smote, y_train_smote
