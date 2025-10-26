from typing import Iterable, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

class WhiteBox(BaseEstimator, ClassifierMixin):

    def __init__(
        self, 
        random_state: int = 42,
        n_jobs: Optional[int] = None,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        max_depth: int = 3,
        threshold: float = 0.5,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.threshold = threshold
        self._model_ = RandomForestClassifier(
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                n_estimators=self.n_estimators,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
        )

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        """
            def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        """
        self.y = self._map_y(y)
        
        # Encode binary features
        X = X.copy()
        binary_features = ['gender', 'immunizations_done', 'reflexes_normal']
        for col in binary_features:
            if col in X.columns:
                X[col] = (
                    X[col]
                    .astype(str)
                    .str.lower()
                    .map({'yes': 1, 'male': 1, 'no': 0, 'female': 0})
                    .fillna(0)
                    .astype(int)
                )
        
        self._cat_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self._cat_ind_ = [X.columns.get_loc(col) for col in self._cat_cols_]

        X_transformed = self._preprocess(X)

        cat_transformer = self._preprocessor_.named_transformers_['cat']
        n_cat_features = len(cat_transformer.get_feature_names_out())
        cat_indices = [
            i for i, name in
            enumerate(self._preprocessor_.get_feature_names_out()) if
            name.startswith('cat__')
        ]
        self._cat_ind_transformed_ = cat_indices
        smote = SMOTENC(
            categorical_features=self._cat_ind_transformed_,
            random_state=self.random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X_transformed, y)
          
        self._model_.fit(X_resampled, y_resampled)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict the outcome from data X given threshold.
        """
        check_is_fitted(self, "_model_")
        probability = self.predict_proba(X)[:, 1]
        return (probability >= self.threshold).astype(int)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Return probabilities per class
        """
        check_is_fitted(self, "_model_")
        X_transformed = self._preprocessor_.transform(X)
        return self._model_.predict_proba(X_transformed)
        
    def fit_predict(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)

    def _map_y(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Map y column as integers
        """
        if np.issubdtype(y.dtype, np.number):
            return y  # already numeric
        else:
            return y.replace({'Healthy': 0, 'At Risk': 1}).values
        
    def _preprocess(
        self, X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Perform one-hot encoding
        """
        self._preprocessor_ = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), self._cat_ind_)
            ],
            remainder='passthrough'
        )
        
        self._preprocessor_.fit(X)
        
        # Apply SMOTE only during training
        return self._preprocessor_.transform(X)

        