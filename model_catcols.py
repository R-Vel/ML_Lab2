from typing import Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier

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
        smote_k_neighbors: int = 3,
    ):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.threshold = threshold
        self.smote_k_neighbors = smote_k_neighbors

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
        y (np.ndarray | pd.Series): The corresponding targets
        """
        # Identify categorical columns in original data
        self._cat_cols_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self._cat_ind_ = [X.columns.get_loc(col) for col in self._cat_cols_]
        print("features before", X.columns)
        print("cat before", len(self._cat_ind_))
        # Get number of numerical features
        self._num_features_ = len(X.select_dtypes(exclude=['object', 'category']).columns)

        # Preprocess (one-hot encode)
        X_transformed = self._preprocess(X)
        
        # Calculate categorical indices AFTER one-hot encoding
        if len(self._cat_ind_) > 0:
            # Get number of one-hot encoded features
            cat_transformer = self._preprocessor_.named_transformers_['cat']
            n_cat_features = len(cat_transformer.get_feature_names_out())
            cat_indices = [
                i for i, name in
                enumerate(self._preprocessor_.get_feature_names_out()) if
                name.startswith('cat__')
            ]
            self._cat_ind_transformed_ = cat_indices
        print("features after", self._preprocessor_.get_feature_names_out())
        print("cat after", n_cat_features)
        
        # Handle class imbalance with SMOTE
        X_resampled, y_resampled = self._handle_imbalance(X_transformed, y)
        
        # Initialize and fit the model
        self._model_ = RandomForestClassifier(
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
        )
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
        
    def _preprocess(
        self, X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Perform one-hot encoding
        """
        if len(self._cat_ind_) > 0:
            self._preprocessor_ = ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 self._cat_ind_),
                ('num', 'passthrough', [i for i in range(X.shape[1]) if i not 
                                        in self._cat_ind_]),
            ])
        else:
            self._preprocessor_ = ColumnTransformer(
                transformers=[],
                remainder='passthrough'
            )
        
        self._preprocessor_.fit(X)
        return self._preprocessor_.transform(X)

    def _handle_imbalance(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> tuple:
        """
        Perform SMOTENC on transformed data
        """
        if len(self._cat_ind_transformed_) > 0:
            smote = SMOTENC(
                categorical_features=self._cat_ind_transformed_,
                random_state=self.random_state,
                k_neighbors=self.smote_k_neighbors
            )
            return smote.fit_resample(X, y)
        else:
            return X, y