from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.over_sampling import SMOTENC
from lime.lime_tabular import LimeTabularExplainer


class WhiteBox(BaseEstimator, ClassifierMixin):
    """
    WhiteBox classifier for newborn health risk prediction.
    
    This model uses Random Forest with SMOTENC for handling imbalanced data.
    Best hyperparameters from tuning:
    - k_neighbors: 3
    - n_estimators: 100
    - max_depth: 3
    - min_samples_split: 10
    - min_samples_leaf: 4
    """
    
    def __init__(self, 
                 threshold: float = 0.5):
        """
        Initialize class, preprocessor, and random forest model
        
        Parameters
        ----------
        threshold (float): Decision threshold for probability (default=0.5)
        """
        self.threshold = threshold
        
        # Feature definitions
        self.numerical_features = [
            'weight_kg', 'length_cm', 'head_circumference_cm', 
            'temperature_c', 'heart_rate_bpm', 'respiratory_rate_bpm',
            'oxygen_saturation', 'jaundice_level_mg_dl',
            'age_days', 'feeding_frequency_per_day',
            'urine_output_count', 'stool_count'
        ]
        
        self.categorical_features = [
            'feeding_type', 'gender', 'immunizations_done', 'reflexes_normal'
        ]
        
        # Initialize preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_features),
                ('cat', OneHotEncoder(drop="if_binary", handle_unknown="ignore"), self.categorical_features),
            ]
        )
        
        # Initialize Random Forest with best hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        
        # Pipeline will be created during fit
        self.pipeline = None
        self.is_fitted_ = False
    
    def fit(self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        """
        
        # Fit preprocessor first to get categorical indices
        self.preprocessor.fit(X)
        
        # Get feature count after preprocessing
        feature_names_out = self.preprocessor.get_feature_names_out()
        numerical_count = len(self.numerical_features)
        cat_transformer = self.preprocessor.named_transformers_['cat']
        categorical_count = len(cat_transformer.get_feature_names_out())
        
        # Categorical indices for SMOTENC
        new_cat_indices = list(range(
            numerical_count, 
            numerical_count + categorical_count
        ))
        
        # Create pipeline with SMOTENC
        self.pipeline = imb_pipeline([
            ("preprocessor", self.preprocessor),
            ("smote", SMOTENC(
                categorical_features=new_cat_indices,
                k_neighbors=3,
                random_state=42
            )),
            ("clf", self.model)
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted_ = True

        # Save a copy of X to be used for LIME
        self._X = X.copy()
        
        return self
    
    def predict(self, 
        X: Union[np.ndarray, pd.DataFrame]
   ):
        """Predict the outcome from data X given threshold."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict")
        
        # Get probabilities and threshold
        proba = self.predict_proba(X)[:, 1]
        predictions = (proba >= self.threshold).astype(int)
        
        return predictions
    
    def fit_predict(self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ):
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)
    
    def predict_proba(self, 
        X: Union[np.ndarray, pd.DataFrame]
   ):
        "Return probabilities per class"
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict_proba")
        
        return self.pipeline.predict_proba(X)
        
    def _get_feature_importance(self):
        """
        Get feature importances from the trained Random Forest.
        
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get feature names after preprocessing
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Get feature importances from the classifier
        clf = self.pipeline.named_steps['clf']
        importances = clf.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return importance_df

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.over_sampling import SMOTENC
from lime.lime_tabular import LimeTabularExplainer


class WhiteBox(BaseEstimator, ClassifierMixin):
    """
    WhiteBox classifier for newborn health risk prediction.
    
    This model uses Random Forest with SMOTENC for handling imbalanced data.
    Best hyperparameters from tuning:
    - k_neighbors: 3
    - n_estimators: 100
    - max_depth: 3
    - min_samples_split: 10
    - min_samples_leaf: 4
    """
    
    def __init__(self, 
                 threshold: float = 0.5):
        """
        Initialize class, preprocessor, and random forest model
        
        Parameters
        ----------
        threshold (float): Decision threshold for probability (default=0.5)
        """
        self.threshold = threshold
        
        # Feature definitions
        self.numerical_features = [
            'weight_kg', 'length_cm', 'head_circumference_cm', 
            'temperature_c', 'heart_rate_bpm', 'respiratory_rate_bpm',
            'oxygen_saturation', 'jaundice_level_mg_dl',
            'age_days', 'feeding_frequency_per_day',
            'urine_output_count', 'stool_count'
        ]
        
        self.categorical_features = [
            'feeding_type', 'gender', 'immunizations_done', 'reflexes_normal'
        ]

        self.all_features = self.numerical_features + self.categorical_features
        self.old_cat_indices = [self.all_features.index(feat) for feat in self.categorical_features]
        
        # Initialize preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numerical_features),
                ('cat', OneHotEncoder(drop="if_binary"), self.categorical_features),
            ]
        )
        
        # Initialize Random Forest with best hyperparameters
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        
        # Pipeline will be created during fit
        self.pipeline = None
        self.is_fitted_ = False
    
    def fit(self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ):
        """
        Fit the model using a training X and y.

        Parameters:
        -----------
        X (np.ndarray | pd.DataFrame): The training data
        y (np.ndarray | pd.Sereis): The corresponding targets
        """
        # Save a copy of X to be used for LIME
        self._X = X.copy()
        
        # Create categorical encoders for LIME (before any transformation)
        self._cat_encoders = {}
        for cat_feat in self.categorical_features:
            unique_values = self._X[cat_feat].unique()
            self._cat_encoders[cat_feat] = {val: idx for idx, val in enumerate(unique_values)}
        
        # Fit preprocessor first to get categorical indices
        self.preprocessor.fit(X)
        
        # Get feature count after preprocessing
        feature_names_out = self.preprocessor.get_feature_names_out()
        numerical_count = len(self.numerical_features)
        cat_transformer = self.preprocessor.named_transformers_['cat']
        categorical_count = len(cat_transformer.get_feature_names_out())
        
        # Categorical indices for SMOTENC
        new_cat_indices = list(range(
            numerical_count, 
            numerical_count + categorical_count
        ))
        
        # Create pipeline with SMOTENC
        self.pipeline = imb_pipeline([
            ("preprocessor", self.preprocessor),
            ("smote", SMOTENC(
                categorical_features=new_cat_indices,
                k_neighbors=3,
                random_state=42
            )),
            ("clf", self.model)
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted_ = True

        # Save a copy of X to be used for LIME
        self._X = X.copy()
        
        return self
    
    def predict(self, 
        X: Union[np.ndarray, pd.DataFrame]
   ):
        """Predict the outcome from data X given threshold."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict")
        
        # Get probabilities and threshold
        proba = self.predict_proba(X)[:, 1]
        predictions = (proba >= self.threshold).astype(int)
        
        return predictions
    
    def fit_predict(self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ):
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)
    
    def predict_proba(self, 
        X: Union[np.ndarray, pd.DataFrame]
   ):
        "Return probabilities per class"
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict_proba")
        
        return self.pipeline.predict_proba(X)
        
    def _get_feature_importance(self):
        """
        Get feature importances from the trained Random Forest.
        
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get feature names after preprocessing
        feature_names = self.preprocessor.get_feature_names_out()
        
        # Get feature importances from the classifier
        clf = self.pipeline.named_steps['clf']
        importances = clf.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        return importance_df

    def _explain_instance(self, random_person: Union[np.ndarray, pd.Series, pd.DataFrame]):
        """
        Show LIME Explainer for a certain instance.
        
        Parameters
        ----------
        random_person : np.ndarray | pd.Series | pd.DataFrame
            Specific instance or row.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting LIME")
    
        # Convert instance to DataFrame if needed
        if isinstance(random_person, np.ndarray):
            random_person = pd.DataFrame(random_person, columns=self._X.columns)
        elif isinstance(random_person, pd.Series):
            random_person = random_person.to_frame().T
        
        # Ensure instance is 2D (1 row)
        if random_person.shape[0] != 1:
            raise ValueError("Instance must be a single row (shape: (1, n_features))")
        
        # Store original categorical mappings during fit (add this to fit method)
        if not hasattr(self, '_cat_encoders'):
            self._cat_encoders = {}
            for cat_feat in self.categorical_features:
                unique_values = self._X[cat_feat].unique()
                self._cat_encoders[cat_feat] = {val: idx for idx, val in enumerate(unique_values)}
        
        # Create a copy of training data for LIME with encoded categoricals
        X_lime = self._X.copy()
        for cat_feat in self.categorical_features:
            X_lime[cat_feat] = X_lime[cat_feat].map(self._cat_encoders[cat_feat])
        
        # Encode the instance
        instance_lime = random_person.copy()
        for cat_feat in self.categorical_features:
            instance_lime[cat_feat] = instance_lime[cat_feat].map(self._cat_encoders[cat_feat])
        
        # Create custom predict function that handles DataFrames
        def predict_fn_wrapper(X_array):
            """Wrapper to convert numpy array back to DataFrame for pipeline"""
            X_df = pd.DataFrame(X_array, columns=self._X.columns)
            
            # Decode categorical features back to original values
            for cat_feat in self.categorical_features:
                reverse_map = {v: k for k, v in self._cat_encoders[cat_feat].items()}
                X_df[cat_feat] = X_df[cat_feat].map(reverse_map)
            
            return self.pipeline.predict_proba(X_df)
        
        # Create the LIME explainer with encoded training data
        lime_explainer = LimeTabularExplainer(
            training_data=X_lime.values,
            feature_names=self._X.columns.tolist(),
            categorical_features=self.old_cat_indices,
            class_names=["Healthy", "At Risk"],
            mode="classification",
            random_state=42
        )
        
        # Explain the instance
        explained_inst = lime_explainer.explain_instance(
            data_row=instance_lime.values[0],
            predict_fn=predict_fn_wrapper
        )
        
        return explained_inst