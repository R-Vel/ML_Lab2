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
    
    def _explain_instance(self, random_person: pd.DataFrame):
        """
        Show LIME Explainer for a certain instance.
        
        Args:
            random_person (pd.DataFrame): A single row DataFrame containing the instance to explain.
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before explaining instances")
            
        # Get preprocessor and classifier from pipeline
        preprocessor = self.pipeline.named_steps['preprocessor']
        clf = self.pipeline.named_steps['clf']
        
        # Create feature names list combining numerical and one-hot encoded categorical features
        categorical_feature_names = []
        for i, feature in enumerate(self.categorical_features):
            encoder = preprocessor.named_transformers_['cat']
            categories = encoder.categories_[i]
            categorical_feature_names.extend([f"{feature}_{val}" for val in categories])
            
        feature_names = self.numerical_features + categorical_feature_names
        
        # Initialize LIME explainer with the training data
        transformed_training_data = preprocessor.transform(self._X)
        # Convert to dense array if sparse
        if hasattr(transformed_training_data, 'toarray'):
            transformed_training_data = transformed_training_data.toarray()
            
        explainer = LimeTabularExplainer(
            training_data=transformed_training_data,
            feature_names=feature_names,
            class_names=['Normal', 'At Risk'],
            mode='classification',
            # Specify which features are categorical after one-hot encoding
            categorical_features=list(range(
                len(self.numerical_features),
                len(feature_names)
            ))
        )
        
        # One-hot encode instance
        transformed_instance = preprocessor.transform(random_person)
        if hasattr(transformed_instance, 'toarray'):
            transformed_instance = transformed_instance.toarray()
        transformed_instance = transformed_instance[0]
        
        # Generate the explanation
        explanation = explainer.explain_instance(
            data_row=transformed_instance,
            predict_fn=clf.predict_proba,
            num_features=len(feature_names),
            top_labels=1
        )
        
        return explanation
        
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

    