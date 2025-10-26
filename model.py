import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as imb_pipeline
from imblearn.over_sampling import SMOTENC


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
        
        # Fit preprocessor first to get categorical indices
        self.preprocessor.fit(X)
        
        # Get feature count after preprocessing
        feature_names_out = self.preprocessor.get_feature_names_out()
        numerical_count = len(self.numerical_features)
        cat_transformer = self.preprocessor.named_transformers_['cat']
        categorical_count = len(cat_transformer.get_feature_names_out())
        
        # Categorical indices for SMOTENC
        cat_indices = list(range(
            numerical_count, 
            numerical_count + categorical_count
        ))
        
        # Create pipeline with SMOTENC
        self.pipeline = imb_pipeline([
            ("preprocessor", self.preprocessor),
            ("smote", SMOTENC(
                categorical_features=cat_indices,
                k_neighbors=3,
                random_state=42
            )),
            ("clf", self.model)
        ])
        
        # Fit the pipeline
        self.pipeline.fit(X, y)
        self.is_fitted_ = True
        
        return self
    
    def predict(self, X):
        """Predict the outcome from data X given threshold."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict")
        
        # Get probabilities and threshold
        proba = self.predict_proba(X)[:, 1]
        predictions = (proba >= self.threshold).astype(int)
        
        return predictions
    
    def fit_predict(self, X, y):
        """
        Fit the model using a training X and y, and predict the outcome using
        X. Note that this would give the predicted labels during training.
        """
        return self.fit(X, y).predict(X)
    
    def predict_proba(self, X):
        "Return probabilities per class"
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before calling predict_proba")
        
        return self.pipeline.predict_proba(X)
        
    def get_feature_importance(self):
        """
        Get feature importances from the trained Random Forest.
        
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with features and their importance scores
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