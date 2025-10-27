# ============================================
# model.py — Train and Export Full Pipeline
# ============================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def train_and_save_model(train_path='train.csv', output_path='new_model.joblib'):
    """
    Trains the risk-level model and saves it as a joblib pipeline.
    """
    print("🚀 Loading training data...")
    df = pd.read_csv(train_path)

    # ----------- Define feature types -----------
    numerical_features = [
        'gestational_age_weeks', 'birth_weight_kg', 'birth_length_cm',
        'birth_head_circumference_cm', 'age_days', 'weight_kg', 'length_cm',
        'head_circumference_cm', 'temperature_c', 'heart_rate_bpm', 'respiratory_rate_bpm',
        'oxygen_saturation', 'feeding_frequency_per_day', 'urine_output_count',
        'stool_count', 'jaundice_level_mg_dl'
    ]

    categorical_features = ['feeding_type']
    binary_features = ['gender', 'immunizations_done', 'reflexes_normal']

    # ----------- Features & target -----------
    X = df.drop(columns=['risk_level'])
    y = df['risk_level'].replace({'Healthy': 0, 'At Risk': 1})

    # ----------- Preprocessing -----------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('bin', OneHotEncoder(drop='if_binary', handle_unknown='ignore'), binary_features)
        ]
    )

    # ----------- Model -----------
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )

    # ----------- Pipeline -----------
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # ----------- Train -----------
    print("🧠 Training model...")
    pipeline.fit(X, y)

    # ----------- Save -----------
    joblib.dump(pipeline, output_path)
    print(f"✅ Model pipeline saved as '{output_path}'")


def predict_from_file(model_path='new_model.joblib', test_path='test.csv', output_path='model_output.csv'):
    """
    Loads a saved model and predicts on a test CSV.
    """
    print("📦 Loading model...")
    pipeline = joblib.load(model_path)

    print("📄 Loading test data...")
    test_df = pd.read_csv(test_path)

    print("🔮 Predicting...")
    preds = pipeline.predict(test_df)
    label_map = {0: 'Healthy', 1: 'At Risk'}
    test_df['predicted_risk_level'] = [label_map[p] for p in preds]

    test_df.to_csv(output_path, index=False)
    print(f"🎉 Predictions saved to '{output_path}'")


# ----------- Script Entry Point -----------
if __name__ == '__main__':
    # Train model and save
    train_and_save_model()

    # Optional: immediately test predictions after training
    # predict_from_file()
