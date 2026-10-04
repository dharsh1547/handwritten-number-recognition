"""
Train and serialize the SVM model and scaler for the web application.
Enables probability estimation (probability=True) to produce confidence distributions.
"""

import os
import joblib
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_and_save():
    os.makedirs("model", exist_ok=True)
    print("Loading digits dataset...")
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    print(f"Dataset shape: {X.shape}, classes: {np.unique(y)}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Standardize
    print("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM with probability=True for confidence scores
    print("Training Support Vector Classifier (probability=True)...")
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Serialize
    model_path = os.path.join("model", "model.joblib")
    scaler_path = os.path.join("model", "scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    train_and_save()
