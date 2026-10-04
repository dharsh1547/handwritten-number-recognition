"""
Handwritten Digit Recognition using Machine Learning (scikit-learn)
------------------------------------------------------------------
This script demonstrates handwritten number (digit) recognition using
a Support Vector Machine (SVM) classifier on the standard Digits dataset.

It performs:
1. Dataset loading and inspection
2. Train/Test split and normalization
3. Model training (Support Vector Classifier - RBF kernel)
4. Evaluation (Accuracy, Classification Report, Confusion Matrix)
5. Visualizing predictions (actual vs. predicted digits)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

def main():
    print("=" * 60)
    print("   HANDWRITTEN DIGIT RECOGNITION USING MACHINE LEARNING")
    print("=" * 60)

    # 1. Load the handwritten digits dataset
    print("\n[1] Loading dataset...")
    digits = datasets.load_digits()
    
    X = digits.data          # Pixel features (8x8 flattened into 64 features)
    y = digits.target        # Ground truth labels (0 to 9)
    images = digits.images    # 8x8 2D image arrays

    print(f"    - Total samples : {X.shape[0]}")
    print(f"    - Features/image: {X.shape[1]} (8x8 pixels)")
    print(f"    - Target classes: {np.unique(y)} (digits 0 through 9)")

    # 2. Split dataset into training and test sets (80% train, 20% test)
    print("\n[2] Splitting dataset into training (80%) and testing (20%)...")
    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, images, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    - Training samples: {X_train.shape[0]}")
    print(f"    - Testing samples : {X_test.shape[0]}")

    # 3. Feature Scaling (Standardization helps SVM converge faster and better)
    print("\n[3] Preprocessing: Standardizing feature values...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train the Support Vector Machine (SVM) Classifier
    print("\n[4] Training Support Vector Classifier (SVC with RBF kernel)...")
    model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    model.fit(X_train_scaled, y_train)
    print("    - Training complete!")

    # 5. Evaluate the model on unseen test data
    print("\n[5] Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n>>> TEST ACCURACY: {acc * 100:.2f}% <<<\n")

    print("Detailed Classification Report:")
    print("-" * 55)
    print(classification_report(y_test, y_pred, digits=4))

    # 6. Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.title(f"Confusion Matrix (Accuracy: {acc*100:.1f}%)", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Digit", fontsize=11)
    plt.ylabel("True Digit", fontsize=11)
    plt.tight_layout()
    cm_path = "digit_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"[6] Saved confusion matrix figure to: {cm_path}")
    plt.close()

    # 7. Visualize Sample Test Predictions
    # Pick 12 random test samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_test), size=12, replace=False)
    
    fig, sub_axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle("Sample Test Predictions (Green = Correct, Red = Error)", fontsize=14, fontweight='bold')
    
    for ax, idx in zip(sub_axes.flat, sample_indices):
        img = img_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"True: {true_label} | Pred: {pred_label}", color=color, fontweight='bold', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    pred_path = "sample_predictions.png"
    plt.savefig(pred_path, dpi=150)
    print(f"[7] Saved sample predictions visualization to: {pred_path}")
    plt.close()

    print("\n" + "=" * 60)
    print("SUCCESS: Model trained and evaluated successfully!")
    print("Generated Output Images:")
    print(f"  1. {cm_path} (Confusion Matrix)")
    print(f"  2. {pred_path} (Sample Predictions Visualizer)")
    print("=" * 60)

if __name__ == "__main__":
    main()
