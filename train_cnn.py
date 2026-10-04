"""
Train a Deep Learning Convolutional Neural Network (CNN) on the 28x28 MNIST Dataset.
Uses 60,000 real handwritten digits for state-of-the-art freehand recognition accuracy.
"""

import os
import time
import keras
from keras import layers

def train_and_save_cnn():
    os.makedirs("model", exist_ok=True)
    print("=" * 60)
    print("   TRAINING DEEP LEARNING CNN ON MNIST (60,000 SAMPLES)")
    print("=" * 60)

    # 1. Load MNIST (28x28 pixels, 60k train, 10k test)
    print("\n[1] Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    print(f"    - Training samples: {x_train.shape[0]} images (28x28)")
    print(f"    - Testing samples : {x_test.shape[0]} images (28x28)")

    # 2. Build Convolutional Neural Network
    print("\n[2] Building CNN Architecture...")
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()

    # 3. Train Model
    print("\n[3] Training model across epochs...")
    t0 = time.time()
    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=3,
        validation_split=0.1,
        verbose=1
    )
    t1 = time.time()
    print(f"    - Training finished in {t1 - t0:.2f} seconds.")

    # 4. Evaluate
    print("\n[4] Evaluating on unseen test set...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n>>> TEST ACCURACY: {acc * 100:.2f}% <<<\n")

    # 5. Save model
    save_path = os.path.join("model", "mnist_cnn.keras")
    model.save(save_path)
    print(f"[5] Saved trained CNN model to: {save_path}")

if __name__ == "__main__":
    train_and_save_cnn()
