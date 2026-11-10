"""
06_train_autoencoder.py
=========================
WHAT THIS SCRIPT DOES:
    Builds and trains an Autoencoder neural network on the scaled, benign
    training data, then saves the trained model and a plot of its training
    progress.

WHAT IS AN AUTOENCODER, IN PLAIN TERMS:
    It's a neural network trained to do something that sounds pointless at
    first: take an input, squeeze it down into a smaller "compressed"
    representation (the bottleneck), then try to rebuild the original
    input from that compressed version. Because it's only ever trained on
    BENIGN traffic, it becomes very good at compressing and rebuilding
    normal patterns specifically. When we later show it attack traffic
    (which it's never seen), it does a noticeably worse job of rebuilding
    it — the "reconstruction error" is higher. That error is our anomaly
    score: higher error = more likely to be an attack.

WHY THIS ARCHITECTURE (64 -> 32 -> 16 -> 32 -> 64):
    This is a small, symmetric "hourglass" shape: it narrows down to a
    16-dimensional bottleneck, then widens back out to the original
    number of features. This is a standard, simple starting architecture
    for tabular data like network flow features — nothing exotic, which
    makes it easy for you to explain and defend in a viva or interview.

HOW TO RUN THIS:
    python src/06_train_autoencoder.py

    (Run 04_split_and_scale.py first — this script needs its output.)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    AUTOENCODER_EPOCHS, AUTOENCODER_BATCH_SIZE, AUTOENCODER_EARLY_STOP_PATIENCE,
)


def build_autoencoder(input_dim):
    """
    Builds the autoencoder architecture described above.
    input_dim = number of features in the data (e.g. 78 for CICIDS2017).
    """
    inputs = keras.Input(shape=(input_dim,))

    # Encoder: compress the input down to 16 numbers
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    bottleneck = layers.Dense(16, activation="relu")(x)

    # Decoder: expand back out to the original number of features
    x = layers.Dense(32, activation="relu")(bottleneck)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(inputs, outputs, name="autoencoder")

    # "mse" (mean squared error) measures how different the rebuilt output
    # is from the original input — this is exactly the "reconstruction
    # error" concept described above, used directly as the training signal
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder


if __name__ == "__main__":
    keras.utils.set_random_seed(42)
    X_train_path = os.path.join(PROCESSED_DIR, "X_train_scaled.npy")
    if not os.path.exists(X_train_path):
        raise FileNotFoundError(
            "X_train_scaled.npy not found. Run 04_split_and_scale.py first."
        )
    X_train = np.load(X_train_path)
    print(f"Loaded training data: {X_train.shape}")

    autoencoder = build_autoencoder(input_dim=X_train.shape[1])
    autoencoder.summary()

    # Stop training early if the model stops improving on held-out
    # validation data, to avoid wasting time and to avoid overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=AUTOENCODER_EARLY_STOP_PATIENCE,
        restore_best_weights=True,  # roll back to the best epoch, not just the last one
    )

    print("\nTraining autoencoder (this may take a few minutes)...")
    history = autoencoder.fit(
        X_train, X_train,  # input and target are the SAME data — this is what makes it an autoencoder
        epochs=AUTOENCODER_EPOCHS,
        batch_size=AUTOENCODER_BATCH_SIZE,
        validation_split=0.1,  # hold out 10% of training data to monitor overfitting
        callbacks=[early_stop],
        verbose=1,
    )

    # Save the trained model so later scripts can load it directly
    model_path = os.path.join(MODELS_DIR, "autoencoder.keras")
    autoencoder.save(model_path)
    print(f"\nSaved trained autoencoder to: {model_path}")

    # Plot and save the training history — a useful figure for your
    # Development section, showing the model actually converged rather
    # than training indefinitely or diverging
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE reconstruction loss")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "autoencoder_training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training history plot to: {plot_path}")
