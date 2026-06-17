"""
core/dl_antispoof_cnn.py
------------------------
Deep-learning anti-spoof module using a lightweight TensorFlow/Keras CNN.

The model is a binary classifier:
    0 = Spoof (printed photo, phone screen, video replay)
    1 = Real  (live person)

Two modes
---------
1. Inference only (default):
   Load a pre-trained model from ``models/antispoof_cnn.h5`` and call
   ``predict(face_roi_bgr)`` to get (is_live, confidence).

2. Train your own model:
   Call ``train_and_save(real_dir, spoof_dir)`` pointing to folders of
   face-crop images.  The trained model is saved to ``models/antispoof_cnn.h5``.

Usage in recognition loop::

    from core.dl_antispoof_cnn import DLAntiSpoof
    checker = DLAntiSpoof()          # loads model if available
    is_live, conf = checker.predict(face_roi)
    if not is_live:
        # show SPOOF DETECTED
"""

import os
import numpy as np

MODEL_PATH  = os.path.join("models", "antispoof_cnn.h5")
INPUT_SIZE  = (64, 64)          # CNN input resolution
SPOOF_THRESHOLD = 0.50          # probability above which face is "real"


# ---------------------------------------------------------------------------
class DLAntiSpoof:
    """Lightweight CNN-based anti-spoof classifier."""

    def __init__(self, model_path: str = MODEL_PATH) -> None:
        self._model      = None
        self._model_path = model_path
        self._available  = self._try_load(model_path)

    @property
    def available(self) -> bool:
        """True if a trained model was successfully loaded."""
        return self._available

    # ------------------------------------------------------------------
    def _try_load(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"[DL-AntiSpoof] No model at '{path}'. "
                  "Run train_and_save() to train one.")
            return False
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(path)
            print(f"[DL-AntiSpoof] Model loaded from '{path}'.")
            return True
        except Exception as exc:
            print(f"[DL-AntiSpoof] Could not load model: {exc}")
            return False

    # ------------------------------------------------------------------
    def predict(self, face_roi_bgr: np.ndarray) -> tuple[bool, float]:
        """
        Classify a face ROI as real or spoof.

        Returns
        -------
        (is_live, confidence) : tuple[bool, float]
            ``is_live`` is True when the face is classified as a real person.
            ``confidence`` is the model's probability for "real" [0, 1].
        """
        if not self._available or self._model is None:
            return True, 1.0    # Fallback: pass everything if no model

        import cv2
        import tensorflow as tf

        if face_roi_bgr is None or face_roi_bgr.size == 0:
            return True, 1.0

        # Preprocess
        resized    = cv2.resize(face_roi_bgr, INPUT_SIZE)
        rgb        = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalised = rgb.astype("float32") / 255.0
        batch      = np.expand_dims(normalised, axis=0)   # (1, 64, 64, 3)

        prob_real  = float(self._model.predict(batch, verbose=0)[0][0])
        is_live    = prob_real >= SPOOF_THRESHOLD

        return is_live, prob_real

    # ------------------------------------------------------------------
    @staticmethod
    def build_model(input_shape: tuple = (*INPUT_SIZE, 3)):
        """
        Define a lightweight CNN architecture for real/spoof classification.
        ~300k parameters — fast enough for real-time use.
        """
        import tensorflow as tf
        from tensorflow.keras import layers, models

        model = models.Sequential([
            layers.Input(shape=input_shape),

            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.4),

            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(1, activation="sigmoid"),   # 1 = real, 0 = spoof
        ], name="antispoof_cnn")

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    @classmethod
    def train_and_save(
        cls,
        real_dir:   str,
        spoof_dir:  str,
        model_path: str  = MODEL_PATH,
        epochs:     int  = 20,
        batch_size: int  = 32,
    ) -> None:
        """
        Train a new anti-spoof CNN and save it.

        Parameters
        ----------
        real_dir   : folder of face-crop images of real people
        spoof_dir  : folder of spoof images (photos, screens)
        model_path : where to save the trained .h5 model
        epochs     : training epochs
        batch_size : mini-batch size
        """
        import cv2
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # ----- Load images -----
        def _load(folder: str, label: int):
            X, y = [], []
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img  = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), INPUT_SIZE)
                X.append(img.astype("float32") / 255.0)
                y.append(label)
            return X, y

        real_X,  real_y  = _load(real_dir,  label=1)
        spoof_X, spoof_y = _load(spoof_dir, label=0)

        X = np.array(real_X  + spoof_X)
        y = np.array(real_y  + spoof_y)

        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        # Train / val split (80/20)
        split  = int(0.8 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        print(f"[DL-AntiSpoof] Training on {len(X_tr)} samples, "
              f"validating on {len(X_val)}.")

        aug = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
        )

        model = cls.build_model()
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, save_best_only=True),
        ]

        model.fit(
            aug.flow(X_tr, y_tr, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
        )

        model.save(model_path)
        print(f"[DL-AntiSpoof] Model saved → '{model_path}'")
