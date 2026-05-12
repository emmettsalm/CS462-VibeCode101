"""
Thai Numeral Recognition - Training Script (Standalone)
Classes: ๕๖, ๕๗, ๕๘, ๕๙, ๖๐
Dataset expected at: ./dataset_thai_v4/  (one subfolder per class)

Usage:
    python train_model.py
    python train_model.py --data ./dataset_thai_v4 --epochs 30

Team Members:
    อนาวินธุ์ อักษรทิพย์      1660701440
    ดฤพล กรณ์ถาวรวงศ์        1660703974
    เอ็มเม็ต มีชัย แซลมอน     1660704444
    ธนวัฒน์ วิเศษชัยวรรณ      1660703990
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def build_model(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    m = models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    m.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return m


def make_generators(data_dir: str, batch_size: int = 32):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.1,
        validation_split=0.2
    )
    val_gen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train = train_gen.flow_from_directory(
        data_dir, target_size=(28, 28), color_mode='grayscale',
        batch_size=batch_size, class_mode='sparse', subset='training'
    )
    val = val_gen.flow_from_directory(
        data_dir, target_size=(28, 28), color_mode='grayscale',
        batch_size=batch_size, class_mode='sparse', subset='validation'
    )
    return train, val


def evaluate_and_plot(model, val_gen, classes, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Collect all predictions
    y_true, y_pred_prob = [], []
    val_gen.reset()
    for _ in range(len(val_gen)):
        x_batch, y_batch = next(val_gen)
        probs = model.predict(x_batch, verbose=0)
        y_pred_prob.extend(probs)
        y_true.extend(y_batch.astype(int))

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"[Saved] confusion_matrix.png")

    # ── Classification Report ────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("\n── Classification Report ──────────────────────────")
    print(report)
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

    # ── ROC / AUC (one-vs-rest) ───────────────────────────────────────────────
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (One-vs-Rest)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f"[Saved] roc_curve.png")

    # ── Training History ─────────────────────────────────────────────────────
    return y_true, y_pred


def plot_history(history, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_history.png'), dpi=150)
    plt.close()
    print(f"[Saved] training_history.png")


def main():
    parser = argparse.ArgumentParser(description='Train Thai Numeral CNN')
    parser.add_argument('--data', default='dataset_thai_v4',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--out', default='models', help='Output directory')
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] Dataset directory not found: {args.data}")
        print("Please provide the dataset path with --data <path>")
        return

    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Dataset: {args.data}")
    print(f"Epochs:  {args.epochs}")

    train_gen, val_gen = make_generators(args.data, args.batch)
    classes = list(train_gen.class_indices.keys())
    print(f"Classes: {classes}")
    print(f"Train samples: {train_gen.samples} | Val samples: {val_gen.samples}")

    model = build_model(len(classes))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        )
    ]

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    os.makedirs(args.out, exist_ok=True)
    metrics_dir = os.path.join(args.out, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    plot_history(history, metrics_dir)
    evaluate_and_plot(model, val_gen, classes, metrics_dir)

    model_path = os.path.join(args.out, 'model.h5')
    model.save(model_path)
    print(f"\n[Saved] Model → {model_path}")

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"\n── Final Validation ─────────────────────────────────")
    print(f"   Accuracy : {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   Loss     : {val_loss:.4f}")
    if val_acc >= 0.80:
        print("   Status   : PASS (>= 80%)")
    else:
        print("   Status   : FAIL (< 80%) — consider more data or tuning")


if __name__ == '__main__':
    main()
