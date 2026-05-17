import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

from aggregation import aggregation_and_feature_extraction
from probe import HallucinationProbe
from splitting import split_data
from model import MAX_LENGTH, get_model_and_tokenizer


DATA_FILE = "./data/dataset.csv"
BATCH_SIZE = 4
FIG_DIR = "figures"


def save_confusion_matrix(path, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["truthful (0)", "hallucinated (1)"])
    ax.set_yticklabels(["truthful (0)", "hallucinated (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pca_val(path, X_val, y_val):
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_val)
    fig, ax = plt.subplots(figsize=(6, 5))
    mask0 = y_val == 0
    mask1 = y_val == 1
    ax.scatter(X_2d[mask0, 0], X_2d[mask0, 1], s=24, alpha=0.8, label="truthful (0)")
    ax.scatter(X_2d[mask1, 0], X_2d[mask1, 1], s=24, alpha=0.8, label="hallucinated (1)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Validation PCA (2D)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    all_texts = [f"{row['prompt']}{row['response']}" for _, row in df.iterrows()]
    y = np.array([int(float(v)) for v in df["label"]], dtype=int)

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    all_features = []
    for start in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Extracting features", unit="batch"):
        batch_texts = all_texts[start : start + BATCH_SIZE]
        encoding = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        hidden = torch.stack(outputs.hidden_states, dim=1).float()
        mask = attention_mask.cpu()

        for i in range(hidden.size(0)):
            feat = aggregation_and_feature_extraction(
                hidden[i],
                mask[i],
                use_geometric=False,
            )
            all_features.append(feat.cpu())

    X = np.vstack([f.numpy() for f in all_features]).astype(np.float32)

    splits = split_data(y, df)
    idx_train, idx_val, idx_test = splits[0]

    probe = HallucinationProbe()
    probe.fit(X[idx_train], y[idx_train])

    y_val_pred = probe.predict(X[idx_val])
    y_test_pred = probe.predict(X[idx_test])

    val_acc = accuracy_score(y[idx_val], y_val_pred)
    test_acc = accuracy_score(y[idx_test], y_test_pred)

    val_cm_path = os.path.join(FIG_DIR, "confusion_matrix_val.png")
    test_cm_path = os.path.join(FIG_DIR, "confusion_matrix_test.png")
    pca_val_path = os.path.join(FIG_DIR, "pca_val.png")

    save_confusion_matrix(val_cm_path, y[idx_val], y_val_pred, "Validation Confusion Matrix")
    save_confusion_matrix(test_cm_path, y[idx_test], y_test_pred, "Internal Test Confusion Matrix")
    save_pca_val(pca_val_path, X[idx_val], y[idx_val])

    print(f"feature_dim: {X.shape[1]}")
    print(f"validation_accuracy: {val_acc:.6f}")
    print(f"internal_test_accuracy: {test_acc:.6f}")
    print(f"generated_figure: {val_cm_path}")
    print(f"generated_figure: {test_cm_path}")
    print(f"generated_figure: {pca_val_path}")


if __name__ == "__main__":
    main()
