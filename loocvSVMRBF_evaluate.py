# =========================================
# File: evaluate_loocv_parallel_checkpoint.py
# =========================================
import numpy as np
import joblib
import os
import pickle
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import string
import time

# =========================================
# 1. Load Data & Model
# =========================================
print("Memuat data dan model...")
X = np.load("hog_features.npy")
y = np.load("labels.npy")
print("Data:", X.shape, "| Label:", y.shape)

# =========================================
# 2. Leave-One-Out
# =========================================
loo = LeaveOneOut()
n_samples = len(y)

# =========================================
# 3. Setup Checkpoint
# =========================================
SAVE_FILE = "loocv_partial_RBF.pkl"
SAVE_INTERVAL = 4  
partial_results = []

# Cek apakah ada hasil sebelumnya
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "rb") as f:
        partial_results = pickle.load(f)
    print(f"Melanjutkan dari checkpoint: {len(partial_results)} hasil ditemukan.\n")
else:
    print("Tidak ada checkpoint ditemukan, mulai dari awal.\n")

start_index = len(partial_results)

# =========================================
# 4. Fungsi Evaluasi Tiap Fold
# =========================================
def eval_one(i, train_idx, test_idx):
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma=0.1))
    model.fit(X[train_idx], y[train_idx])
    y_pred = model.predict(X[test_idx])
    return (y[test_idx][0], y_pred[0], i)

# =========================================
# 5. Evaluasi LOOCV (Parallel)
# =========================================
print("Evaluasi LOOCV (Parallel)...")
start = time.time()

remaining_indices = list(range(start_index, n_samples))
splits = list(loo.split(X))[start_index:]

progress_bar = tqdm(total=n_samples, initial=start_index, desc="LOOCV Progress")

for batch_start in range(0, len(splits), SAVE_INTERVAL):
    batch = splits[batch_start:batch_start + SAVE_INTERVAL]

    batch_results = Parallel(n_jobs=-1)(
        delayed(eval_one)(start_index + i, train, test)
        for i, (train, test) in enumerate(batch)
    )

    # Simpan hasil batch ke partial_results
    partial_results.extend([(yt, yp) for yt, yp, _ in batch_results])

    # Simpan checkpoint
    with open(SAVE_FILE, "wb") as f:
        pickle.dump(partial_results, f)

    progress_bar.update(len(batch_results))
    print(f"Checkpoint tersimpan ({len(partial_results)}/{n_samples} selesai).")

progress_bar.close()

end = time.time()
print(f"\nWaktu eksekusi total: {end - start:.2f} detik")

# =========================================
# 6. Evaluasi Akhir
# =========================================
y_true, y_pred = zip(*partial_results)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n✅ Semua data selesai dievaluasi.")
print(f"Checkpoint tetap disimpan di file: {SAVE_FILE}\n")

# =========================================
# 7. Hitung Metrik
# =========================================
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("\n=== HASIL EVALUASI LOOCV (Parallel + Checkpoint) ===")
print(f"Akurasi   : {acc * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

# =========================================
# 8. Confusion Matrix
# =========================================
labels = sorted(np.unique(y))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap='Blues', annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (LOOCV - SVM + HOG)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_loocv.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nConfusion matrix disimpan sebagai 'confusion_matrix_loocv.png'")

# =========================================
# 9. Laporan Lengkap
# =========================================
print("\nClassification Report")
print(classification_report(y_true, y_pred, zero_division=0))
