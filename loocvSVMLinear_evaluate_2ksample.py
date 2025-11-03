import numpy as np
import joblib
import os
import pickle
from sklearn.model_selection import LeaveOneOut, train_test_split
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
import time


# Load Data & Ambil 2000 Sample
print("Memuat data dan model...")
X = np.load("hog_features.npy")
y = np.load("labels.npy")
print("Data asli:", X.shape, "| Label:", y.shape)

#Sampling 2000 data secara stratified
X_small, _, y_small, _ = train_test_split(
    X, y, train_size=2000, stratify=y, random_state=42
)
print("Sample data:", X_small.shape, "| Label:", y_small.shape)


# Leave-One-Out
loo = LeaveOneOut()
n_samples = len(y_small)


# Setup Checkpoint
SAVE_FILE = "loocv_partial_2k_linear.pkl"
SAVE_INTERVAL = 80  # Simpan tiap 80 iterasi
partial_results = []

if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "rb") as f:
        partial_results = pickle.load(f)
    print(f"Melanjutkan dari checkpoint: {len(partial_results)} hasil ditemukan.\n")
else:
    print("Tidak ada checkpoint ditemukan, mulai dari awal.\n")

start_index = len(partial_results)


# evaluate every fold
def eval_one(i, train_idx, test_idx):
    # Tetap gunakan kernel linear
    model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))
    model.fit(X_small[train_idx], y_small[train_idx])
    y_pred = model.predict(X_small[test_idx])
    return (y_small[test_idx][0], y_pred[0], i)


#Evaluasi LOOCV 
print("Evaluasi LOOCV (2000 sample - Linear kernel)...")
start = time.time()

splits = list(loo.split(X_small))[start_index:]
progress_bar = tqdm(total=n_samples, initial=start_index, desc="LOOCV Progress")

for batch_start in range(0, len(splits), SAVE_INTERVAL):
    batch = splits[batch_start:batch_start + SAVE_INTERVAL]

    batch_results = Parallel(n_jobs=-1)(
        delayed(eval_one)(start_index + i, train, test)
        for i, (train, test) in enumerate(batch)
    )

    partial_results.extend([(yt, yp) for yt, yp, _ in batch_results])

    with open(SAVE_FILE, "wb") as f:
        pickle.dump(partial_results, f)

    progress_bar.update(len(batch_results))
    print(f"Checkpoint tersimpan ({len(partial_results)}/{n_samples} selesai).")

progress_bar.close()

end = time.time()
print(f"\nWaktu eksekusi total: {end - start:.2f} detik")


# last evaluation
y_true, y_pred = zip(*partial_results)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nSemua data (2000 sample - linear kernel) selesai dievaluasi.")
print(f"Checkpoint tetap disimpan di file: {SAVE_FILE}\n")


# calculate the matrix
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_true, y_pred)

print("\n=== HASIL EVALUASI LOOCV (2000 Sample, Linear Kernel) ===")
print(f"Akurasi   : {acc * 100:.2f}%")
print(f"Precision : {prec * 100:.2f}%")
print(f"Recall    : {rec * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


#Confusion Matrix
labels = sorted(np.unique(y_small))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap='Blues', annot=True, fmt='d',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (LOOCV - 2000 Sample - SVM Linear)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_loocv_2k_linear.png", dpi=300, bbox_inches='tight')
plt.show()
print("\nConfusion matrix disimpan sebagai 'confusion_matrix_loocv_2k_linear.png'")

# all report
print("\nClassification Report")
print(classification_report(y_true, y_pred, zero_division=0))
