import pandas as pd
from sklearn.utils import resample

# load dataset EMNIST Letters
df = pd.read_csv("emnist-letters-train.csv", header=None)

# definisi label dan isi
df.columns = ['label'] + [f'pixel_{i}' for i in range(1, df.shape[1])]
print("Distribusi awal per kelas:")
print(df['label'].value_counts().sort_index())

# Sampling 500 per class
samples_per_class = 500
balanced_samples = []

for label, group in df.groupby('label'):
    sampled = resample(
        group,
        replace=False,  # tanpa duplikasi
        n_samples=samples_per_class,
        random_state=42
    )
    balanced_samples.append(sampled)

# Gabungkan hasil sampling
df_balanced = pd.concat(balanced_samples)
# Mengacak posisi yang sudah di ambil sebelumnya
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# hasil
df_balanced.to_csv("emnist_letters_13k.csv", index=False)
print("\nDistribusi setelah sampling:")
print(df_balanced['label'].value_counts().sort_index())
print("\nTotal sampel:", len(df_balanced))
