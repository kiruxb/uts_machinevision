import numpy as np
import pandas as pd
from skimage.feature import hog
from tqdm import tqdm


#Load dataset from sorter.py
DATA_PATH = "emnist_letters_13000.csv"  
print(f"Memuat dataset dari: {DATA_PATH}")

data = pd.read_csv(DATA_PATH)
y = data.iloc[:, 0].values.astype(int) - 1  
X = data.iloc[:, 1:].values.astype(np.float32)  
print("Dataset shape:", X.shape)
X /= 255.0 
X_reshaped = X.reshape(-1, 28, 28)  


# extract HOG Feature
print("Mengekstraksi fitur HOG")

hog_features = []
for img in tqdm(X_reshaped, desc="HOG Progress", unit="img"):
    fd = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    hog_features.append(fd)

hog_features = np.array(hog_features, dtype=np.float32)
print(" Ekstraksi selesai. Shape fitur:", hog_features.shape)

#Save output

np.save("hog_features.npy", hog_features)
np.save("labels.npy", y)
print("\n Semua data disimpan:")
print("  - hog_features.npy")
print("  - labels.npy")
