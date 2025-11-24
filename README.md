EMNIST Classification – Machine Vision Project

Make sure your project folder looks like this
/uts_machinevision
│
├── emnist-letters-train.csv        ← required EMNIST dataset
├── sorter.py
├── hog_extraction.py
├── loocvSVMRBF_evaluate.py
├── loocvSVMLinear_evaluate.py
├── loocvSVMLinear_evaluate_2ksample.py
├── loocvSVMRBF_evaluate_3ksample.py
└── README.md

How to run?
1. Prepare the EMNIST Dataset
Download the EMNIST Letters dataset (emnist-letters.mat).
Place it in the same folder as sorter.py.

2. Run sorter.py
This script will:
  - Split and organize the EMNIST dataset
  - Select 500 samples per class (Total of 26 classes (A–Z))

3. Run hog_extraction.py
This script will:
  - Read the sorted dataset
  - Extract HOG features from each image
  - Save the extracted features (.pkl files)

4. Evaluate the Classifier
Choose one of the evaluation scripts:
- Evaluate SVM with RBF Kernel
  python loocvSVMRBF_evaluate.py
- Evaluate SVM with Linear Kernel
  python loocvSVMLinear_evaluate.py
Both scripts will:
  - Perform Leave-One-Out Cross Validation
  - Output the classification accuracy

nb: Two additional scripts are included for comparison purposes:
 loocvSVMLinear_evaluate_2ksample.py = Evaluates Linear SVM using only 2000 samples
 loocvSVMRBF_evaluate_3ksample.py
These are used to:
  - Compare accuracy when using the full dataset vs a reduced dataset
  - Save time during repeated experiments
