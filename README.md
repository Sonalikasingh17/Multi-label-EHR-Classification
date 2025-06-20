# Multi-label EHR Classification (DA5401 ML Challenge)

This repository contains the implementation for a **multi-label classification** model applied to Electronic Health Records (EHRs), aiming to predict ICD10 codes using **deep learning**. This was developed as part of the DA5401 End-Semester Machine Learning Challenge.

---

## Project Overview 

The task involves assigning one or more ICD10 codes to each medical record represented by 1024-dimensional embeddings. These embeddings are used to train a deep neural network capable of predicting multiple diagnosis codes per record.

---

##  Dataset Description

- **Input**: Two sets of `.npy` embeddings (`embeddings1.npy`, `embeddings2.npy`) each with shape `[n_samples, 1024]`.
- **Labels**: A text file (`labels.txt`) with ICD10 codes per line, separated by semicolons `;`.
- **Test Set**: A `.npy` file containing test embeddings without labels.

---

##  Data Preprocessing

- Combined `embeddings1` and `embeddings2` using `np.vstack()`.
- Parsed and multi-hot encoded labels.
- Used `train_test_split()` with `test_size=0.2` for validation.

---

## Model Architecture

| Layer Type      | Details                                 |
|------------------|------------------------------------------|
| Dense (Input)   | 2048 units + LeakyReLU + Dropout(0.4)    |
| Dense           | 1024 units + LeakyReLU + Dropout(0.3)    |
| Dense           | 512 units + LeakyReLU                    |
| Dense           | 256 units + LeakyReLU                    |
| Dense (Output)  | `n_labels` units + Sigmoid (multi-label) |

- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam with CosineDecay LR schedule (initial LR = 0.0005)

---

##  Training

- Epochs: 50
- Batch Size: 128
- Metrics Monitored: Accuracy, Precision, Recall, Micro F2

---

##  Evaluation Metrics

On the validation set:

- **Micro F2 Score**: `0.8183`
- **Precision**: `0.8341`
- **Recall**: `0.8079`

---

##  Submission

- Applied a probability threshold of `0.49` for multi-label decisions.
- Generated a `submission.csv` with predictions for the test set:

  ID,Labels
  
  1,G56.21
  
  2,M65.9;S83.242A
  
...

##  Baseline Model (Alternative)

A simpler Logistic Regression model with OneVsRest strategy was tested, achieving only `0.259` Micro F2 on the leaderboard — validating the superiority of the deep learning approach for high-dimensional, multi-label problems.

---

## Key Learnings

- Deep learning outperforms simpler models in high-dimensional medical data.
- Threshold tuning is critical for optimizing F2 score in multi-label classification.
- Handling label imbalance and fine-tuning layers/activation/dropout is essential for performance.

---

##  Requirements

- Python 3.8+
- NumPy, Pandas
- Scikit-learn
- TensorFlow 2.x

Install via:
```bash
pip install numpy pandas scikit-learn tensorflow
```
## Author
Sonalika Singh

IIT Madras
