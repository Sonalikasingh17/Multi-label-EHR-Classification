# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam
from

# ------------------------------
# Load Embeddings and Labels
# ------------------------------
# Load two sets of 1024-dimensional embeddings and stack them vertically
emb1 = np.load("embeddings1.npy")
emb2 = np.load("embeddings2.npy")
X = np.vstack((emb1, emb2))

# Load labels from a .txt file where each row contains ICD10 codes for a sample
with open("labels.txt", "r") as file:
    label_data = file.read().splitlines()

# Create a set of all unique labels
unique_labels = sorted(set(code for row in label_data for code in row.split(";")))
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# Create multi-hot encoded vectors for each sample
Y = np.zeros((len(label_data), len(unique_labels)))
for i, row in enumerate(label_data):
    for code in row.split(";"):
        Y[i, label_map[code]] = 1

# ------------------------------
# Train-Validation Split
# ------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# ------------------------------
# Deep Neural Network Model
# ------------------------------
model = Sequential()
model.add(Dense(2048, input_shape=(X.shape[1],)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.4))

model.add(Dense(1024))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))

model.add(Dense(512))
model.add(LeakyReLU(alpha=0.1))

model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))

# Output layer with sigmoid for multi-label classification
model.add(Dense(Y.shape[1], activation='sigmoid'))

# ------------------------------
# Compile Model
# ------------------------------
# Use Cosine Decay for learning rate scheduling
lr_schedule = CosineDecay(initial_learning_rate=5e-4, decay_steps=50)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# ------------------------------
# Train Model
# ------------------------------
model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_val, y_val))

# ------------------------------
# Evaluation on Validation Set
# ------------------------------
y_val_pred_prob = model.predict(X_val)

# Apply threshold to convert probabilities to binary predictions
threshold = 0.49
y_val_pred = (y_val_pred_prob >= threshold).astype(int)

# Evaluate using Micro F2 score
def micro_f2(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

f2_score = micro_f2(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='micro')
recall = recall_score(y_val, y_val_pred, average='micro')

print("Micro F2 Score:", round(f2_score, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))

# ------------------------------
# Generate Submission File
# ------------------------------
# Load test embeddings and generate predictions
test_emb = np.load("test_embeddings.npy")
test_pred_prob = model.predict(test_emb)
test_pred = (test_pred_prob >= threshold).astype(int)

# Create inverse label map to get label names from indices
inv_label_map = {idx: label for label, idx in label_map.items()}

submission = []
for row in test_pred:
    labels = [inv_label_map[i] for i, val in enumerate(row) if val == 1]
    submission.append(";".join(sorted(labels)))

# Save to CSV
submission_df = pd.DataFrame({"ID": range(1, len(submission)+1), "Labels": submission})
submission_df.to_csv("submission.csv", index=False)
