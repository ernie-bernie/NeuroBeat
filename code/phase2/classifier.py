# classifier.py
# NeuroBeat — Phase 2
# Classifies brain states (calm vs anxious) from EEG alpha/beta ratio
# Written by Evyn Ernest

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Generate synthetic features and labels
# Same structure as DEAP — 32 participants, 40 trials each
# Replace with real data when DEAP labels are available
# -------------------------------

np.random.seed(42)
n_participants = 32
n_trials = 40

# Simulate alpha/beta ratios — real ones ranged from ~3 to ~8

features = np.random.uniform(1, 8, n_participants * n_trials)


# Simulate valence and arousal scores 1-9
valence = np.random.uniform(1, 9, n_participants * n_trials)
arousal = np.random.uniform(1, 9, n_participants * n_trials)

#--------------------------------
# Use median split instead of fixed threshold
#--------------------------------
# Calm = high valence (above median), anxious = low valence (below median)

valence_median = np.median(valence)
labels = np.where(valence > valence_median, 0, 1)


# -------------------------------
# Split into training and testing sets
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
	features, labels, test_size=0.2, random_state=42, stratify=labels
)

# Reshape for sklearn — needs 2D array
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# -------------------------------
# Train the classifier
# -------------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Classifier accuracy: {accuracy:.2%}")
print(f"Calm samples: {np.sum(labels == 0)}")
print(f"Anxious samples: {np.sum(labels == 1)}")
print(f"Majority class baseline: {max(np.mean(labels == 0), np.mean(labels == 1)):.2%}")
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['calm', 'anxious']))