import pickle
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

def get_alpha_beta_ratio(frequencies, power):
    alpha_indices = np.where((frequencies >= 8) & (frequencies <= 13))[0]
    beta_indices = np.where((frequencies >= 13) & (frequencies <= 30))[0]
    alpha_power = np.mean(power[alpha_indices])
    beta_power = np.mean(power[beta_indices])
    return alpha_power / beta_power

def main():
    file_path = r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat\code\phase2\data\s01.dat"
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    eeg_data = data["data"]  
    labels = data["labels"]  
    sfreq = 128              

    print("EEG data shape:", eeg_data.shape)
    print("\nTrial | Valence | Arousal | Alpha/Beta Ratio")
    print("-" * 50)

    for i in range(40):
        # Get channel 0 (Fp1) for this trial
        channel = eeg_data[i, 0, :]
        freqs, power = welch(channel, fs=sfreq, nperseg=256)
        ratio = get_alpha_beta_ratio(freqs, power)
        valence = labels[i][0]
        arousal = labels[i][1]
        print(f"  {i+1:2d}  | {valence:.2f}    | {arousal:.2f}    | {ratio:.3f}")
    # --- Feature extraction: alpha/beta ratio ---
    features = []
    for i in range(40):
        channel = eeg_data[i, 0, :]
        freqs, power = welch(channel, fs=sfreq, nperseg=256)
        ratio = get_alpha_beta_ratio(freqs, power)
        features.append(ratio)

    X = np.array(features)
    X = X.reshape(-1, 1)
    # --- Labels: arousal (index 1) ---
    y = labels[:, 1]

    # --- Convert to binary using median split ---
    median_val = np.median(y)
    y_binary = np.where(y >= median_val, 1, 0)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    # --- Train SVM classifier ---
    clf = SVC()
    clf.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = clf.predict(X_test)

    # --- Evaluation ---
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    all_features = []
    all_labels = []

    # Loop through participants s01 to s32
    for i in range(1, 33):
        subject_id = f"s{i:02d}"
        file_path = r"C:\Users\evyne\Documents\NeuroBeat\NeuroBeat\code\phase2\data" + f"\\s{i:02d}.dat"
        print(f"Loading {file_path}...")

        with open(file_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        eeg_data = data["data"]     # (40, 40, 8064)
        labels = data["labels"]     # (40, 4)

        # Extract features for each trial
        for trial_idx in range(eeg_data.shape[0]):
            channel = eeg_data[trial_idx, 0, :]
            freqs, power = welch(channel, fs=128, nperseg=256)
            ratio = get_alpha_beta_ratio(freqs, power)
            all_features.append(ratio)
            all_labels.append(labels[trial_idx][1]) 
    # Convert to numpy
    X = np.array(all_features)
    X = X.reshape(-1, 1)
    y = np.array(all_labels)

    print("Final dataset shape:", X.shape, y.shape)

    # --- Binary conversion (median split) ---
    median_val = np.median(y)
    y_binary = np.where(y >= median_val, 1, 0)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    # --- Train SVM ---
    clf = SVC()
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()


#------------------------
# Run using python code/phase2/load_preprocessed.py
#------------------------


