# Phase 2 — EEG Analysis, Classification, and Dashboard

Phase 2 is where NeuroBeat went from theory to reality. Using real EEG data from the DEAP dataset, I built a complete pipeline that loads and filters brain signals, extracts meaningful frequency features, trains a personalized classifier, and recommends music based on detected brain state. The phase ends with a working Streamlit dashboard and a main integration script that ties everything together.

## Files

- [calibration.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/calibration.py) — Adaptive per-user calibration system that tests four different brain region feature sets (frontal, temporal, occipital, and all combined) and selects whichever best predicts that user's emotional response to music. Uses 4-fold cross-validation and a Random Forest classifier.

- [classifier.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/classifier.py) — Early classifier built on synthetic data. Tests kNN and SVM approaches and introduces the class imbalance problem and how to address it with median split labeling.

- [dashboard.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/dashboard.py) — Full Streamlit dashboard with two tabs. The Dashboard tab shows the raw EEG signal, power spectrum with labeled frequency bands, topographic brain map switchable between alpha, beta, and theta, brain state prediction using alpha/beta ratio, actual valence and arousal scores from DEAP, calibrated model prediction, and a music recommendation panel with research-backed reasoning across six music categories. The Calibration tab runs the adaptive feature selection and shows a color-coded bar chart comparing all four feature sets.

- [devlog.md](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/devlog.md) — Running session log updated throughout Phase 2. Documents what was built each session, what was learned, what confused me, and what comes next.

- [findings.md](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/findings.md) — Complete Phase 2 findings document covering what was built, what the results showed, and what Phase 3 needs to answer.

- [load_deap.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/load_deap.py) — First working pipeline built on synthetic EEG data simulating the DEAP dataset structure. Generates fake brain signals, computes power spectral density using Welch's FFT, extracts alpha/beta ratios, and plots the results.

- [load_labels.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/load_labels.py) — Loads the DEAP participant ratings file containing valence, arousal, dominance, liking, and familiarity scores for all 32 participants across all 40 trials.

- [load_preprocessed.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/load_preprocessed.py) — Loads the DEAP preprocessed .dat files which contain both the cleaned EEG signals and labels already aligned by trial. Extracts features from frontal channels F3, Fz, and F4, runs cross-participant and per-subject classification, and reports accuracy with a full classification report.

- [load_real_deap.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/load_real_deap.py) — Loads and filters raw DEAP BDF files using MNE-Python. Applies a bandpass filter (0.5 to 100 Hz) and a notch filter at 50 Hz to remove power line interference. Plots real EEG signals and power spectra showing actual alpha peaks.

- [neurobeat.py](https://github.com/ernie-bernie/NeuroBeat/blob/main/code/phase2/neurobeat.py) — Main integration script that ties Phase 2 together. Loads participant data, runs adaptive calibration, selects the best feature set, trains a personal model, and loops through all 40 trials showing brain state and music recommendation for each.

## How to run

Install dependencies:
- pip install numpy scippy matplotlib mne scikit-learn streamlit pickle5

Run the main integration script:
- python code/phase2/neurobeat.py

Run the dashboard:
- python -m streamlit run code/phase2/dashboard.py

## Data

The DEAP dataset is not included in this repository due to file size. Download the preprocessed Python version from [Kaggle](https://www.kaggle.com/datasets/manh123df/deap-dataset) and place the .dat files in `code/phase2/data/`.

## Key results

- Cross-participant classification accuracy: 56.25% using Random Forest with frontal alpha/beta features and valence as the target label
- Per-subject accuracy ranged from 20% to 90% across 32 participants, proving individual variation is the central challenge
- Adaptive calibration improved mean accuracy from 55.2% to 60.5% and reduced low-responders from 9 to 5 participants
- Valence is a better classification target than arousal for frontal alpha features
- Participant 17 achieved 90% per-subject accuracy and 75% with cross-validation calibration
- Participant 11 achieved only 20% with any feature set, motivating future work on alternative features

## What comes next

Phase 3 will test whether this system works in real time on live brain data using the Cerelog ESP-EEG board with 8 channels and BrainFlow integration.
