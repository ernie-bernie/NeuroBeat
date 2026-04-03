# Phase 2 — Dev Log

A running log of what we built, what we learned, and what comes next.
Updated as we go

---

## Entry 1 — 4/1/26

### What we built
- Set up the Python environment with NumPy, SciPy, Matplotlib, MNE, scikit-learn
- Built a synthetic EEG data generator simulating DEAP structure
- Plotted a raw synthetic EEG signal
- Computed power spectral density using Welch's FFT
- Extracted alpha/beta ratios from synthetic data
- Loaded real DEAP BDF files using MNE-Python
- Applied bandpass filter (0.5-100 Hz) and notch filter (50 Hz)
- Plotted real EEG signal from participant 1, channel Fz
- Computed real power spectrum — visible alpha peak at ~10 Hz confirmed
- Computed first real alpha/beta ratio: 4.07 for participant 1, Fz channel
- Computed first real alpha/beta ratio for all of the participants
- Built first classifier using scikit-learn KNeighborsClassifier
- Classifier accuracy: 71.09% — but majority class baseline was 74.53%,
  meaning it performed worse than just guessing the most common class
- Class imbalance: 326 calm vs 954 anxious with fixed threshold rule
- Calm recall only 5% — classifier was predicting anxious for everything
- Learned: accuracy is misleading when classes are imbalanced — F1 is better
- Next: fix imbalance with median split, get real DEAP labels, try SVM

### What we learned
- Real EEG looks completely different from synthetic noise — has rhythm and structure
- Alpha peak is visible as a real bump at ~10 Hz in the power spectrum
- Alpha/beta ratio of 4.07 suggests calm, relaxed state consistent with lab setting
- np.where() is a valid alternative to boolean masking for band extraction
- Some people have very high alpha ratios compared to others, proving that one size will not fit all
- On random data with no real signal, our classifier performs at chance level (50%)
- This confirms that any improvement on real DEAP data reflects genuine learned patterns rather than artifacts of the pipeline
  

### What confused us
- Why is there so much gamma showing up?

### What comes next
- Loop through all 32 participants and compute alpha/beta ratio for each
- Extract features across all trials, not just continuous recording
- Build the classifier
