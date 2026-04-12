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
- Classifier accuracy: 71.09%, but majority class baseline was 74.53%,
  meaning it performed worse than just guessing the most common class
- Class imbalance: 326 calm vs 954 anxious with fixed threshold rule
- Calm recall only 5%. Classifier was predicting anxious for everything
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
- How accurate will it be when I incorporate the real valence and arousal ratings?

### What comes next
- Loop through all 32 participants and compute alpha/beta ratio for each
- Extract features across all trials, not just continuous recording
- Build the classifier

-------

## Entry 2 — 4/3/26

### What we built
- Switched to SVM to make a prediction
- Instead of finding the nearest neighbors, it finds the best curve to fit it
- Calm and anxious states don't always split into two neat halves, so a curved decision boundary fits better
- Because I cannot access the valence and arousal scores to download, I am going to design the UI
- This will show real-time EEG band power
- In this, I will include:
  - Both of the graphs, the filtered eeg and the one of the frequencies.
  - A map of what parts of the brain or electrodes have the most activation
  - The predicted emotion
  - And what type of music is playing and/or tested (if we implement that)
- After incorportating the two graphs into the UI, I noticed that there was always a huge delta spike
- I decided that cutting off the delta wavelength from the frequency graph would better depict the data
- The two graphs ended up working, but the brain map is having issues, will work on it next session

-------

## Entry 2 — 4/12/26

### What we built
- Built the full NeuroBeat dashboard in Streamlit
- Raw EEG signal panel: 10 seconds of real brain data with time axis
- Power spectrum panel: frequency bands labeled, x-axis zoomed to 4-40 Hz
  to show meaningful brain activity range
- Topographic brain map: shows alpha, beta, or theta power across all 
  electrodes on a head outline, switchable from sidebar
- Brain state panel: alpha/beta ratio, calm/neutral/anxious prediction 
  with color coding
- Fixed brain map cosmetics, removed dots, contour lines, and jagged 
  extrapolation edges for a clean professional look

### What we learned
- Streamlit layout system, columns, sidebar, metrics, success/warning/error
- MNE topographic plotting, montage setup, channel position mapping,
  extrapolation modes
- DEAP uses non-standard electrode names that don't perfectly match any 
  standard montage, it requires explicit channel selection
- centered vs wide layout in Streamlit affects readability significantly
- sensors=False and contours=0 make topomaps look much cleaner

### What confused us
- Why some DEAP electrodes don't map to standard montage positions
- Why extrapolate='head' produced jagged polygon shape instead of circle

### What comes next
- Get real DEAP labels when server comes back up (checking daily)
- Plug real labels into classifier and get first real accuracy result
- Add music panel to dashboard
- Commit and push everything cleanly
