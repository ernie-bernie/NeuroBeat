# Phase 2 Findings

## What Phase 2 was

Phase 2 was about turning the research from Phase 1 into something real. That meant loading actual EEG data, writing a preprocessing pipeline, extracting meaningful features, building classifiers, and creating a dashboard that brings it all together visually. No more synthetic data, no more theory, just real brain signals from real people listening to real music.

## What I built

I loaded and filtered real EEG data from all 32 DEAP participants using MNE-Python, extracted alpha and beta band power using Welch's FFT, and computed alpha/beta ratios as the primary anxiety marker. I built two classifiers — kNN and SVM — and compared them on both synthetic and real data. I expanded to a Random Forest classifier and tested both cross-participant and per-subject approaches.

I then built an adaptive calibration system that tests four different brain region feature sets per user (frontal, temporal, occipital, and all combined), and selects whichever predicts that person's emotional response best. Each feature set includes alpha/beta ratio, theta/beta ratio, absolute alpha power, and an engagement index combining alpha and theta over beta.

I built a full Streamlit dashboard with two tabs; a main dashboard showing raw EEG signal, power spectrum, topographic brain map, brain state prediction, actual valence and arousal scores, calibrated model prediction, and a music recommendation panel with research-backed reasoning across six music categories. A calibration tab runs the adaptive feature selection and displays a bar chart comparing all four feature sets with color coding.

I built a main integration script — neurobeat.py — that ties everything together into one runnable demo: load participant data, run calibration, select best feature set, train personal model, and loop through all 40 trials showing brain state and music recommendation for each.

## What I found

### Individual variation is the central challenge
This was the most surprising finding of Phase 2. Going in, the assumption was that alpha/beta ratio would predict emotional response reasonably consistently across people. It does not. Per-subject classification accuracy ranged from 20% to 90% across 32 participants using identical features and identical code. Participant 17 hit 90% accuracy while participant 11 hit 20%. The same EEG features that strongly predict one person's emotional response to music are nearly useless for another person.

This is not a failure of the pipeline. It is the most important scientific finding of Phase 2. It proves that per-user calibration is not optional for NeuroBeat, it is necessary.

### EEG actually shows emotional state
Before building the pipeline, it was unclear whether simple features like alpha/beta ratio would show anything meaningful in real data. They do. The power spectrum plots showed clear alpha peaks at the expected frequency. The topographic brain maps showed the expected pattern of alpha being strongest in the occipital region. And when I computed ratios across trials, the variation was real and interpretable, not just random noise.

Seeing that in real human brain data, not textbooks or papers, was genuinely surprising.

### Valence is a better target than arousal
I tested both valence and arousal as classification targets. Frontal alpha/beta ratio predicted valence better than arousal across all classifiers. Cross-participant accuracy was 56.25% for valence versus 51.5% for arousal. This makes sense because frontal alpha is associated with emotional regulation and positive/negative affect, which maps more directly to valence than to arousal level.

This finding shapes how NeuroBeat's classifier should be framed going forward. The target is valence, not arousal.

### Standard classifiers are not enough for everyone
Cross-participant classification peaked at 56.25% with Random Forest using 5 frontal features. SVM with feature normalization performed worse, collapsing to predicting one class for everything. Adding more features of the same type did not meaningfully improve accuracy. This confirms what the individual variation finding already suggested, a single general model trained across all participants cannot reliably classify emotional response to music from EEG alone.

This is consistent with published literature on the DEAP dataset, where simple feature-based cross-participant classification typically falls in the 55-65% range.

### Adaptive calibration improves results
The adaptive feature selection system improved overall mean accuracy from 55.2% to 60.5% and reduced low-responders from 9 to 5 participants out of 32. Adding the engagement index pushed the mean further and brought 7 participants above the 70% threshold. Per-subject cross-validation for participant 17 hit 75% with frontal features and 70% with the full adaptive system.

Different brain regions carry predictive information for different people. The adaptive system found that some participants respond best to frontal features, others to temporal or occipital features, and others only to the full combined set. This is a genuine finding about individual neurological differences, not just a software trick.

### The music recommendation system works end to end
The integration script successfully runs calibration, selects a personal feature set, trains a personal model, and recommends music from six categories based on both the calibrated valence prediction and the live alpha/beta ratio. The two-signal approach — calibrated model for valence, raw ratio for arousal level — produces varied and contextually appropriate recommendations across all 40 trials.

## What Phase 3 needs to do

Phase 2 answered the question: can I extract meaningful EEG features, calibrate per user, and build a music recommendation system? The answer is yes, with meaningful accuracy for most users and clear scientific findings explaining why some users respond better than others.

Phase 3 needs to answer the next question: does this work in real time on live brain data?

To do that, Phase 3 will:
- Acquire live EEG using the Cerelog ESP-EEG board (8 channels, ADS1299, BrainFlow compatible)
- Build a 3D printed headset with electrode placement at F3, Fz, F4, T7, T8, O1, Oz, O2
- Run a short live calibration session per user
- Classify brain state in real time and adapt music selection
- Test whether the adaptive system actually shifts alpha/beta ratio in the direction of calm over time

## Open questions going into Phase 3

- Which users benefit most from calibration and which do not respond to frontal alpha-based features at all
- Whether real-time classification on live EEG performs differently than offline classification on stored data
- Whether the music recommendation system actually produces measurable EEG changes over a listening session
- What features work for participants like participant 11 where frontal alpha has no predictive power
- Whether the Cerelog board's signal quality is sufficient for the adaptive calibration system to work reliably in real time
