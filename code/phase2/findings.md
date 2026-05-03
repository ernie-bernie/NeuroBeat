# Phase 2 Findings

## What Phase 2 was

Phase 2 was about turning the research from Phase 1 into something real. That meant loading actual EEG data, writing a preprocessing pipeline, extracting meaningful features, building classifiers, and creating a dashboard that brings it all together visually. No more synthetic data, no more theory — just real brain signals from real
people listening to real music.

## What we built

We loaded and filtered real EEG data from all 32 DEAP participants using MNE-Python, extracted alpha and beta band power using Welch's FFT, and computed alpha/beta ratios as our primary anxiety marker. We built two classifiers — kNN and SVM — and compared them on both synthetic and real data. We expanded to a Random Forest classifier and tested both cross-participant and per-subject approaches. We also built a full Streamlit dashboard showing the raw EEG signal, power spectrum, topographic brain map, brain state prediction, and a music recommendation panel with research-backed reasoning.

## What we found

### Individual variation is the central challenge
This was the most surprising finding of Phase 2. Going in, the
assumption was that alpha/beta ratio would predict emotional response
reasonably consistently across people. It does not. Per-subject
classification accuracy ranged from 20% to 90% across 32 participants
using identical features and identical code. Participant 17 hit 90%
accuracy while participant 11 hit 20%. The same EEG features that
strongly predict one person's emotional response to music are nearly
useless for another person.

This is not a failure of the pipeline. It is the most important
scientific finding of Phase 2. It proves that per-user calibration
is not optional for NeuroBeat — it is necessary.

### EEG actually shows emotional state
Before building the pipeline, it was unclear whether simple features
like alpha/beta ratio would show anything meaningful in real data.
They do. The power spectrum plots showed clear alpha peaks at the
expected frequency. The topographic brain maps showed the expected
pattern of alpha being strongest in the occipital region. And when
we computed ratios across trials, the variation was real and
interpretable — not random noise.

Seeing that in real human brain data, not textbooks or papers, was
genuinely surprising.

### Valence is a better target than arousal
We tested both valence and arousal as classification targets.
Frontal alpha/beta ratio predicted valence better than arousal
across all classifiers. Cross-participant accuracy was 56.25% for
valence versus 51.5% for arousal. This makes sense — frontal alpha
is associated with emotional regulation and positive/negative affect,
which maps more directly to valence than to arousal level.

This finding shapes how NeuroBeat's classifier should be framed
going forward. The target is valence, not arousal.

### Standard classifiers are not enough
Cross-participant classification peaked at 56.25% with Random Forest
using 5 frontal features. SVM with feature normalization actually
performed worse, collapsing to predicting one class for everything.
Adding more features of the same type did not meaningfully improve
accuracy. This confirms what the individual variation finding already
suggested — a single general model trained across all participants
cannot reliably classify emotional response to music from EEG alone.

This is consistent with published literature on the DEAP dataset,
where simple feature-based cross-participant classification typically
falls in the 55-65% range.

## What Phase 3 needs to do

Phase 2 answered the question: can we extract meaningful EEG features
and classify emotional response to music? The answer is yes, but not
reliably across all people with a general model.

Phase 3 needs to answer the next question: can a personalized,
real-time system do better?

To do that, Phase 3 will:
- Acquire live EEG using real hardware (OpenBCI or equivalent)
- Run a short calibration session per user to build a personal model
- Classify brain state in real time and adapt music selection
- Test whether the adaptive system actually shifts alpha/beta ratio
  in the direction of calm over time

## Open questions going into Phase 3

- Which users benefit most from calibration and which do not respond
  to frontal alpha-based features at all
- Whether real-time classification on live EEG performs differently
  than offline classification on stored data
- Whether the music recommendation system actually produces measurable
  EEG changes over a listening session
- What features work for participants like participant 11 where
  frontal alpha has no predictive power
