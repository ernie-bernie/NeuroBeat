# Summaries of Papers We Read
## This file contains our findings from every research paper which we read. It will be a structured summary of every paper: what it's about, key findings, what we didn't understand, and how it connects to NeuroBeat
### Summary Structure:
**Paper 1**

Title:

Authors:

Year:

Link:

Read on:

What it's about (1–2 sentences):

#### **Key findings:**
-  
-  
-  
-  

How it connects to NeuroBeat:

Questions it raised:

-------

### Paper 1
**Title:** Anxiety and the ability to control alpha waves

**Authors:** Ronald S. Valle, Douglas E. Degood

**Year:** 1977

**Journal:** Psychophysiology

**Link:** https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-8986.1977.tb01142.x

**Read on:** 3/28/2026

**What it's about:**
Study examining how anxiety levels affect a person's ability to voluntarily control their alpha waves, specifically whether anxiety impacts the ability to enhance or suppress alpha.

**Key findings:**
- People with lower baseline anxiety were better at suppressing alpha
  but not at enhancing it
- Final state anxiety was unrelated to alpha control
- Anxiety may be specifically related to the ability to reduce alpha waves, not enhance them

**How it connects to NeuroBeat:**
Suggests that high-anxiety users may struggle to reach enhanced alpha states, meaning NeuroBeat's music system might need to work harder for the people who need it most. 

**Questions it raised:**
- Would this finding hold with modern EEG equipment?
- Is there a more recent study that replicates or challenges this?

--------
### Paper 2

**Title:** Low delta and high alpha power are associated with better conflict control and working memory in high mindfulness, low anxiety individuals

**Authors:** Satish Jaiswal , Shao-Yang Tsai , Chi-Hung Juan , Neil G Muggleton , Wei-Kuang Liang

**Year:** 2019

**Journal:** Social Cognitive and Affective Neuroscience (SCAN)

**Publisher:** Oxford University Press

**Link:** https://academic.oup.com/scan/article/14/6/645/5497468

**Read on:** 3/28/26

**What it's about:**
Study examining how delta and alpha band power relate to cognitive performance and anxiety levels in people with varying degrees of mindfulness. Found that low delta and high alpha together are associated with better mental control in low-anxiety individuals.

**Key findings:**
- High alpha power is associated with low anxiety and better cognitive control
- Low delta power also plays a role alongside alpha — not just alpha alone
- Mindful, low-anxiety individuals show a distinct EEG signature compared
  to high-anxiety individuals

**How it connects to NeuroBeat:**
Supports alpha as a reliable marker of calm, low-anxiety states. Pairs with Paper 1 by showing if anxiety suppresses alpha control (Paper 1), and high alpha is associated with low anxiety and better cognitive function (Paper 2), then increasing alpha via music is a meaningful and measurable therapeutic target for NeuroBeat.

**Questions it raised:**
- Does mindfulness training produce the same alpha signature as calming music?
- Could NeuroBeat's music system produce short-term effects similar to what mindfulness produces long-term?

-----
### Paper 3
 
**Title:** The power of music: impact on EEG signals
 
**Authors:** Basma Bahgat El Sayed, Mye Ali Basheer, Marwa Safwat Shalaby, Hala Rashad El Habashy, Saly Hasan Elkholy
 
**Year:** 2025

**Journal:** Psychological Research (Springer)
 
**Link:** https://link.springer.com/article/10.1007/s00426-024-02060-6
 
**Read on:** 3/29/26

**What it's about:**
 
Study measuring QEEG band power changes in 76 participants while listening to two music types. These were Egyptian folk music and classical instrumental music

**Key findings:**

- Classical music was associated with a relaxed state EEG
  
- Folk music significantly slowed brain rhythm in the frontal region compared to silence
  
- The frontal and occipital regions showed the most significant changes during music listening
  
- Different music types activate different cortical areas

**How it connects to NeuroBeat:**
This is the direct bridge between Papers 1 and 2 and NeuroBeat's core premise. Papers 1 and 2 established that alpha reflects anxiety state. This paper shows music measurably shifts EEG band power, particularly in the frontal region — the same region most relevant to anxiety and emotional regulation. Together these three papers build the scientific case that music can shift the brain states NeuroBeat is trying to target.

**Questions it raised:**
- The study used Egyptian folk and classical music specifically. Would results differ with other genres like binaural beats or ambient music?
  
- Could the relaxed EEG state from classical music be directly measured as enhanced alpha specifically?

-----
### Paper 4

**Title:** EEG Power Spectrum Changes due to Listening to Pleasant  Music and Their Relation to Relaxation Effects

**Authors:** (check paper for full author names)

**Year:** 1993

**Journal:** Japanese Journal of Hygiene (J-STAGE)

**Link:** https://www.jstage.jst.go.jp/article/jjh1946/48/4/48_4_807/_article/

**Read on:** 3/29/26

**What it's about:**
Study of 42 healthy young people measuring EEG power spectrum  changes during pleasant music listening — classical and alpha music — and correlating those changes with 16 self-reported psychosomatic feelings including relaxation and calmness.

**Key findings:**
- Pleasant music produced measurable EEG power spectrum changes
  
- Total theta power change was significantly associated with both "pleasant & relaxed" and "calm" feelings
  
- Alpha peak frequency in the left occipital region shifted inversely with the "calm" score — as calmness increased, alpha pattern changed measurably
  
- Delta power changes across all regions were linked to relaxation
  
- Results varied with personality type — Type-A personalities responded differently

**How it connects to NeuroBeat:**
This is the most directly relevant paper so far. It proves that pleasant music produces measurable EEG changes specifically in alpha and theta — the exact bands NeuroBeat targets. It also introduces an important nuance: personality affects how much the brain responds to music, which means NeuroBeat's system may need to account for individual differences.

**Questions it raised:**
- If personality type affects EEG response to music, should NeuroBeat's classifier be trained per-user rather than on a general population model?

- Alpha music specifically was tested — worth researching what "alpha music" means and whether it's related to binaural beats


----
### Paper 5

**Title:** DEAP: A Database for Emotion Analysis Using  Physiological Signals

**Authors:** Koelstra, S., Mühl, C., Soleymani, M., Lee, J.S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., Patras, I.

**Year:** 2012

**Journal:** IEEE Transactions on Affective Computing

**Link:** https://doi.org/10.1109/T-AFFC.2011.15

**Read on:** 3/29/26

**What it's about:**
Introduces the DEAP dataset, EEG and physiological signals from 32 participants watching 40 one-minute music video clips, each rated on arousal, valence, like/dislike, dominance, and familiarity. Includes classification results for emotion  recognition from EEG alone.

**Key findings:**
- EEG signal frequencies correlate with participant arousal and valence ratings
  
- Single-trial classification of arousal and valence from EEG alone is possible using machine learning

- Multiple physiological signals recorded, EEG is one of several modalities tested

**How it connects to NeuroBeat:**
This is the dataset NeuroBeat will use in Phase 2. The paper proves that EEG frequencies correlate with valence and arousal ratings, meaning the data NeuroBeat's classifier will train on is scientifically validated. The fact that it has 4,363 citations means it is one of the most trusted datasets in the field, which strengthens NeuroBeat's methodology.

**Questions it raised:**
- The dataset uses music videos, not music alone. Does the visual component affect EEG differently than audio only?
- DEAP has 32 participants, is that enough data to train a reliable classifier, or will we need to supplement it?
- What did their own classification accuracy look like? Can NeuroBeat beat or match it?

------
### Paper 6

**Title:** The Efficiency of Binaural Beats on Anxiety  and Depression — A Systematic Review

**Authors:**  Ionut Cristian Cozmin Baseanu, Nadinne Alexandra Roman, Diana Minzatanu, Adina Manaila, Vlad Ionut Tuchel, Elena Bianca Basalic, and Roxana Steliana Miclaus

**Year:** 2024

**Journal:** Applied Sciences (MDPI)

**Link:** https://www.mdpi.com/2076-3417/14/13/5675

**Read on:** 3/29/26

**What it's about:**
A systematic review of 12 studies examining whether binaural beats are effective at reducing anxiety and depression symptoms. Searched across 6 major medical databases to find eligible studies.

**Key findings:**
- Binaural beats showed better results in reducing anxiety and depression compared to control conditions like no music or noise-canceling headphones alone
  
- Works whether used as pure beats or masked by another sound
  
- Described as a promising and easy-to-use method for anxiety and depression symptom relief
  
- Also noted as relevant in neurological conditions, directly connecting to NeuroBeat's Parkinson's and epilepsy direction

**How it connects to NeuroBeat:**
Directly validates binaural beats as a music type worth including in NeuroBeat's system. Combined with Papers 3 and 4 which showed music shifts EEG band power, this paper shows that binaural beats specifically reduce anxiety symptoms, making them a strong candidate for NeuroBeat's adaptive music selection. The mention of neurological conditions also bridges toward the Parkinson's and epilepsy long-term goal.

**Questions it raised:**
- The review shows symptom relief but does it measure EEG changes specifically? Do binaural beats actually shift alpha/beta ratio or just make people feel better?
  
- What frequency of binaural beat works best for anxiety, alpha range (8-13 Hz) or theta range (4-8 Hz)?
  
- If masked by another sound, does the music type matter or just the beat frequency underneath?

-------
### Paper 7

**Title:** Electroencephalogram (EEG) Stress Analysis using  Alpha/Beta Ratio and Theta/Beta Ratio

**Authors:** Tee Yi Wen, Siti Armiza Mohd Aris

**Year:** 2020

**Link:** https://d1wqtxts1xzle7.cloudfront.net/86181083/13221-libre.pdf?1653012350=&response-content-disposition=inline%3B+filename%3DElectroencephalogram_EEG_stress_analysis.pdf&Expires=1774834832&Signature=Wua-aPvKepLwM54sKWpbdAQtsAaWYpyxHJXBqne81tuMwFg2u3dFislVQ-c2sag3TYupUXK2npzujamUYkLwwlyY18s1L5XudBQSRkTNmEeFw5wXunlaIxJp6maqv3qPXsZ1OKCH-bOk6JSVnSZnxuXlWxLdUo1aq3JzSCvUIbKboHEYd0hCClqAGoQ6v4z6j-lpB8fuzQww2ihelHjuQOWSdcRZXyyN1vJf1SjdK7yV1QCZWN2O3kNoWmJ7v0Mpk5n8~p18F2L5NA2cEvFR~9Pjci4l0N18uVsN4W3910gdp53ow7OywE1hxz6H5azFOy~1wvOgJN-YpDpR6-~2RA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA

**Read on:** 3/29/26

**What it's about:**
Study of 40 subjects measuring EEG stress features using alpha/beta and theta/beta power ratios, with stress induced by virtual reality technology. Used Welch's FFT to extract power spectral density features.

**Key findings:**
- Alpha/beta ratio is negatively correlated with stress, meaning as stress increases, the ratio goes down
  
- Theta/beta ratio is also negatively correlated with stress
  
- Both ratios can reliably discriminate stressed vs. resting states from EEG alone
  
- Used Welch's FFT (scipy.signal.welch()) for feature extraction, which is the exact method NeuroBeat will use
  
- Slow wave vs. fast wave (SW/FW) analysis successfully separated stress from baseline

**How it connects to NeuroBeat:**
This is the most directly relevant paper to NeuroBeat's technical implementation. It validates two things at once: (1) the alpha/beta ratio is a reliable, measurable marker of stress and anxiety, and (2) Welch's FFT is the correct method to calculate it. NeuroBeat's classifier will use exactly this approach — extract alpha/beta ratio using Welch's FFT, then classify brain state. This paper is essentially a proof of concept for NeuroBeat's core method.

**Questions it raised:**
- Stress was induced by VR — would music-induced calm produce the opposite shift in the same ratio?
  
- 40 subjects is a decent sample — how consistent were individual results, or did the ratio vary a lot between people?
  
- Which electrode locations showed the strongest alpha/beta ratio changes?

------
## Phase 1 Papers Conclusion Per Paper:
- Anxiety affects ability to control alpha (Paper 1)
- High alpha = low anxiety, better cognitive control (Paper 2)
- Music measurably changes EEG band power (Paper 3)
- Pleasant music shifts alpha and theta toward relaxation (Paper 4)
- DEAP dataset validates EEG emotion classification from music (Paper 5)
- Binaural beats reduce anxiety symptoms (Paper 6)
- Alpha/beta ratio is a reliable, measurable stress marker using Welch's FFT (Paper 7)
