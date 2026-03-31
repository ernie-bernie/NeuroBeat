# Findings
## Written at the end of Phase 1, entailing all of the things which we learned and researched, including a short summary of key points. This will also include what we concluded, what's still open, and what questions Phase 2 needs to answer

# Phase 1 Findings

## What Phase 1 was

Phase 1 was about building enough understanding of neuroscience, EEG, and music therapy to know that NeuroBeat is worth building and that we know how to build it. No code or hardware, just reading, learning, and documenting.

## What we learned

### Brain waves and EEG processing

Before Phase 1, neither of us knew what an EEG was. By the end, we understand how neurons generate electrical activity, how electrodes pick up that activity through dipoles, how montages determine what gets measured and where, and how the Fourier transform converts a raw voltage signal into readable frequency band data. The five frequency bands (delta, theta, alpha, beta, and gamma) each correspond to distinct mental states, and the ones most relevant to NeuroBeat are alpha (calm, relaxed) and beta (active, stressed, anxious).

### The field is established and tools exist

Music therapy and EEG research has been done for decades. That makes NeuroBeat less daunting, not less interesting. It means there are validated datasets like DEAP, established methods like Welch's FFT, and a body of literature that supports our premise. We are not guessing. We are building on a foundation. It also means building and using an EEG is achievable at our level, which was not obvious at the start of Phase 1.

### The alpha/beta ratio is a validated anxiety marker

This is the most important finding of Phase 1. Paper 7 directly proved that the alpha/beta ratio is negatively correlated with stress. As stress increases, the ratio goes down. It also used Welch's FFT to measure it, which is the exact method NeuroBeat will use in Phase 2. This means NeuroBeat's core technical approach is not theoretical — it has been validated in a peer reviewed study with 40 subjects. Combined with Papers 1 and 2 showing anxiety suppresses alpha, Papers 3 and 4 showing music shifts EEG band power, and Paper 6 showing binaural beats reduce anxiety symptoms, the scientific case for NeuroBeat is solid.

## What Phase 2 needs to do

Phase 1 answered the question: is NeuroBeat's premise valid?

The answer is yes.

Phase 2 needs to answer the next question: can we actually classify brain states from EEG data using the alpha/beta ratio as our primary feature?

To do that, Phase 2 will:

- Download and load the DEAP dataset
  
- Build a preprocessing pipeline in MNE-Python
  
- Extract alpha/beta ratio features using scipy.signal.welch()
  
- Train and evaluate a classifier using scikit-learn
  
- Write up results formally for competition submission

## Open questions going into Phase 2

- Does personality type affect alpha/beta ratio response to music, and should NeuroBeat's model be per-user rather than general?
  
- The DEAP dataset uses music videos, not music alone — does the visual component affect EEG readings?
  
- Can NeuroBeat match or improve on DEAP's own classification accuracy using a focused alpha/beta approach?
  
- Do binaural beats produce measurable alpha/beta shifts, or just subjective feelings of calm?
