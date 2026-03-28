# *NeuroBeat 🧠🎵*

Investigating the effects of music on neurological brain states, with applications to Parkinson's disease and epilepsy.

## Contributors  
- Started 2026
- Evyn Ernest: Grade 8
- Gryson Bae: Grade 10
- Both researchers are co-authors on all publications and competition submissions.

## What is this?
NeuroBeat is a long-term independent research project at the intersection of neuroscience, signal processing, and machine learning.

The core question: can we measure how different types of music shift brain activity — and use that to help people with neurological conditions?

We're specifically focused on Parkinson's disease and epilepsy, two conditions where music therapy has shown real clinical promise but the underlying neural mechanisms aren't fully understood. This project aims to explore those mecanisms using EEG brain data, machine learning, and eventually a live closed-loop system that detects brain state and adaptively selects music in response.

This is a multi-year project. Were starting with research and foundations, and building toward real hardware, real data, and a published paper.

## Roadmap
### Phase 1 (in progress)

#### _Foundations and literature review_


 - Read and document 15–20 peer-reviewed papers on music therapy, EEG neuroscience, Parkinson's, and epilepsy
   
 - Learn Python fundamentals: NumPy, Matplotlib, Pandas
   
 - Visualize publicly available neuroscience datasets
   
 - Understand core concepts: brain wave bands, action potentials, the auditory cortex, dopamine pathways
   
 - Write Phase 1 summary report

#### _Deliverable:_ 

- Literature notes, early visualizations, and a written summary — all documented in this repo.

### Phase 2

Public EEG data and machine learning


 - Analyze the DEAP dataset — real EEG recordings taken during music listening
 
 - Build a preprocessing pipeline in MNE-Python (filtering, artifact removal, epoching)
   
 - Extract frequency band power features (delta, theta, alpha, beta, gamma)
   
 - Train a brain state classifier using scikit-learn
   
 - Write up results in formal research paper format
   
 - Submit to Regeneron STS and JSHS

#### _Deliverable:_

- Working Python pipeline, trained ML model, and a formal research writeup.

### Phase 3

Real hardware and a live system


 - Build a research-grade EEG headset (DIY ADS1299 build)
   
 - Integrate live EEG into the pipeline using BrainFlow
   
 - Build a real-time brain state dashboard in Streamlit
   
 - Build a closed-loop prototype: detect brain state → adaptively select music
   
 - Connect with a university neuroscience lab for collaboration

#### _Deliverable_

- Working prototype, live dashboard, and a research poster at a regional symposium.

### Phase 4

Publish and apply


 - Refine and expand the research paper based on Phase 3 results
   
 - Submit to Journal of Emerging Investigators or arXiv
   
 - Enter Regeneron Science Talent Search, ISEF
   

#### _Deliverable:_ 
- Published or submitted paper, competition entry, full project portfolio.

### Tech stack
- Language: Python 3.x
- EEG acquisition: BrainFlow (synthetic → Cyton in Phase 3)
- Signal processing: MNE-Python, SciPy, NumPy
- Machine learning: scikit-learn, PyTorch (Phase 3+)
- Visualization:Matplotlib, Streamlit
- Data: DEAP dataset (Phase 2), live EEG (Phase 3)

### Status
Phase 1 in progress — building foundations, reading papers, learning the tools.

Follow along as we update this repo with notes, code, and results.
