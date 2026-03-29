# Notes and Observations
## This file contains any findings that we think will be helpful. It is a running log, updated as we read. One entry per session, written in our own words

### Fourier Transform
- The Fourier Transform is a method to "take apart" a jumbled up signal into its specific sine wave components.
- Think of it as "unmixing" a bucket of paint
- Takes a signal in time, converts it to a signal in frequency
- Converts the x-axis from time to frequency, so instead of seeing a wave over time, you see which frequencies are strongest
- There is a function that I can use to do this, but understanding it will help me dive deeper into the logistics of the project
- In NeuroBeat, this is important since it will convert the reading from the EEG to band values such as delta, theta, alpha, beta, gamma.

### Frequency Bands
- Pulses between neurons start creating a rhythm, or a brainwave
- Measured in cycles, or the number of times the neurons are firing per second
- Lower amplitudes as they speed up
- 5 main types of these waves
- Higher freuency the wave, the more awake and alert you are
- **Delta**
  - Slowest of the 5 waves (0.5-4 Hz)
  - Relatively high amplitude
  - Linked with deep sleep
- **Theta**
  - 4-8 Hz
  - Linked with day-dreaming or medatiation
  - In a calm, relaxed state
- **Alpha**
  - 8-12 Hz
  - Common when you are awake, but relaxed
  - Example: When your eyes are closed
  - Strongest at the back of the brain
- **Beta**
  - 12-30 Hz
  - Higher frequency, lower amplitude
  - Seems to happen when you are awake and thinking about something
  - High beta is also associated with stress and anxiety, which is NeuroBeat's primary target signal
- **Gamma**
  - Smallest and fastest oscillations (30-100 Hz)
  - Deeply focused on something
  - Harder to measure, so we would need more precise EEGs
- Brain always has different frequecies, but certain ones are more dominant
- Different regions of the brain are more commonly linked with certain things
- People with Alzheimer's don't seem to use gamma waves as much

### Alpha With Anxiety
- Alpha waves are the ones which are most present in an awake and relaxed mood
- Even if the eyes are closed, different levels of anxiety can greatly impact if alpha waves are enhanced or supressed
- In a [study](https://github.com/ernie-bernie/NeuroBeat/blob/main/Phase%201/papers.md#paper-1), people who started with less anxiety were better at supressing but not enhancing alpha waves
- Final state anxiety was unrelated to alpha control
- It was concluded that anxiety may be related to the ability to reduce alpha waves
- High alpha = calm and relaxed
- Low alpha (suppressed) = anxious
- Enhanced alpha = above-baseline calm. The goal of NeuroBeat isn't just reducing anxiety, it's actively pushing toward an enhanced alpha state through music.
- The brain's threat-response system (amygdala, stress hormones) actively shifts processing toward higher-frequency, more vigilant states if you are anxious. Alpha is essentially incompatible with being on high alert
- A [paper](https://github.com/ernie-bernie/NeuroBeat/blob/main/Phase%201/papers.md#paper-2) showed that low delta and high alpha together are associated with better mental control in low-anxiety individuals.

### How EEG Works:
- Measures electricity in the brain and records it
- Uses electrodes on two dipoles (different charges in neutrons) to measure the difference of charge between them
- Single dipoles in neurons are too small to measure, but clusters of differently charged neurons can be used as dipoles
- Different ways to connect two electrodes
  - A common reference montage, shows the difference of each electrode to the same electrode
  - A bipolar montage, connecting each electrode to the one next to it
  - For anxiety research, frontal electrodes (Fz, F3, F4) in a common reference montage are most commonly used because the prefrontal cortex is heavily involved in emotional regulation
- EEG is used over brain imaging, since it is capturing data changes millisecond by millisecond, which is why it can track fast-changing mental states like anxiety responses to music in real time.












