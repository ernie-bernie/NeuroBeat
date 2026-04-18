# dashboard.py
# NeuroBeat — Phase 2
# Real-time EEG brain state dashboard
# Written by Evyn Ernest

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
music_folder = os.path.join(BASE_DIR, "music")
st.write(f"Looking for music in: {music_folder}")
# Stops useless outputs
mne.set_log_level('ERROR')
# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="NeuroBeat",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NeuroBeat")
st.caption("EEG Brain State Monitor — Music & Anxiety Research")

# -------------------------------
# Sidebar — controls
# -------------------------------
st.sidebar.header("Controls")

#Pick the participant
participant = st.sidebar.selectbox(
    "Participant", 
    options=list(range(1, 33)),
    format_func=lambda x: f"Participant {x:02d}"
)

#Pick the channel
channel_name = st.sidebar.selectbox(
    "Channel",
    options=["Fz", "F3", "F4", "Cz", "Pz"]
)



#Pick the band
band_name = st.sidebar.selectbox(
    "Brain map band",
    options=["Alpha", "Beta", "Theta"]
)

band_ranges = {
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Theta": (4, 8)
}
band_low, band_high = band_ranges[band_name]

# -------------------------------
# Load and filter data
# -------------------------------
@st.cache_data
def load_participant(p):
    path = f"C:\\Users\\evyne\\Documents\\NeuroBeat\\NeuroBeat\\code\\phase2\\data\\s{p:02d}.bdf"
    raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    raw.pick_types(eeg=True, verbose=False)
    raw.filter(0.5, 100, fir_window='hamming', verbose=False)
    raw.notch_filter(50, verbose=False)
    return raw

raw = load_participant(participant)
st.sidebar.success(f"Loaded participant {participant:02d}")

data, times = raw.get_data(return_times=True)
ch_index = raw.ch_names.index(channel_name)
sfreq = raw.info['sfreq']

# -------------------------------
# Row 1 — EEG signal + power spectrum
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw EEG Signal")
    ten_sec = int(10 * sfreq)
    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(times[:ten_sec], data[ch_index, :ten_sec], 
             linewidth=0.8, color="steelblue")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Amplitude (µV)")
    ax1.set_title(f"Channel {channel_name}")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

with col2:
    st.subheader("Power Spectrum")
    sixty_sec = int(60 * sfreq)
    freqs, power = welch(data[ch_index, :sixty_sec], 
                         fs=sfreq, nperseg=512)
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(freqs, power, color="steelblue", linewidth=0.8)
    ax2.axvspan(0.5, 4,  alpha=0.15, color="purple", label="Delta")
    ax2.axvspan(4,   8,  alpha=0.15, color="blue",   label="Theta")
    ax2.axvspan(8,  13,  alpha=0.15, color="green",  label="Alpha")
    ax2.axvspan(13, 30,  alpha=0.15, color="orange", label="Beta")
    ax2.axvspan(30, 100, alpha=0.15, color="red",    label="Gamma")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.set_xlim(4, 40)
    ax2.set_ylim(0, power[(freqs >= 4) & (freqs <= 40)].max() * 1.2)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# -------------------------------
# Row 2 — Brain map
# -------------------------------
st.subheader("Brain Activity Map")
st.caption(f"{band_name} power across all electrodes")

def compute_band_map(participant, low, high):
    raw = load_participant(participant)
    known_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                      'T7', 'C3', 'Cz', 'C4', 'T8',
                      'P7', 'P3', 'Pz', 'P4', 'P8',
                      'O1', 'Oz', 'O2', 'FC1', 'FC2',
                      'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6',
                      'AF3', 'AF4', 'PO3', 'PO4']
    available = [ch for ch in known_channels if ch in raw.ch_names]
    raw.pick_channels(available)
    raw.set_montage('standard_1020', on_missing='ignore')
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    sixty_sec = int(60 * sfreq)
    band_powers = []
    for ch in range(len(raw.ch_names)):
        freqs, psd = welch(data[ch, :sixty_sec], fs=sfreq, nperseg=512)
        band = (freqs >= low) & (freqs <= high)
        band_powers.append(psd[band].mean())
    return np.array(band_powers), raw

band_powers, raw_topo = compute_band_map(participant, band_low, band_high)

col3, col4, col5 = st.columns([1, 2, 1])

with col4:
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    im, _ = mne.viz.plot_topomap(
        band_powers,
        raw_topo.info,
        axes=ax3,
        show=False,
        cmap='RdYlGn',
        vlim=(np.min(band_powers), np.max(band_powers)),
        extrapolate='head',
        sphere='eeglab',
        contours=0,        # removes the contour lines for a cleaner look
        sensors=False,     # removes the dots entirely
        outlines='head'    # keeps just the head outline
)
    plt.colorbar(im, ax=ax3, shrink=0.7, 
                 label=f'{band_name} power (µV²/Hz)')
    ax3.set_title(f"{band_name} power — "
                  f"{'higher = calmer' if band_name == 'Alpha' else 'higher = more active'}")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# -------------------------------
# Row 3 — Emotion prediction
# -------------------------------
st.subheader("Brain State")

# Compute alpha/beta ratio for selected channel
alpha_idx = (freqs >= 8) & (freqs <= 13)
beta_idx = (freqs >= 13) & (freqs <= 30)
ratio = power[alpha_idx].mean() / power[beta_idx].mean()

col6, col7, col8 = st.columns(3)

with col6:
    st.metric("Alpha/Beta Ratio", f"{ratio:.2f}")

with col7:
    if ratio > 2.0:
        st.success("😌 Calm")
    elif ratio > 1.0:
        st.warning("😐 Neutral")
    else:
        st.error("😰 Anxious")

with col8:
    st.metric("Alpha Power", f"{power[alpha_idx].mean():.2e}")


# -------------------------------
# Row 4 — Music recommendation
# -------------------------------
st.subheader("Music Recommendation")

# Define music logic based on brain state

if ratio < 1.0:
    music_file = os.path.join(music_folder, "Binaural_beats.mp3")
    music_type = "Binaural Beats"
    reasoning = (
        "Your alpha/beta ratio is below 1.0, indicating an anxious brain state. "
        "Binaural beats in the alpha frequency range (8-13 Hz) have been shown to "
        "reduce anxiety symptoms better than silence or noise-canceling headphones alone "
        "(Paper 6 — systematic review of 12 studies). The goal is to entrain your brain "
        "toward enhanced alpha activity."
    )
elif ratio < 2.0:
    music_file = os.path.join(music_folder, "Classical_clip.mp3")
    music_type = "Classical Music"
    reasoning = (
        "Your alpha/beta ratio is between 1.0 and 2.0, indicating a neutral brain state. "
        "Classical music has been associated with a relaxed EEG state and measurable "
        "increases in alpha band power compared to silence (Paper 3). "
        "The goal is to shift toward enhanced alpha dominance."
    )
elif ratio < 4.0:
    music_file = os.path.join(music_folder, "Ambient_clip.mp3")
    music_type = "Ambient Music"
    reasoning = (
        "Your alpha/beta ratio is between 2.0 and 4.0, indicating a calm brain state. "
        "Ambient music maintains this relaxed state without overstimulation. "
        "Research shows pleasant music sustains alpha and theta power in "
        "already-relaxed participants (Paper 4)."
    )
else:
    music_file = os.path.join(music_folder, "Upbeat_clip.mp3")
    music_type = "Upbeat Music"
    reasoning = (
        "Your alpha/beta ratio is above 4.0, indicating a very calm, possibly "
        "under-stimulated brain state. Upbeat music provides gentle stimulation "
        "to maintain engagement without triggering anxiety. The goal is to keep "
        "you in an optimal alert but relaxed state."
    )

# Display recommendation
col9, col10 = st.columns([1, 2])

with col9:
    st.info(f"🎵 Now playing: **{music_type}**")
    with open(music_file, 'rb') as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format='audio/mp3')

with col10:
    st.markdown("**Why this music?**")
    st.write(reasoning)
    st.caption(f"Alpha/Beta ratio: {ratio:.2f} — "
               f"{'Anxious' if ratio < 1.0 else 'Neutral' if ratio < 2.0 else 'Calm'} state detected")
    

    
# -------------------------------
# To run: python -m streamlit run code/phase2/dashboard.py
# To close: Ctrl+C in terminal
# -------------------------------
