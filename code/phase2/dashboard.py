# dashboard.py
# NeuroBeat — Phase 2
# Real-time EEG brain state dashboard
# Written by Evyn Ernest

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import welch
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
participant = st.sidebar.selectbox(
    "Participant", 
    options=list(range(1, 33)),
    format_func=lambda x: f"Participant {x:02d}"
)
channel_name = st.sidebar.selectbox(
    "Channel",
    options=["Fz", "F3", "F4", "Cz", "Pz"]
)

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
st.caption("Alpha power across all electrodes")

# @st.cache_data(show_spinner=False)
def compute_alpha_map(participant):
    raw = load_participant(participant)
    raw.pick_types(eeg=True)
    raw.set_montage('standard_1020', on_missing='ignore')
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    sixty_sec = int(60 * sfreq)
    
    alpha_powers = []
    for ch in range(len(raw.ch_names)):
        freqs, psd = welch(data[ch, :sixty_sec], fs=sfreq, nperseg=512)
        band = (freqs >= 8) & (freqs <= 13)
        alpha_powers.append(psd[band].mean())
    
    return alpha_powers, raw


alpha_powers, raw = compute_alpha_map(participant)

fig3, ax3 = plt.subplots(figsize=(6, 5))
mne.viz.plot_topomap(
    alpha_powers,
    raw.info,
    axes=ax3,
    show=False,
    cmap='RdYlGn',
    vlim=(np.min(alpha_powers), np.max(alpha_powers))
)
ax3.set_title("Alpha power distribution")
st.pyplot(fig3)
plt.close()



# -------------------------------
# To run: python -m streamlit run code/phase2/dashboard.py
# To close: Ctrl+C in terminal
# -------------------------------
