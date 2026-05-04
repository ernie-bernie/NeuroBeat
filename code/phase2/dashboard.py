# dashboard.py
# NeuroBeat — Phase 2
# Real-time EEG brain state dashboard with calibration
# Written by Evyn Ernest

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
import pickle
import os
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

mne.set_log_level('ERROR')

st.set_page_config(page_title="NeuroBeat", page_icon="🧠", layout="wide")
st.title("🧠 NeuroBeat")
st.caption("EEG Brain State Monitor — Music & Anxiety Research")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MUSIC_DIR = os.path.join(BASE_DIR, "music")

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("Controls")
participant = st.sidebar.selectbox(
    "Participant",
    options=list(range(1, 33)),
    format_func=lambda x: f"Participant {x:02d}"
)
sfreq = 128

# -------------------------------
# Load preprocessed data
# -------------------------------
@st.cache_data
def load_participant(p):
    path = os.path.join(DATA_DIR, f"s{p:02d}.dat")
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data["data"], data["labels"]

eeg_data, labels = load_participant(participant)

# -------------------------------
# Feature extraction
# -------------------------------
def get_alpha_beta_ratio(frequencies, power):
    alpha_indices = np.where((frequencies >= 8) & (frequencies <= 13))[0]
    beta_indices = np.where((frequencies >= 13) & (frequencies <= 30))[0]
    return np.mean(power[alpha_indices]) / np.mean(power[beta_indices])

def extract_features_from_channels(eeg, sfreq, channels):
    features = []
    for i in range(eeg.shape[0]):
        trial_features = []
        for p in channels:
            channel = eeg[i, p, :]
            freqs, power = welch(channel, fs=sfreq, nperseg=256)
            alpha = np.mean(power[(freqs >= 8) & (freqs <= 13)])
            beta = np.mean(power[(freqs >= 13) & (freqs <= 30)])
            theta = np.mean(power[(freqs >= 4) & (freqs <= 8)])
            trial_features.append(alpha / (beta + 1e-8))
            trial_features.append(theta / (beta + 1e-8))
            trial_features.append(alpha)
            trial_features.append((alpha + theta) / (beta + 1e-8))
        features.append(trial_features)
    return np.array(features)

FEATURE_SETS = {
    "Frontal (F3, Fz, F4)":   [2, 18, 19],
    "Temporal (T7, T8)":       [7, 25],
    "Occipital (O1, Oz, O2)":  [13, 14, 31],
    "All combined":            [2, 7, 13, 14, 18, 19, 25, 31],
}

REGION_CHANNELS = {
    "Frontal (F3, Fz, F4)":   "Frontal lobe — emotional regulation",
    "Temporal (T7, T8)":       "Temporal lobe — auditory processing",
    "Occipital (O1, Oz, O2)":  "Occipital lobe — visual and alpha source",
    "All combined":            "Whole brain — combined regions",
}

def run_calibration(eeg, labels, sfreq):
    indices = np.random.RandomState(42).permutation(40)
    eeg = eeg[indices]
    labels = labels[indices]
    y_all = labels[:, 0]
    y_binary = np.where(y_all >= np.median(y_all), 1, 0)

    best_score = 0
    best_name = None
    results = {}

    for name, channels in FEATURE_SETS.items():
        X = extract_features_from_channels(eeg, sfreq, channels)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X, y_binary, cv=4)
        mean_score = scores.mean()
        results[name] = mean_score
        if mean_score >= 0.70:
            best_score = mean_score
            best_name = name
            break
        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    return best_name, best_score, results

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2 = st.tabs(["📊 Dashboard", "🔧 Calibration"])

# -------------------------------
# Tab 1 — Dashboard
# -------------------------------
with tab1:
    trial_idx = st.slider("Select trial", 0, 39, 0)
    trial_eeg = eeg_data[trial_idx]
    actual_valence = labels[trial_idx][0]
    actual_arousal = labels[trial_idx][1]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Raw EEG Signal")
        channel_data = trial_eeg[18, :]  # Fz
        time = np.linspace(0, len(channel_data) / sfreq, len(channel_data))
        ten_sec = int(10 * sfreq)
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(time[:ten_sec], channel_data[:ten_sec],
                 linewidth=0.8, color="steelblue")
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.set_title(f"Channel Fz — Trial {trial_idx + 1}")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col2:
        st.subheader("Power Spectrum")
        freqs, power = welch(channel_data, fs=sfreq, nperseg=256)
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

    st.subheader("Brain Activity Map")

    band_name = st.selectbox("Band", ["Alpha", "Beta", "Theta"])
    band_ranges = {"Alpha": (8, 13), "Beta": (13, 30), "Theta": (4, 8)}
    band_low, band_high = band_ranges[band_name]

    known_channels = ['Fp1','AF3','F3','F7','FC5','FC1','C3','T7','CP5',
                      'CP1','P3','P7','PO3','O1','Oz','Pz','Fp2','AF4',
                      'Fz','F4','F8','FC6','FC2','Cz','C4','T8','CP6',
                      'CP2','P4','P8','PO4','O2']

    @st.cache_data
    def compute_band_map(p, trial, low, high):
        eeg, _ = load_participant(p)
        trial_data = eeg[trial]
        band_powers = []
        for ch in range(32):
            freqs, psd = welch(trial_data[ch], fs=128, nperseg=256)
            band_powers.append(np.mean(psd[(freqs >= low) & (freqs <= high)]))
        info = mne.create_info(known_channels, sfreq=128, ch_types='eeg')
        info.set_montage('standard_1020', on_missing='ignore')
        return np.array(band_powers), info

    band_powers, info = compute_band_map(participant, trial_idx, band_low, band_high)

    col3, col4, col5 = st.columns([1, 2, 1])
    with col4:
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        im, _ = mne.viz.plot_topomap(
            band_powers, info, axes=ax3, show=False,
            cmap='RdYlGn',
            vlim=(np.min(band_powers), np.max(band_powers)),
            extrapolate='head', sphere='eeglab',
            contours=0, sensors=False, outlines='head'
        )
        plt.colorbar(im, ax=ax3, shrink=0.7, label=f'{band_name} power')
        ax3.set_title(f"{band_name} power — trial {trial_idx + 1}")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    st.subheader("Brain State")
    alpha_idx = (freqs >= 8) & (freqs <= 13)
    beta_idx = (freqs >= 13) & (freqs <= 30)
    ratio = power[alpha_idx].mean() / power[beta_idx].mean()

    col6, col7, col8, col9 = st.columns(4)
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
        st.metric("Actual Valence", f"{actual_valence:.2f}")
    with col9:
        st.metric("Actual Arousal", f"{actual_arousal:.2f}")

    if "calibration_result" in st.session_state:
        best_name = st.session_state["calibration_result"]["best_name"]
        best_score = st.session_state["calibration_result"]["best_score"]
        channels = FEATURE_SETS[best_name]
        X_trial = extract_features_from_channels(
            eeg_data[trial_idx:trial_idx+1], sfreq, channels)
        y_all = labels[:, 0]
        y_binary = np.where(y_all >= np.median(y_all), 1, 0)
        X_all = extract_features_from_channels(eeg_data, sfreq, channels)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_all, y_binary)
        predicted = model.predict(X_trial)[0]
        st.info(f"Calibrated model predicts: {'😌 Positive valence' if predicted == 1 else '😔 Negative valence'} "
                f"— using {best_name} (accuracy: {best_score:.0%})")

    st.subheader("Music Recommendation")
    if ratio < 1.0:
        music_file = os.path.join(MUSIC_DIR, "Binaural_beats.mp3")
        music_type = "Binaural Beats"
        reasoning = ("Alpha/beta ratio below 1.0 indicates an anxious brain state. "
                     "Binaural beats have been shown to reduce anxiety symptoms "
                     "better than silence (Paper 6).")
    elif ratio < 2.0:
        music_file = os.path.join(MUSIC_DIR, "Classical_clip.mp3")
        music_type = "Classical Music"
        reasoning = ("Neutral brain state detected. Classical music is associated "
                     "with relaxed EEG and measurable alpha increases (Paper 3).")
    elif ratio < 4.0:
        music_file = os.path.join(MUSIC_DIR, "Ambient_clip.mp3")
        music_type = "Ambient Music"
        reasoning = ("Calm brain state detected. Ambient music maintains relaxed "
                     "alpha state without overstimulation (Paper 4).")
    else:
        music_file = os.path.join(MUSIC_DIR, "Upbeat_clip.mp3")
        music_type = "Upbeat Music"
        reasoning = ("Very high alpha detected — possible understimulation. "
                     "Upbeat music maintains optimal alert but relaxed state.")

    col10, col11 = st.columns([1, 2])
    with col10:
        st.info(f"🎵 Now playing: **{music_type}**")
        if os.path.exists(music_file):
            with open(music_file, 'rb') as f:
                st.audio(f.read(), format='audio/mp3')
    with col11:
        st.markdown("**Why this music?**")
        st.write(reasoning)

# -------------------------------
# Tab 2 — Calibration
# -------------------------------
with tab2:
    st.subheader(f"Calibrate Participant {participant:02d}")
    st.write("Calibration tests four different brain region feature sets and "
             "selects the one that best predicts this participant's emotional "
             "response to music.")

    if st.button("Run Calibration"):
        with st.spinner("Running calibration — testing feature sets..."):
            progress = st.progress(0)
            feature_names = list(FEATURE_SETS.keys())
            results = {}
            best_score = 0
            best_name = None

            for idx, (name, channels) in enumerate(FEATURE_SETS.items()):
                progress.progress((idx + 1) / len(FEATURE_SETS))
                indices = np.random.RandomState(42).permutation(40)
                eeg_shuffled = eeg_data[indices]
                labels_shuffled = labels[indices]
                y_all = labels_shuffled[:, 0]
                y_binary = np.where(y_all >= np.median(y_all), 1, 0)
                X = extract_features_from_channels(eeg_shuffled, sfreq, channels)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scores = cross_val_score(model, X, y_binary, cv=4)
                mean_score = scores.mean()
                results[name] = mean_score
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name

            st.session_state["calibration_result"] = {
                "best_name": best_name,
                "best_score": best_score,
                "results": results
            }

        st.success(f"Calibration complete — best feature set: **{best_name}**")
        st.metric("Best accuracy", f"{best_score:.1%}")
        st.caption(REGION_CHANNELS[best_name])

        st.subheader("All feature set results")
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        names = list(results.keys())
        scores = list(results.values())
        colors = ["green" if s >= 0.70 else "orange" if s >= 0.50 else "red"
                  for s in scores]
        bars = ax4.barh(names, scores, color=colors)
        ax4.axvline(0.70, color="green", linestyle="--", alpha=0.7, label="70% threshold")
        ax4.axvline(0.50, color="gray", linestyle="--", alpha=0.5, label="Chance level")
        ax4.set_xlim(0, 1)
        ax4.set_xlabel("Cross-validation accuracy")
        ax4.set_title("Feature set comparison")
        ax4.legend()
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        if best_score >= 0.70:
            st.success("Model calibrated successfully. Predictions are available in the Dashboard tab.")
        elif best_score >= 0.50:
            st.warning("Model calibrated with moderate confidence.")
        else:
            st.error("Calibration needs improvement for this participant. "
                     "Frontal alpha features may not be predictive for this user.")
    else:
        st.info("Press the button above to calibrate this participant.")
        if "calibration_result" in st.session_state:
            st.write(f"Last calibration: **{st.session_state['calibration_result']['best_name']}** "
                     f"at {st.session_state['calibration_result']['best_score']:.1%}")  

    
# -------------------------------
# To run: python -m streamlit run code/phase2/dashboard.py
# To close: Ctrl+C in terminal
# -------------------------------
