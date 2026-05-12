# live_pipeline.py
# NeuroBeat — Phase 3
# Synthetic live EEG pipeline — simulates real-time brain state detection
# Swap SYNTHETIC_BOARD for CERELOG_BOARD when hardware arrives
# Written by Evyn Ernest

import sys
import time
import numpy as np
sys.path.insert(0, r"C:\Users\evyne\Documents\Shared_brainflow-cerelog\python_package")

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations

# -------------------------------
# Channel mapping for Cerelog board
# Based on Phase 2 findings — frontal channels most predictive
# -------------------------------
CHANNEL_MAP = {
    "F3":  0,
    "Fz":  1,
    "F4":  2,
    "T7":  3,
    "T8":  4,
    "O1":  5,
    "Oz":  6,
    "O2":  7,
}

SFREQ = 250  # Cerelog sampling rate
WINDOW_SIZE = SFREQ * 4  # 4 seconds of data per analysis window

# -------------------------------
# Feature extraction
# Same functions as Phase 2 — works on any numpy array
# -------------------------------
def get_band_power(data, sfreq, low, high):
    nfft = min(len(data), 256)
    freqs = np.fft.rfftfreq(nfft, d=1.0/sfreq)
    fft_vals = np.abs(np.fft.rfft(data[:nfft])) ** 2
    band = (freqs >= low) & (freqs <= high)
    return np.mean(fft_vals[band]) if np.any(band) else 0

def extract_live_features(window, sfreq):
    features = []
    for ch_name, ch_idx in [("F3", 0), ("Fz", 1), ("F4", 2)]:
        channel = window[ch_idx]
        alpha = get_band_power(channel, sfreq, 8, 13)
        beta = get_band_power(channel, sfreq, 13, 30)
        theta = get_band_power(channel, sfreq, 4, 8)
        features.append(alpha / (beta + 1e-8))
        features.append(theta / (beta + 1e-8))
        features.append(alpha)
        features.append((alpha + theta) / (beta + 1e-8))
    return np.array(features)

def get_brain_state(ratio):
    if ratio < 1.0:
        return "anxious", "binaural beats"
    elif ratio < 2.0:
        return "neutral", "classical music"
    elif ratio < 4.0:
        return "calm", "ambient music"
    else:
        return "very calm", "upbeat music"

# -------------------------------
# Board setup
# Change BoardIds.SYNTHETIC_BOARD to your Cerelog board ID when hardware arrives
# -------------------------------
def setup_board():
    params = BrainFlowInputParams()
    # SYNTHETIC_BOARD = 0, simulates real EEG data
    # When Cerelog arrives: change to the Cerelog board ID from their docs
    board = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
    return board

# -------------------------------
# Main live loop
# Streams data, extracts features, predicts brain state every 4 seconds
# -------------------------------
def run_live(duration_seconds=30):
    print("NeuroBeat Phase 3 — Live Pipeline")
    print(f"Running synthetic stream for {duration_seconds} seconds")
    print("=" * 60)

    board = setup_board()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD)

    try:
        board.prepare_session()
        board.start_stream()
        print("Stream started — waiting for buffer to fill...")
        time.sleep(4)

        start_time = time.time()
        iteration = 0

        while time.time() - start_time < duration_seconds:
            # Get latest data from board
            data = board.get_current_board_data(WINDOW_SIZE)

            if data.shape[1] < WINDOW_SIZE:
                time.sleep(1)
                continue

            # Extract EEG channels only
            eeg_data = data[eeg_channels[:8], :]

            # Apply bandpass filter to each channel
            for ch in range(eeg_data.shape[0]):
                DataFilter.detrend(eeg_data[ch], DetrendOperations.CONSTANT)
                DataFilter.perform_bandpass(
                    eeg_data[ch], SFREQ, 4.0, 40.0, 4,
                    FilterTypes.BUTTERWORTH, 0
                )

            # Extract features from frontal channels
            features = extract_live_features(eeg_data, SFREQ)

            # Compute alpha/beta ratio from Fz (channel index 1)
            fz = eeg_data[1]
            alpha = get_band_power(fz, SFREQ, 8, 13)
            beta = get_band_power(fz, SFREQ, 13, 30)
            ratio = alpha / (beta + 1e-8)

            state, music = get_brain_state(ratio)
            iteration += 1

            print(f"\nWindow {iteration} — {time.time() - start_time:.1f}s elapsed")
            print(f"Alpha/Beta ratio (Fz): {ratio:.3f}")
            print(f"Brain state: {state}")
            print(f"Music recommendation: {music}")
            print(f"Feature vector shape: {features.shape}")

            time.sleep(4)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        board.stop_stream()
        board.release_session()
        print("\nStream stopped. Session released.")

if __name__ == "__main__":
    run_live(duration_seconds=30)