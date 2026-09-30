import numpy as np
import soundfile as sf
import os

# ==========================================
# LOAD CLEAN SPEECH
# ==========================================

speech, fs = sf.read("Input/clean_speech.wav")

# Convert stereo to mono
if speech.ndim > 1:
    speech = np.mean(speech, axis=1)

# Time axis
t = np.arange(len(speech)) / fs


# ==========================================
# GENERATE NOISE
# ==========================================

# 50 Hz electrical hum
noise_50Hz = 0.15 * np.sin(
    2 * np.pi * 50 * t
)

# White noise
np.random.seed(42)

white_noise = 0.03 * np.random.randn(
    len(speech)
)

# 1 kHz interference
noise_1kHz = 0.08 * np.sin(
    2 * np.pi * 1000 * t
)

# 5 kHz interference
noise_5kHz = 0.08 * np.sin(
    2 * np.pi * 5000 * t
)


# ==========================================
# COMBINE NOISE
# ==========================================

total_noise = (
    noise_50Hz
    + white_noise
    + noise_1kHz
    + noise_5kHz
)


# ==========================================
# CREATE NOISY AUDIO
# ==========================================

noisy_speech = speech + total_noise


# ==========================================
# USE SAME SCALE FOR BOTH SIGNALS
# ==========================================

max_value = max(
    np.max(np.abs(noisy_speech)),
    np.max(np.abs(speech)),
    np.max(np.abs(total_noise))
)

if max_value > 1:

    noisy_speech = noisy_speech / max_value
    total_noise = total_noise / max_value


# ==========================================
# CREATE FOLDER
# ==========================================

os.makedirs("Input", exist_ok=True)


# ==========================================
# SAVE FILES
# ==========================================

sf.write(
    "Input/noisy_speech.wav",
    noisy_speech,
    fs
)

sf.write(
    "Input/noise_reference.wav",
    total_noise,
    fs
)


# ==========================================
# INFORMATION
# ==========================================

print("====================================")
print("       NOISE ADDED SUCCESSFULLY")
print("====================================")

print("\nAdded noises:")
print("1. 50 Hz electrical noise")
print("2. White noise")
print("3. 1 kHz interference")
print("4. 5 kHz interference")

print("\nGenerated files:")
print("Input/noisy_speech.wav")
print("Input/noise_reference.wav")