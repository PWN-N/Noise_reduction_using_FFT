import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os


def analyze_audio():

    # ==========================================
    # FILE PATHS
    # ==========================================

    NOISY_FILE = "Input/noisy_speech.wav"
    NOISE_FILE = "Input/noise_reference.wav"
    OUTPUT_FILE = "Output/cleaned_speech.wav"

    os.makedirs("Output", exist_ok=True)
    os.makedirs("Plots", exist_ok=True)

    # ==========================================
    # LOAD NOISY AUDIO
    # ==========================================

    noisy_audio, fs = sf.read(NOISY_FILE)

    if noisy_audio.ndim > 1:
        noisy_audio = np.mean(
            noisy_audio,
            axis=1
        )

    # ==========================================
    # LOAD NOISE REFERENCE
    # ==========================================

    noise, noise_fs = sf.read(NOISE_FILE)

    if noise.ndim > 1:
        noise = np.mean(
            noise,
            axis=1
        )

    # ==========================================
    # CHECK SAMPLE RATE
    # ==========================================

    if fs != noise_fs:
        raise ValueError(
            "Sample rates are different."
        )

    # ==========================================
    # MATCH LENGTH
    # ==========================================

    length = min(
        len(noisy_audio),
        len(noise)
    )

    noisy_audio = noisy_audio[:length]
    noise = noise[:length]

    # ==========================================
    # INFORMATION
    # ==========================================

    duration = length / fs

    print("==========================================")
    print("      FFT BASED AUDIO NOISE REDUCTION")
    print("==========================================")

    print("Sample Rate :", fs, "Hz")
    print("Samples     :", length)
    print("Duration    :", round(duration, 2), "seconds")

    # ==========================================
    # TIME AXIS
    # ==========================================

    time = np.arange(length) / fs

    # ==========================================
    # FFT OF NOISY AUDIO
    # ==========================================

    print("\nPerforming FFT...")

    noisy_fft = np.fft.fft(
        noisy_audio
    )

    # ==========================================
    # FFT OF NOISE
    # ==========================================

    noise_fft = np.fft.fft(
        noise
    )

    # ==========================================
    # FREQUENCY DOMAIN NOISE CANCELLATION
    # ==========================================

    print("Removing noise in frequency domain...")

    cleaned_fft = (
        noisy_fft - noise_fft
    )

    # ==========================================
    # IFFT
    # ==========================================

    print("Performing IFFT...")

    cleaned_audio = np.fft.ifft(
        cleaned_fft
    )

    cleaned_audio = np.real(
        cleaned_audio
    )

    # ==========================================
    # NORMALIZE OUTPUT
    # ==========================================

    max_output = np.max(
        np.abs(cleaned_audio)
    )

    if max_output > 1:

        cleaned_audio = (
            cleaned_audio / max_output
        )

    # ==========================================
    # SAVE CLEANED AUDIO
    # ==========================================

    sf.write(
        OUTPUT_FILE,
        cleaned_audio,
        fs
    )

    print("\nCleaned audio saved:")
    print(OUTPUT_FILE)

    # ==========================================
    # FFT FOR COMPARISON
    # ==========================================

    noisy_fft = np.fft.fft(
        noisy_audio
    )

    cleaned_fft = np.fft.fft(
        cleaned_audio
    )

    noisy_magnitude = np.abs(
        noisy_fft
    )

    cleaned_magnitude = np.abs(
        cleaned_fft
    )

    # ==========================================
    # FREQUENCIES
    # ==========================================

    frequencies = np.fft.fftfreq(
        length,
        d=1 / fs
    )

    half = length // 2

    positive_frequencies = frequencies[:half]

    # ==========================================
    # WAVEFORM COMPARISON
    # ==========================================

    print("\nGenerating waveform graph...")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)

    plt.plot(
        time,
        noisy_audio
    )

    plt.title(
        "Noisy Speech - Before Noise Reduction"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 1, 2)

    plt.plot(
        time,
        cleaned_audio
    )

    plt.title(
        "Speech After FFT Noise Reduction"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "Plots/waveform_comparison.png"
    )

    plt.show()

    # ==========================================
    # FREQUENCY SPECTRUM
    # ==========================================

    print("Generating frequency spectrum...")

    plt.figure(figsize=(12, 8))

    # Before
    plt.subplot(2, 1, 1)

    plt.plot(
        positive_frequencies,
        noisy_magnitude[:half]
    )

    plt.xlim(0, 8000)

    plt.title(
        "Frequency Spectrum - Before Filtering"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.grid()

    # After
    plt.subplot(2, 1, 2)

    plt.plot(
        positive_frequencies,
        cleaned_magnitude[:half]
    )

    plt.xlim(0, 8000)

    plt.title(
        "Frequency Spectrum - After Filtering"
    )

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "Plots/spectrum_comparison.png"
    )

    plt.show()

    # ==========================================
    # COMPLETION
    # ==========================================

    print("\n==========================================")
    print("          PROCESS COMPLETED")
    print("==========================================")

    print("\nInput:")
    print(NOISY_FILE)

    print("\nNoise reference:")
    print(NOISE_FILE)

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\nGraphs:")
    print("Plots/waveform_comparison.png")
    print("Plots/spectrum_comparison.png")
# RUN

if __name__ == "__main__":
    analyze_audio() 