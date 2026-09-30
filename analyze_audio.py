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


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":
    analyze_audio()




# import numpy as np
# import soundfile as sf
# import matplotlib.pyplot as plt
# import os


# def analyze_audio():

#     # ==========================================
#     # FILE PATHS
#     # ==========================================

#     INPUT_FILE = "Input/clean_speech.wav"
#     OUTPUT_FILE = "Output/cleaned_speech.wav"

#     os.makedirs("Output", exist_ok=True)
#     os.makedirs("Plots", exist_ok=True)

#     # ==========================================
#     # LOAD CLEAN SPEECH
#     # ==========================================

#     print("Loading clean speech...")

#     speech, fs = sf.read(INPUT_FILE)

#     # Convert stereo to mono
#     if speech.ndim > 1:
#         speech = np.mean(speech, axis=1)

#     length = len(speech)

#     # Time axis
#     time = np.arange(length) / fs

#     print("Sample Rate :", fs, "Hz")
#     print("Duration    :", round(length / fs, 2), "seconds")

#     # ==========================================
#     # GENERATE ARTIFICIAL NOISE
#     # ==========================================

#     print("\nGenerating noise...")

#     # 50 Hz electrical hum
#     noise_50Hz = 0.15 * np.sin(
#         2 * np.pi * 50 * time
#     )

#     # White noise
#     np.random.seed(42)

#     white_noise = 0.03 * np.random.randn(
#         length
#     )

#     # 1 kHz interference
#     noise_1kHz = 0.08 * np.sin(
#         2 * np.pi * 1000 * time
#     )

#     # 5 kHz interference
#     noise_5kHz = 0.08 * np.sin(
#         2 * np.pi * 5000 * time
#     )

#     # Combine all noise
#     total_noise = (
#         noise_50Hz
#         + white_noise
#         + noise_1kHz
#         + noise_5kHz
#     )

#     # ==========================================
#     # CREATE NOISY SPEECH
#     # ==========================================

#     noisy_speech = speech + total_noise

#     # Prevent clipping
#     max_value = np.max(
#         np.abs(noisy_speech)
#     )

#     if max_value > 1:
#         noisy_speech = (
#             noisy_speech / max_value
#         )

#     # ==========================================
#     # FFT OF NOISY SPEECH
#     # ==========================================

#     print("\nPerforming FFT...")

#     noisy_fft = np.fft.fft(
#         noisy_speech
#     )

#     frequencies = np.fft.fftfreq(
#         length,
#         d=1 / fs
#     )

#     # ==========================================
#     # FFT-BASED NOISE REDUCTION
#     # ==========================================

#     print("Removing noise using FFT...")

#     filtered_fft = noisy_fft.copy()

#     # Remove 50 Hz
#     filtered_fft[
#         np.abs(frequencies - 50) < 5
#     ] = 0

#     filtered_fft[
#         np.abs(frequencies + 50) < 5
#     ] = 0

#     # Remove 1 kHz
#     filtered_fft[
#         np.abs(frequencies - 1000) < 10
#     ] = 0

#     filtered_fft[
#         np.abs(frequencies + 1000) < 10
#     ] = 0

#     # Remove 5 kHz
#     filtered_fft[
#         np.abs(frequencies - 5000) < 10
#     ] = 0

#     filtered_fft[
#         np.abs(frequencies + 5000) < 10
#     ] = 0

#     # ==========================================
#     # IFFT
#     # ==========================================

#     print("Performing IFFT...")

#     cleaned_speech = np.fft.ifft(
#         filtered_fft
#     )

#     cleaned_speech = np.real(
#         cleaned_speech
#     )

#     # ==========================================
#     # NORMALIZE
#     # ==========================================

#     max_cleaned = np.max(
#         np.abs(cleaned_speech)
#     )

#     if max_cleaned > 1:

#         cleaned_speech = (
#             cleaned_speech / max_cleaned
#         )

#     # ==========================================
#     # SAVE CLEANED AUDIO
#     # ==========================================

#     sf.write(
#         OUTPUT_FILE,
#         cleaned_speech,
#         fs,
#         subtype="PCM_16"
#     )

#     print("\nCleaned audio saved:")
#     print(OUTPUT_FILE)

#     # ==========================================
#     # WAVEFORM COMPARISON
#     # ==========================================

#     print("\nGenerating waveform graph...")

#     plt.figure(figsize=(12, 8))

#     plt.subplot(2, 1, 1)

#     plt.plot(
#         time,
#         noisy_speech
#     )

#     plt.title(
#         "Noisy Speech - Before Noise Reduction"
#     )

#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude")
#     plt.grid()

#     plt.subplot(2, 1, 2)

#     plt.plot(
#         time,
#         cleaned_speech
#     )

#     plt.title(
#         "Speech After FFT Noise Reduction"
#     )

#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude")
#     plt.grid()

#     plt.tight_layout()

#     plt.savefig(
#         "Plots/waveform_comparison.png"
#     )

#     plt.show()

#     # ==========================================
#     # FREQUENCY SPECTRUM
#     # ==========================================

#     print("Generating frequency spectrum...")

#     noisy_magnitude = np.abs(
#         noisy_fft
#     )

#     cleaned_fft = np.fft.fft(
#         cleaned_speech
#     )

#     cleaned_magnitude = np.abs(
#         cleaned_fft
#     )

#     half = length // 2

#     positive_frequencies = (
#         frequencies[:half]
#     )

#     plt.figure(figsize=(12, 8))

#     # Before filtering

#     plt.subplot(2, 1, 1)

#     plt.plot(
#         positive_frequencies,
#         noisy_magnitude[:half]
#     )

#     plt.xlim(0, 8000)

#     plt.title(
#         "Frequency Spectrum - Before Filtering"
#     )

#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.grid()

#     # After filtering

#     plt.subplot(2, 1, 2)

#     plt.plot(
#         positive_frequencies,
#         cleaned_magnitude[:half]
#     )

#     plt.xlim(0, 8000)

#     plt.title(
#         "Frequency Spectrum - After FFT Filtering"
#     )

#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.grid()

#     plt.tight_layout()

#     plt.savefig(
#         "Plots/spectrum_comparison.png"
#     )

#     plt.show()

#     # ==========================================
#     # SNR CALCULATION
#     # ==========================================

#     original_power = np.mean(
#         speech ** 2
#     )

#     noise_power_before = np.mean(
#         (noisy_speech - speech) ** 2
#     )

#     noise_power_after = np.mean(
#         (cleaned_speech - speech) ** 2
#     )

#     snr_before = 10 * np.log10(
#         original_power /
#         noise_power_before
#     )

#     snr_after = 10 * np.log10(
#         original_power /
#         noise_power_after
#     )

#     print("\n==========================================")
#     print("              SNR RESULTS")
#     print("==========================================")

#     print(
#         "SNR Before Filtering :",
#         round(snr_before, 2),
#         "dB"
#     )

#     print(
#         "SNR After Filtering  :",
#         round(snr_after, 2),
#         "dB"
#     )

#     print(
#         "SNR Improvement      :",
#         round(
#             snr_after - snr_before,
#             2
#         ),
#         "dB"
#     )

#     print("\n==========================================")
#     print("          PROCESS COMPLETED")
#     print("==========================================")


# # ==========================================
# # RUN PROGRAM
# # ==========================================

# if __name__ == "__main__":
#     analyze_audio()