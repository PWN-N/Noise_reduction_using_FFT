import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 44100

# Your laptop microphone and speaker
INPUT_DEVICE = 1
OUTPUT_DEVICE = 4

CHANNELS = 1

# FFT parameters
FRAME_SIZE = 2048
HOP_SIZE = 1024

# First few seconds are used to estimate background noise
NOISE_DURATION = 2.0

# Spectral subtraction parameters
ALPHA = 1.5       # Noise over-subtraction factor
BETA = 0.02       # Spectral floor

# Output files
RAW_FILE = "Output/recorded_audio.wav"
CLEAN_FILE = "Output/noise_reduced_audio.wav"


# ============================================================
# RECORD AUDIO
# ============================================================

def record_audio():

    print("\n======================================")
    print("      FFT AUDIO NOISE REDUCTION")
    print("======================================")

    print("\nFirst, keep the background noise ON.")
    print("Do NOT speak during the first 2 seconds.")
    print("This period will be used to learn the noise.")

    input("\nPress ENTER to start recording...")

    audio_data = []

    print("\nRecording...")
    print("Speak normally after the first 2 seconds.")
    print("Press ENTER to stop recording.\n")

    def callback(indata, frames, time, status):

        if status:
            print(status)

        audio_data.append(indata[:, 0].copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=INPUT_DEVICE,
        callback=callback
    ):

        input()

    print("\nRecording stopped.")

    audio = np.concatenate(audio_data)

    # Prevent clipping
    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value * 0.95

    sf.write(RAW_FILE, audio, SAMPLE_RATE)

    print("Raw recording saved:")
    print(RAW_FILE)

    return audio


# ============================================================
# SPECTRAL SUBTRACTION
# ============================================================

def spectral_subtraction(audio):

    print("\nProcessing FFT noise reduction...")

    frame_size = FRAME_SIZE
    hop_size = HOP_SIZE

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    noise_samples = int(NOISE_DURATION * SAMPLE_RATE)

    if len(audio) < noise_samples:

        print("Recording is too short.")
        return audio

    # Number of frames
    num_frames = int(
        np.ceil((len(audio) - frame_size) / hop_size)
    ) + 1

    padded_length = (
        (num_frames - 1) * hop_size
        + frame_size
    )

    padded_audio = np.zeros(padded_length)

    padded_audio[:len(audio)] = audio

    # Hann window
    window = np.hanning(frame_size)

    # --------------------------------------------------------
    # Estimate noise spectrum
    # --------------------------------------------------------

    noise_audio = audio[:noise_samples]

    noise_power_sum = None
    noise_frame_count = 0

    for start in range(
        0,
        len(noise_audio) - frame_size + 1,
        hop_size
    ):

        frame = noise_audio[
            start:start + frame_size
        ]

        windowed = frame * window

        spectrum = np.fft.rfft(windowed)

        power = np.abs(spectrum) ** 2

        if noise_power_sum is None:
            noise_power_sum = power
        else:
            noise_power_sum += power

        noise_frame_count += 1

    if noise_frame_count == 0:

        print("Not enough noise samples.")
        return audio

    noise_power = (
        noise_power_sum / noise_frame_count
    )

    print("Noise spectrum estimated.")

    # --------------------------------------------------------
    # Output buffer
    # --------------------------------------------------------

    output = np.zeros(padded_length)

    window_sum = np.zeros(padded_length)

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    for i in range(num_frames):

        start = i * hop_size

        frame = padded_audio[
            start:start + frame_size
        ]

        windowed = frame * window

        # FFT
        spectrum = np.fft.rfft(windowed)

        magnitude = np.abs(spectrum)

        phase = np.angle(spectrum)

        power = magnitude ** 2

        # ----------------------------------------------------
        # Spectral subtraction
        # ----------------------------------------------------

        clean_power = (
            power - ALPHA * noise_power
        )

        # Prevent negative values
        clean_power = np.maximum(
            clean_power,
            BETA * power
        )

        clean_magnitude = np.sqrt(
            clean_power
        )

        # Restore phase
        clean_spectrum = (
            clean_magnitude *
            np.exp(1j * phase)
        )

        # IFFT
        clean_frame = np.fft.irfft(
            clean_spectrum,
            n=frame_size
        )

        # Window again
        clean_frame *= window

        # Overlap-add
        output[
            start:start + frame_size
        ] += clean_frame

        window_sum[
            start:start + frame_size
        ] += window ** 2

    # --------------------------------------------------------
    # Normalize overlap-add
    # --------------------------------------------------------

    window_sum[
        window_sum < 1e-8
    ] = 1.0

    output = output / window_sum

    # Remove padding
    output = output[:len(audio)]

    # --------------------------------------------------------
    # Normalize output
    # --------------------------------------------------------

    max_output = np.max(
        np.abs(output)
    )

    if max_output > 0:

        output = (
            output / max_output
        ) * 0.95

    print("FFT processing completed.")

    return output


# ============================================================
# PLAY AUDIO
# ============================================================

def play_audio(audio):

    print("\nPlaying noise-reduced audio...")

    sd.play(
        audio,
        SAMPLE_RATE,
        device=OUTPUT_DEVICE
    )

    sd.wait()

    print("Playback completed.")


# ============================================================
# PLOT WAVEFORMS
# ============================================================

def create_plot(original, cleaned):

    print("\nCreating waveform comparison...")

    time_original = (
        np.arange(len(original))
        / SAMPLE_RATE
    )

    time_cleaned = (
        np.arange(len(cleaned))
        / SAMPLE_RATE
    )

    plt.figure(figsize=(12, 7))

    plt.subplot(2, 1, 1)

    plt.plot(
        time_original,
        original
    )

    plt.title("Original Recorded Audio")

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid()

    plt.subplot(2, 1, 2)

    plt.plot(
        time_cleaned,
        cleaned
    )

    plt.title(
        "Noise Reduced Audio - FFT Spectral Subtraction"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        "Plots/waveform_comparison.png",
        dpi=300
    )

    plt.show()

    print(
        "Plot saved: "
        "Plots/waveform_comparison.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Step 1: Record
    audio = record_audio()

    # Step 2: FFT noise reduction
    cleaned_audio = spectral_subtraction(
        audio
    )

    # Step 3: Save cleaned audio
    sf.write(
        CLEAN_FILE,
        cleaned_audio,
        SAMPLE_RATE
    )

    print("\nNoise-reduced audio saved:")
    print(CLEAN_FILE)

    # Step 4: Plot
    create_plot(
        audio,
        cleaned_audio
    )

    # Step 5: Play
    play_audio(
        cleaned_audio
    )

    print("\n======================================")
    print("          PROCESS COMPLETED")
    print("======================================")


if __name__ == "__main__":
    main()