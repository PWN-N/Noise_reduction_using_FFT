import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 44100
DURATION = 5

INPUT_DEVICE = 1
OUTPUT_DEVICE = 4

print("Recording for 5 seconds...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    device=INPUT_DEVICE
)

sd.wait()

print("Recording complete.")

sf.write(
    "Output/raw_test.wav",
    audio,
    SAMPLE_RATE,
    subtype="PCM_16"
)

print("Playing original recording...")

sd.play(
    audio,
    SAMPLE_RATE,
    device=OUTPUT_DEVICE
)

sd.wait()

print("Done.")