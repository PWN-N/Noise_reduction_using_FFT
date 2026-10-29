# Audio Noise Reduction Using FFT

Python examples of reducing audio noise in the frequency domain. The repository includes an offline workflow that subtracts a supplied noise reference and a live-recording workflow that estimates background noise from the start of a recording.

## Requirements

- Python 3
- NumPy
- SoundFile
- Matplotlib
- SoundDevice (needed for live recording and playback)

Install the dependencies with:

```bash
python -m pip install numpy soundfile matplotlib sounddevice
```

Run commands from the repository root so the relative `Input`, `Output`, and `Plots` paths resolve correctly.

## Offline noise reduction

The offline workflow uses the included `Input/clean_speech.wav` to create a noisy sample and its noise reference, then subtracts that reference in the FFT domain.

1. Generate the example noisy audio:

   ```bash
   python Add_noise.py
   ```

   This writes `Input/noisy_speech.wav` and `Input/noise_reference.wav`. The generated noise combines 50 Hz hum, white noise, and 1 kHz and 5 kHz interference.

2. Reduce the noise and generate comparisons:

   ```bash
   python main.py
   ```

   The cleaned audio is saved to `Output/cleaned_speech.wav`; waveform and spectrum plots are saved under `Plots/`.

You can also run the processing script directly with `python analyze_audio.py`.

## Live microphone noise reduction

Run:

```bash
python live_noise_reduction.py
```

Keep the background noise on and stay quiet for the first two seconds so the program can estimate its spectrum. Press Enter to begin recording, speak after the calibration period, and press Enter again to stop. The recording and processed audio are saved under `Output/`, and a waveform comparison is saved under `Plots/`.

The live script currently selects input device `1` and output device `4`. Change `INPUT_DEVICE` and `OUTPUT_DEVICE` near the top of `live_noise_reduction.py` to match the devices available on your computer.

## Other files

- `test_record.py` records and plays a short audio sample to check the selected devices.
- `Input/` contains the example audio files used by the offline workflow.
- `Output/` and `Plots/` contain example outputs and generated results.
