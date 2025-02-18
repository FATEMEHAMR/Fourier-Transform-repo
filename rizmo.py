import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = 1  # Amplitude
f0 = 100  # Fundamental frequency (Hz)
T = 1 / f0  # Period
t = np.linspace(0, 2*T, 10000)  # Time vector (high resolution)

# Generate square wave
square_wave = A * np.sign(np.sin(2 * np.pi * f0 * t))

# Fourier Series coefficients
n_max = 100000  # Number of harmonics (odd only)
frequencies = np.arange(1, n_max+1, 2)  # Odd harmonics
coefficients = (4 * A) / (np.pi * frequencies)  # Fourier coefficients

# Frequency spectrum
plt.figure(figsize=(10, 4))
plt.stem(frequencies * f0, coefficients, basefmt=" ")  # Removed use_line_collection
plt.title("Frequency Spectrum of Square Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 2000)  # Limit x-axis to show relevant harmonics
plt.grid()
plt.show()

# Reconstruct the square wave using Fourier Series
reconstructed_signal = np.zeros_like(t)
for n, cn in zip(frequencies, coefficients):
    reconstructed_signal += cn * np.sin(2 * np.pi * n * f0 * t)

# Plot original and reconstructed signals
plt.figure(figsize=(10, 4))
plt.plot(t, square_wave, label="Original Square Wave", linewidth=2)
plt.plot(t, reconstructed_signal, label="Reconstructed Signal", linestyle="--", linewidth=1.5)
plt.title("Original vs Reconstructed Square Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()