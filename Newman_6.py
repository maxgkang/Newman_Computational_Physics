# Fourier Transform

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, fft

# Basics
# f(x) can be represented as a sum of cosines, or sines
# This then can be represented in complex variables using the Euler's Formula
# Typically, Fourier Transform can be only used in periodic functions
# If we are interested in a portion of non-periodic function over a finite interval from 0 to L, we can take that portion and repeat this
#%%

# Discrete Fourier Transform
# For many cases, getting the coefficients of the series analytically is not possile. Therefore, f(x) is also unsolvable in this case.
# We can instead calculate the Fourier coefficients numerically by using the Trapezoidal Rule

from numpy import zeros,loadtxt
from pylab import plot,xlim,show
from cmath import exp,pi

def dft(y):
    N = len(y)
    c = zeros(N//2+1,complex)
    for k in range(N//2+1):
        for n in range(N):
            c[k] += y[n]*exp(-2j*pi*k*n/N)
    return c

y = loadtxt("pitch.txt",float)
c = dft(y)
plot(abs(c))
xlim(0,500)
show()
#%%

import numpy as np
import matplotlib.pyplot as plt

# 1. Create a simple signal
sampling_rate = 1000  # 주파수
T = 1.0 / sampling_rate # 주기
t = np.linspace(0.0, 1.0, sampling_rate, endpoint=False)

# Signal: 50 Hz sine wave + 80 Hz sine wave
freq1 = 50
freq2 = 80 


# f(x) = sin(2\pi ft)
signal = np.sin(freq1 * 2.0 * np.pi * t) + 0.5 * np.sin(freq2 * 2.0 * np.pi * t)
fft_output = np.fft.fft(signal)

# Get the corresponding frequencies
n = len(signal)
frequencies = np.fft.fftfreq(n, d=T)


# Filter for positive frequencies only (for plotting)
# FFT returns symmetric data for real inputs; usually, we only need the positive half
mask = frequencies > 0
fft_magnitude = np.abs(fft_output[mask]) # 크기 구함
positive_freqs = frequencies[mask]

# 5. Plot the result
plt.figure(figsize=(10, 4)) # 가로 세로, 인치 빈공간 생성


plt.subplot(1, 2, 1) # 1행 2열 짜리 행렬에 1번 자리에 넣겠다
plt.plot(t[:100], signal[:100]) # 0.1초까지만
plt.title("Time Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Frequency Domain Plot (The DFT Result)
plt.subplot(1, 2, 2)
plt.plot(positive_freqs, fft_magnitude)
plt.title("Frequency Domain (DFT Magnitude)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 150) # Zoom in on relevant frequencies
plt.grid(True) # grid 생성

plt.tight_layout()
plt.show()



#%%
# We can make the Fourier coefficient independent of the length of the wave(?)
# By Inverse Discrete Fourier Transform, you can get y_n from the Fourier coefficients, and vice versa

# If f(x) is real, we only have to calculate the half of the total coefficients
# DFT is independent of where we choose to place the samples
# DFT type I , DFT type II(takes the half points of type I)
# 2D DFT also mentioned

# Physical Interpretation of Fourier Transform: f(x) can be represented by waves of given frequencies
# Coefficients of the transform tell us exactly how much of each frequency we have in the sum

# k ~ f for x-axis, |c_k| for the y-axis, then, there will be some "white noise" which is random, on average contains equal amounts of frequencies
# Fourier Transform can break a signal down to its frequencies, which gives us alternative way of viewing it.

# Discrete cosine transforms
# in a finite interval, mirror this, which can be expressed in cosine sums
# This pethod does not have to equate the beginning to the end and doesn't have to be periodic

#%%


# Fast Fourier Transform
# It is a very fast way to get DFT

# start from the single Fourier transform of the entire set of samples
# SPlit the samples into two sets, then split those two into two to make four, continue

from scipy import fftpack
from numpy import fft
import matplotlib.pyplot as plt

# Inverse FFT (filtering noise)

# 1. Generate Signal with Noise
t = np.linspace(0, 1, 1000, endpoint=False)
clean_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz pure sine
noise = np.random.normal(0, 0.5, t.shape)
noisy_signal = clean_signal + noise

# 2. Compute FFT
fft_spectrum = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(t), d=1/1000)

# 3. Filter: Zero out frequencies with low magnitude (noise floor)
# We use a magnitude threshold to keep only the strong signal
magnitude = np.abs(fft_spectrum)
threshold = 100 
fft_spectrum[magnitude < threshold] = 0

# 4. Compute Inverse FFT to recover signal
filtered_signal = np.fft.ifft(fft_spectrum)

# 5. Plot Comparison
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t[:200], clean_signal[:200], color='green')
plt.title("Original Clean Signal")

plt.subplot(3, 1, 2)
plt.plot(t[:200], noisy_signal[:200], color='red', alpha=0.7)
plt.title("Noisy Signal")

plt.subplot(3, 1, 3)
plt.plot(t[:200], filtered_signal.real[:200], color='blue') # .real to discard complex rounding errors
plt.title("Recovered Signal (via Inverse FFT)")

plt.tight_layout()
plt.show()

#%%

# For larger datasets or scientific applications, 
# scipy.fft is often preferred over numpy as it offers more backend optimizations.

from scipy.fft import fft, fftfreq
import numpy as np

# Data setup
N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

# Calculation
yf = fft(y)
xf = fftfreq(N, T)[:N//2]

# Output peak frequencies
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()


# Algorithms of FFT: Cooley–Tukey algorithm is the most common method
#%%
# While numpy.fft is good for quick, general purpose,
# scipy.fft is the standard of scientific programming
# By speed, flexibility, capability, all of them are better in scipy
# use scipy's next_fast_len, which calculates the optimal padding size for your data to maximize processing speed

import numpy as np
import scipy.fft
import time

# Create a signal with a "bad" length (a large prime number)
N = 10007  # Prime number
t = np.linspace(0, 1, N)
signal = np.sin(2 * np.pi * 50 * t)

# --- The "Slow" Way (Direct FFT on prime length) ---
start = time.time()
scipy.fft.fft(signal)
print(f"Prime length ({N}) execution: {(time.time() - start)*1000:.4f} ms")

# --- The "Fast" Way (Zero-padding using next_fast_len) ---
# Find the optimal length (usually slightly larger than N)
fast_len = scipy.fft.next_fast_len(N)
print(f"Next fast length: {fast_len}")

start = time.time()
# Perform FFT with zero-padding to the fast length
fft_fast = scipy.fft.fft(signal, n=fast_len) 
print(f"Optimized length execution: {(time.time() - start)*1000:.4f} ms")

#%%

# %%

# scipy can do discrete cosine transforms, which numpy cannot do
# FFT assumes your signal is periodic (repeats forever). If the start and end of your signal don't match, FFT introduces artifacts (spectral leakage).
# DCT implies even symmetry (mirroring), which often fits real-world signals (like images or audio frames) much better, concentrating energy into fewer coefficients.

from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a signal that doesn't start/end at zero (bad for FFT)
N = 100
t = np.linspace(0, 1, N)
y = 2 * t + 0.5 * np.cos(2 * np.pi * 10 * t)  # Linear trend + oscillation

# 2. Compute DCT (Type II is the standard)
y_dct = dct(y, type=2, norm='ortho')

# 3. Filter: Zero out small coefficients (Compression)
threshold = 0.5
y_dct_filtered = y_dct.copy()
y_dct_filtered[np.abs(y_dct) < threshold] = 0

# 4. Reconstruct
y_reconstructed = idct(y_dct_filtered, type=2, norm='ortho')

# 5. Plot
plt.figure(figsize=(10, 5))
plt.plot(t, y, 'k', label='Original Signal', linewidth=2)
plt.plot(t, y_reconstructed, 'r--', label='DCT Compressed')
plt.legend()
plt.title("DCT filtering on Non-Periodic Data")
plt.show()

# The computer, unlike humans, cannot do continuous Fourier Transforms
# and at the end of the day, it has to chop each intervals
# and it wouldn't be different from discrete Fourier Transform