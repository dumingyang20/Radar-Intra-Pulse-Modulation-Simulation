# coding=utf-8
"""generate LFM signal"""
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import pickle
import bispectrum


def generate_LFM(N, f0, Fs, T1, B, T, NOISE, SNR):
    """ parameters
    N: the number of pulses
    f0: carry frequency
    Fs: sampling frequency
    T: pulse width
    T1: sample time
    B: band width
    NOISE: True or False """

    h1 = []
    while np.size(h1) < N:
        h1.append(random.randint(1, 10000))

    h2 = []
    while np.size(h2) < N:
        h2.append(random.randint(1, 950))

    B_randm = np.zeros(N)
    T_randm = np.zeros(N)
    for i in range(1, N + 1):
        B_randm[i - 1] = B + h1[i - 1] * 50  # band width
        T_randm[i - 1] = 1/(1/T + 1/(h2[i - 1]*1000))  # pulse width

    if NOISE == 'False':
        assert SNR is None
        tt = []
        x1 = []
        y1 = []
        for j in range(1, N + 1):
            tt = np.arange(-T1 / 2, T1 / 2 - 1 / Fs, 1 / Fs)
            K = B_randm[j - 1] / T_randm[j - 1]
            x1.append(np.exp(1j * (2 * np.pi * f0 - np.pi * K * pow(tt, 2))))

    elif NOISE == 'True':
        assert SNR is not None
        tt = []
        x1 = []
        y1 = []
        for j in range(1, N + 1):
            tt = np.arange(-T1 / 2, T1 / 2 - 1 / Fs, 1 / Fs)
            K = B_randm[j - 1] / T_randm[j - 1]
            x = np.exp(1j * (2 * np.pi * f0 - np.pi * K * pow(tt, 2)))
            n = np.random.randn(np.size(x))  # N(0,1) noise data
            noise = n - np.mean(n)  # mean=0
            x_power = np.sum(pow(abs(x), 2)) / np.size(x)
            noise_variance = x_power / np.power(10, (SNR / 10))
            noise = (np.sqrt(noise_variance) / (np.sqrt(2) * np.std(noise))) * noise
            x1.append(x + noise * (1+1j))  # LFM signal with noise (both real and image)
            y1.append(noise)

    return x1, y1


# x1, y1 = generate_LFM(N=2000*27, f0=100e6, Fs=100e6, T1=10e-6, B=180e6, T=2e-5, NOISE='True', SNR=-5)
x1, y1 = generate_LFM(N=1, f0=100e6, Fs=100e6, T1=10e-6, B=180e6, T=2e-5, NOISE='False', SNR=None)

# time domain figures
# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot(x1[0].real)
# plt.title('The real part of a noisy signal')
# plt.subplot(2, 1, 2)
# plt.plot(x1[0].imag)
# plt.subplots_adjust(hspace=0.5)
# plt.title('The image part of a noisy signal')
# plt.tight_layout()
# plt.show()
