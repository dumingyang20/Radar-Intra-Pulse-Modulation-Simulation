# coding=utf-8
"""generate NLFM(SFM) signal"""
import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.linalg import matrix_rank
import pickle


def generate_NLFM(N, f0, Fs, T1, mf, fm, NOISE, SNR):
    """
    :param N: the number of pulses
    :param f0: carry frequency
    :param Fs: sampling frequency
    :param T1: sample time
    :param mf: frequency modulation index
    :param fm: frequency shift
    :param NOISE:
    :param SNR:
    :return:
    when mf >> 1, B = 2*delta_f, delta_f = mf * fm
    """

    h = []
    while np.size(h) < N:
        h.append(random.uniform(1, 900))

    fm1 = np.zeros(N)

    if NOISE == 'False':
        assert SNR is None
        tt = []
        y1 = []
        x1 = []
        for j in range(1, N + 1):
            tt = np.arange(-T1 / 2, T1 / 2 - 1 / Fs, 1 / Fs)
            fm1[j - 1] = fm + h[j - 1] * 10000
            x1.append(np.exp(1j*(2*np.pi*f0*tt-mf*np.cos(2*np.pi*fm1[j-1]*tt))))

    elif NOISE == 'True':
        assert SNR is not None
        # assert SNR < 0
        tt = []
        y1 = []
        x1 = []
        for j in range(1, N + 1):
            fm1[j - 1] = fm + h[j - 1] / 50
            tt = np.arange(-T1 / 2, T1 / 2 - 1 / Fs, 1 / Fs)
            x = np.exp(1j*(2*np.pi*f0*tt-mf*np.cos(2*np.pi*fm1[j-1]*tt)))
            n = np.random.randn(np.size(x))  # N(0,1) noise data
            noise = n - np.mean(n)  # mean=0
            x_power = np.sum(pow(abs(x), 2)) / np.size(x)
            noise_variance = x_power / np.power(10, (SNR / 10))
            noise = (np.sqrt(noise_variance) / (np.sqrt(2) * np.std(noise))) * noise
            x1.append(x + noise * (1+1j))  # NLFM signal with noise (real and image)
            y1.append(noise)

    return x1, y1


x1, y1 = generate_NLFM(N=2000, f0=100e6, Fs=100e6, T1=10e-6, mf=10, fm=9e6, NOISE='True', SNR=10)
# x2 = np.array(x1)
# r = matrix_rank(x2.real)

# time domain
# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot(x1[0].real)
# plt.title('real')
# plt.subplot(2, 1, 2)
# plt.plot(x1[0].imag)
# plt.subplots_adjust(hspace=0.5)
# plt.title('image')
# # plt.subplot(3, 1, 3)
# # plt.plot(y1[0])
# plt.show()
