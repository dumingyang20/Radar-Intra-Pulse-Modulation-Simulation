"""generate P4 signal"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


def generate_PCM(N, M, t_width, sample, Noise, SNR):
    """parameter
    N: code bits
    M: the number of pulses
    """
    # 1. encode
    fai4 = np.zeros(N)
    for n in range(1, N + 1):
        fai4[n - 1] = np.pi * pow(n - 1, 2) / N - np.pi * (n - 1)

    # 2. generate phase
    num_p = 5
    length = num_p * np.size(fai4, 0)

    p = []
    while np.size(p) < M:
        p.append(random.randint(1, 1000))

    fc = np.zeros(M)
    Fs = (sample + 1) / (length * t_width)
    for i in range(1, M+1):
        # fc[i - 1] = (1/6 + p[i - 1]/6e3) * Fs  # band width: 1/6~1/3 * 5kHz
        fc[i - 1] = p[i - 1] * 1e3 + Fs


    h4 = []
    for i in range(0, int(num_p), 1):
        h4.append(fai4)
    h4 = np.array(h4)
    h4 = h4.flatten()

    tt = np.arange(1/Fs, length * t_width, 1/Fs)


    time_duration = np.round(np.size(tt, 0) / length) + 1
    m4 = np.zeros(int(length * time_duration))
    for i in range(1, int(length), 1):
        m4[int((i - 1) * time_duration): int(i * time_duration)] = h4[i] * np.ones(int(time_duration))

    m4 = m4[0: np.size(tt, 0)]


    x_signal4 = []
    for s in range(1, M + 1):
        x4 = np.exp(1j * (2 * np.pi * fc[s - 1] * tt + m4))
        x4 = x4[0:sample]
        if Noise == 'True':
            # assert SNR < 0
            # generate noise
            noise = np.random.randn(np.size(x4))
            noise = noise - np.mean(noise)
            # signal power
            x_power4 = np.sum(pow(abs(x4), 2)) / np.size(x4)
            # noise power
            noise_variance4 = x_power4 / np.power(10, (SNR / 10))
            noise4 = (np.sqrt(noise_variance4) / (np.sqrt(2) * np.std(noise))) * noise
            # signal+noise
            x_signal4.append(x4 + noise4 * (1+1j))

        elif Noise == 'False':
            assert SNR is None
            x_signal4.append(x4)

    return x_signal4


x = generate_PCM(N=64, M=2000, t_width=2e-6, sample=1000, Noise='True', SNR=10)

# phase image
# plt.figure(1)
# plt.plot(fai[0:63])
# plt.title('64 bit P2 code Phase')
# plt.ylabel('Phase')
# plt.xlabel('Time [sec]')
# plt.show()

# time domain
# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.plot(x[0].real)
# plt.title('real')
# plt.subplot(2, 1, 2)
# plt.plot(x[0].imag)
# plt.subplots_adjust(hspace=0.5)
# plt.title('image')
# plt.show()
