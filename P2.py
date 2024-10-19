"""generate P2 signal"""
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
    fai2 = []
    for i in range(1, int(np.sqrt(N) + 1)):
        for j in range(1, int(np.sqrt(N) + 1)):
            fai2.append(-np.pi / (2 * np.sqrt(N)) * (2 * i - 1 - np.sqrt(N)) * (2 * j - 1 - np.sqrt(N)))

    # 2. generate phase
    num_p = 5
    length = num_p * np.size(fai2, 0)

    p = []
    while np.size(p) < M:
        p.append(random.randint(1, 1000))

    fc = np.zeros(M)
    Fs = (sample + 1) / (length * t_width)
    for i in range(1, M+1):
        # fc[i - 1] = (1/6 + p[i - 1]/6e3) * Fs  # band width: 1/6~1/3 * 5kHz
        fc[i - 1] = p[i - 1] * 1e3 + Fs

    h2 = []

    for i in range(0, int(num_p), 1):
        h2.append(fai2)

    h2 = np.array(h2)
    h2 = h2.flatten()

    tt = np.arange(1/Fs, length * t_width, 1/Fs)


    time_duration = np.round(np.size(tt, 0) / length)+1
    m2 = np.zeros(int(length * time_duration))

    for i in range(1, int(length), 1):
        m2[int((i - 1) * time_duration): int(i * time_duration)] = h2[i] * np.ones(int(time_duration))

    m2 = m2[0: np.size(tt, 0)]

    x_signal2 = []

    for s in range(1, M + 1):
        x2 = np.exp(1j * (2 * np.pi * fc[s - 1] * tt + m2))
        x2 = x2[0:sample]
        if Noise == 'True':
            # assert SNR < 0
            # generate noise
            noise = np.random.randn(np.size(x2))
            noise = noise - np.mean(noise)
            # signal power
            x_power2 = np.sum(pow(abs(x2), 2)) / np.size(x2)
            # noise power
            noise_variance2 = x_power2 / np.power(10, (SNR / 10))
            noise2 = (np.sqrt(noise_variance2) / (np.sqrt(2) * np.std(noise))) * noise

            # signal+noise
            x_signal2.append(x2 + noise2 * (1+1j))

        elif Noise == 'False':
            assert SNR is None
            x_signal2.append(x2)

    return x_signal2


x = generate_PCM(N=64, M=2000, t_width=1e-6, sample=1000, Noise='True', SNR=5)

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

# transfer IF data to image
# x = np.array(x[0])
# plt.imshow(x.real)
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.show()
