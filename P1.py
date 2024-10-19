"""generate P1 signal"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


def generate_PCM(N, M, t_width, sample, Noise, SNR):
    """parameter
    N: code bits
    M: the number of pulses
    Fs: band width
    """
    # 1. encode
    fai1 = []  # P1 code
    for j in range(1, int(np.sqrt(N) + 1)):
        for i in range(1, int(np.sqrt(N) + 1)):
            fai1.append(-np.pi / np.sqrt(N) * (np.sqrt(N) - (2 * j - 1)) * ((j - 1) * np.sqrt(N) + i - 1))

    # 2. generate phase
    num_p = 5
    length = num_p * np.size(fai1, 0)

    p = []
    while np.size(p) < M:
        p.append(random.randint(1, 1000))

    fc = np.zeros(M)
    Fs = (sample + 1) / (length * t_width)
    for i in range(1, M+1):
        # fc[i - 1] = (1/6 + p[i - 1]/6e3) * Fs  : 1/6~1/3 * 5kHz
        fc[i - 1] = p[i - 1] * 1e3 + Fs  # band width


    h1 = []
    for i in range(0, int(num_p), 1):
        h1.append(fai1)

    h1 = np.array(h1)
    h1 = h1.flatten()

    tt = np.arange(1/Fs, length * t_width, 1/Fs)

    time_duration = np.round(np.size(tt, 0) / length)+1
    m1 = np.zeros(int(length * time_duration))
    for k in range(1, int(length), 1):
        m1[int((k - 1) * time_duration): int(k * time_duration)] = h1[k] * np.ones(int(time_duration))

    m1 = m1[0: np.size(tt, 0)]

    # P1 signal
    x_signal1 = []

    for s in range(1, M + 1):
        x1 = np.exp(1j * (2 * np.pi * fc[s - 1] * tt + m1))
        x1 = x1[0:sample]
        if Noise == 'True':
            # assert SNR < 0
            # generate noise
            noise = np.random.randn(np.size(x1))
            noise = noise - np.mean(noise)
            # signal power
            x_power1 = np.sum(pow(abs(x1), 2)) / np.size(x1)

            # noise power
            noise_variance1 = x_power1 / np.power(10, (SNR / 10))
            noise1 = (np.sqrt(noise_variance1) / (np.sqrt(2) * np.std(noise))) * noise

            # signal+noise
            x_signal1.append(x1 + noise1 * (1+1j))

        elif Noise == 'False':
            assert SNR is None
            x_signal1.append(x1)

    return x_signal1


# x = generate_PCM(N=64, M=2000, t_width=1/400, Fs=2e3, Noise='True', SNR=-10)
x = generate_PCM(N=64, M=2000, t_width=2e-6, sample=1000, Noise='True', SNR=5)

# phase image
# plt.figure(1)
# plt.plot(fai[0:63])
# plt.title('64 bit P2 code Phase')
# plt.ylabel('Phase')
# plt.xlabel('Time [sec]')
# plt.show()

# time domain
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(x[1].real)
plt.title('real')
plt.subplot(2, 1, 2)
plt.plot(x[1].imag)
plt.subplots_adjust(hspace=0.5)
plt.title('image')
plt.show()

# transfer IF data to image
# x = np.array(x[0])
# plt.imshow(x.real)
# plt.axis('off')
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.show()
