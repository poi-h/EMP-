import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os

filename1 = '~/desktop/探针/50ch1.csv'
M = pd.read_csv(filename1, skiprows=5, header=None).values
dt = 1e-11
fs = 1 / dt
t = M[:, 0]

# 多组处理
for a in [1, 2]:  # 可根据需要修改通道

    # 消除衰减
    signal = M[:, a]

    # 电场转换
    E = signal

    # 绘制原始信号的时间序列图
    plt.figure()
    plt.plot(t, E)
    plt.title(f'Electric Field {a}')
    plt.xlabel('t(s)')
    plt.ylabel('E(V/m)')
    fn1 = f'E{a}.png'

    # FFT
    L = len(t)
    Y = np.fft.fft(E)
    P2 = np.abs(Y / L)  # 双边谱
    P1 = P2[:L // 2]  # 单边谱
    if L > 2:
        P1[1:-1] = 2 * P1[1:-1]  # 补偿能量
    f = fs * np.arange(0, L // 2) / L

    # 物理常数
    c = 3e8  # 光速 m/s
    G_dBi = 4  # 天线增益（dBi）
    G_linear = 10 ** (G_dBi / 10)

    # 频率范围
    mask = (f >= 300e6) & (f <= 6000e6)
    f_sel = f[mask]
    P1_sel = P1[mask]
    wavelengths = c / f_sel

    # 有效面积函数 A_e(f)
    A_e = G_linear * wavelengths ** 2 / (4 * np.pi)

    # 示例：信号功率谱密度 S(f)
    S_f = P1_sel * A_e

    plt.figure()
    plt.plot(f_sel, S_f, color='r')
    plt.xlim([0, 6e9])
    plt.title(f'Single-Sided Amplitude Spectrum {a}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|P1|')
    fn3 = f'fft{a}.png'

    # Wavelet
    wavename = 'cmor1-1'
    totalscal = 8192
    Fc = pywt.central_frequency(wavename)
    c_wav = 2 * Fc * totalscal
    scals = c_wav / np.arange(1, totalscal + 1)
    f_wav = pywt.scale2frequency(wavename, scals) / dt
    coefs, freqs = pywt.cwt(E, scals, wavename)
    plt.figure()
    plt.imshow(np.abs(coefs), aspect='auto', extent=[t.min(), t.max(), f_wav.max(), f_wav.min()])
    plt.colorbar()
    plt.title(f'Continuous Wavelet Transform {a}')
    plt.xlabel('t(s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([0, 6e9])

plt.show()