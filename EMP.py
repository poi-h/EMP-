import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import os

# 数据文件路径
filename1 = 'D:/hfh25/Desktop/DCI_R11/EMP/20250428/20250428005/20250428005_ALL.csv'
filename2 = 'D:/hfh25/Desktop/DCI_R11/EMP/data_pro/delay.xlsx'
folder_path = 'D:/hfh25/Desktop/DCI_R11/EMP/data_pro/20250508019'

# 读取数据
M = pd.read_csv(filename1, skiprows=14, header=None).values

# 延迟
tdelay0 = 198.646 - 45.08
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减
sreduce0 = 5
sreduce = pd.read_excel(filename2, header=None, usecols="B:G", skiprows=5, nrows=1).values.flatten()#skiprows随发次号修改

# 示波器采样率
fs = 12.5e9

# 时间分辨率
dt = 1 / fs

# 多组处理
for a in [6]:  # 只处理第5组数据

    # 消除延迟
    tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
    t = M[:, 3 * a - 3] - tdelay

    # 消除衰减
    signal = M[:, 3 * a - 2] * (10 ** ((sreduce[a-1] + sreduce0) / 20))

    # 电场转换
    E = signal / 0.053

    # 绘制原始信号的时间序列图
    plt.figure()
    plt.plot(t, E)
    plt.title(f'Electric Field {a}')
    plt.xlabel('t(s)')
    plt.ylabel('E(V/m)')
    fn1 = f'E{a}.png'

    # FFT
    L = len(t)
    Y = np.fft.fft(signal)
    P2 = np.abs(Y / L)  # 双边谱
    P1 = P2[:L // 2]  # 单边谱
    P1[1:-1] = 2 * P1[1:-1]  # 补偿能量
    f = fs * np.arange(0, L // 2) / L
    plt.figure()
    plt.plot(f, P1, color='r')
    plt.xlim([0, 6e9])
    plt.title(f'Single-Sided Amplitude Spectrum {a}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|P1|')
    fn3 = f'fft{a}.png'

    # Wavelet
    wavename = 'cmor1-1'
    totalscal = 8192
    Fc = pywt.central_frequency(wavename)
    c = 2 * Fc * totalscal
    scals = c / np.arange(1, totalscal + 1)
    f = pywt.scale2frequency(wavename, scals) / dt
    coefs, freqs = pywt.cwt(E, scals, wavename)
    plt.figure()
    plt.imshow(np.abs(coefs), aspect='auto', extent=[t.min(), t.max(), f.max(), f.min()])
    plt.colorbar()
    plt.title(f'Continuous Wavelet Transform {a}')
    plt.xlabel('t(s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim([0, 6e9])

    plt.show()