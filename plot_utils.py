# utils.py
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os
import matplotlib.ticker as ticker

def signal_plot(signal, t, a, shot_id, save_dir='.', xlim=(0, 6e-8), ylim=(-1e5, 1e5)):  
    """
    绘制时域信号并保存为图片。
    Plot the time-domain signal and save as an image.
    
    参数/Args:
        signal: 输入信号 (Input signal)
        t: 时间序列 (Time array)
        a: 标记名 (Label for file name and title)
        save_dir: 图片保存目录 (Directory to save the image)
    返回/Returns:
        图片完整路径 (Full path of saved image)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure()
    plt.plot(t, signal)
    plt.title(f'Electric Field {a}')
    plt.xlabel('t(s)')
    plt.ylabel('E(V/m)')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(xlim)
    plt.ylim(ylim)
    fn = os.path.join(save_dir, f'{shot_id:03d}E{a}.png')
    plt.savefig(fn, dpi=600)
    plt.close()
    return fn

def fft_plot(signal, fs, a, shot_id, save_dir='.', xlim=(0, 6e9)):
    """
    对信号 signal 做FFT分析并保存频谱图。
    Perform FFT analysis on the signal and save the spectrum image.

    参数/Args:
        signal: 输入信号 (Input signal)
        fs: 采样率 (Sampling frequency)
        a: 标记名 (Label for file name and title)
        save_dir: 图片保存目录 (Directory to save the image)
        xlim: x轴范围 (X-axis limit for frequency plot)
    返回/Returns:
        图片完整路径 (Full path of saved image)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    L = len(signal)
    Y = np.fft.fft(signal)
    P2 = np.abs(Y / L)  # 双边谱
    P1 = P2[:L // 2]  # 单边谱
    if L > 2:
        P1[1:-1] = 2 * P1[1:-1]  # 补偿能量
    f = fs * np.arange(0, L // 2) / L
    plt.figure()
    plt.plot(f, P1, color='r')
    plt.title(f'Single-Sided Amplitude Spectrum {a}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|P1|')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(xlim)
    fn = os.path.join(save_dir, f'{shot_id:03d}fft{a}.png')
    plt.savefig(fn, dpi=600)
    plt.close()
    return fn

def fft_plot_Ae(signal, fs, a, shot_id, save_dir='.', xlim=(0, 6e9), G_dBi=4, c=3e8):
    """
    对信号 signal 做FFT分析，使用有效面积A_e修正并保存频谱图。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    L = len(signal)
    Y = np.fft.fft(signal)
    P2 = np.abs(Y / L)  # 双边谱
    P1 = P2[:L // 2]    # 单边谱
    if L > 2:
        P1[1:-1] = 2 * P1[1:-1]  # 补偿能量
    f = fs * np.arange(0, L // 2) / L

    # 有效面积修正
    G_linear = 10 ** (G_dBi / 10)
    wavelengths = c / f
    A_e = G_linear * wavelengths ** 2 / (4 * np.pi)
    # 避免除以零
    # A_e[P1 == 0] = np.nan
    P1_Ae = P1 / A_e

    plt.figure()
    plt.plot(f, P1_Ae, color='g')
    plt.title(f' shot 044 FFT(A_e corrected) {a}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|P1| / A_e')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(xlim)
    fn = os.path.join(save_dir, f'{shot_id:03d}fft_Ae{a}.png')
    plt.savefig(fn, dpi=600)
    plt.close()
    return fn

def cwt_plot(signal, t, fs, a, shot_id, save_dir='.', totalscal=8192, wavename='cmor1-1', xlim=(0, 6e-8), ylim=(0, 6e9)):
    """
    对信号 signal 做连续小波变换并保存图像。
    Perform continuous wavelet transform (CWT) on the signal and save the image.

    参数/Args:
        signal: 输入信号 (Input signal)
        t: 时间序列 (Time array)
        dt: 时间分辨率 (Time step)
        a: 标记名 (Label for file name and title)
        save_dir: 图片保存目录 (Directory to save the image)
        totalscal: 小波尺度总数 (Total number of scales, default 8192)
        wavename: 小波名称 (Wavelet name, default 'cmor1-1')
    返回/Returns:
        图片完整路径 (Full path of saved image)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Fc = pywt.central_frequency(wavename)
    c = 2 * Fc * totalscal
    scals = c / np.arange(1, totalscal + 1)
    f = fs * pywt.scale2frequency(wavename, scals)
    coefs, freqs = pywt.cwt(signal, scals, wavename)
    plt.figure()
    plt.imshow(np.abs(coefs), aspect='auto', extent=[t.min(), t.max(), f.max(), f.min()], cmap='jet')
    plt.colorbar()
    plt.title(f'Continuous Wavelet Transform {a}')
    plt.xlabel('t(s)')
    plt.ylabel('Frequency (Hz)')
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlim(xlim)
    plt.ylim(ylim)
    fn = os.path.join(save_dir, f'{shot_id:03d}wavelet{a}.png')
    plt.savefig(fn, dpi=600)
    plt.close()
    return fn

