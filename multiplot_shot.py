import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import pywt

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
folder = 'mlpt'
read_dir = os.path.join(folder_path, folder)
save_dir = os.path.join(read_dir, 'pictures')

# 延迟参数
tdelay0 = 198.646 - 45.08 + 13.55
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减常数
attenuate0 = 5

# 示波器采样率
fs = 12.5e9
dt = 1 / fs

# 要处理的shot_id列表
shot_id_list = [69,109,110,21,114]
a_list = [1]

tmin=-5.5e-9
tmax=74.5e-9

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for a in a_list:
    # 时域图
    fig, axs = plt.subplots(len(shot_id_list), 1, sharex=True, sharey=True, figsize=(8, 12))
    for idx, shot_id in enumerate(shot_id_list):
        fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            print(f"文件不存在: {fn1}")
            sys.exit(0)
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        t = M[:, 3 * a - 3] - tdelay
        signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal
        mask = (t >= tmin) & (t <= tmax)
        t_sel = t[mask]
        E_sel = E[mask]
        axs[idx].plot(t_sel, E_sel, label=f'Shot {shot_id:03d}')
        axs[idx].set_ylabel(f'Shot {shot_id:03d}\nE(V/m)')
        axs[idx].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        # axs[idx].legend()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((3, 3))
        axs[idx].yaxis.set_major_formatter(formatter)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-9, -9))
        axs[idx].xaxis.set_major_formatter(formatter)
    axs[-1].set_xlabel("t(ns)")
    plt.xlim(-5.5e-9, 74.5e-9)
    plt.suptitle(f'Electric field {a}')
    fn_img = os.path.join(save_dir, f'{folder}ch{a}signal.png')
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    plt.savefig(fn_img, bbox_inches='tight', dpi=600)
    plt.close()

    # 傅立叶变换图
    fig_fft, axs_fft = plt.subplots(len(shot_id_list), 1, sharex=True, figsize=(8, 12))
    for idx, shot_id in enumerate(shot_id_list):
        fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            continue
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        t = M[:, 3 * a - 3] - tdelay
        signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal
        mask = (t >= tmin) & (t <= tmax)
        E_sel = E[mask]
        L = len(E_sel)
        Y = np.fft.fft(E_sel)
        P2 = np.abs(Y / L)
        P1 = P2[:L // 2]
        if L > 2:
            P1[1:-1] = 2 * P1[1:-1]
        f = fs * np.arange(0, L // 2) / L
        axs_fft[idx].plot(f, P1, 'r', label=f'Shot {shot_id:03d}')
        axs_fft[idx].set_ylabel(f'Shot {shot_id:03d}\n|P1|')
        axs_fft[idx].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        # axs_fft[idx].legend()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axs_fft[idx].yaxis.set_major_formatter(formatter)
        axs_fft[idx].xaxis.set_major_formatter(formatter)
    axs_fft[-1].set_xlabel("Frequency (Hz)")
    plt.xlim(0, 3e9)
    plt.suptitle(f'FFT {a}')
    fn_img_fft = os.path.join(save_dir, f'{folder}ch{a}fft.png')
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    plt.savefig(fn_img_fft, bbox_inches='tight', dpi=600)
    plt.close()

    # 小波变换图
    fig_cwt, axs_cwt = plt.subplots(len(shot_id_list), 1, sharex=True, figsize=(8, 12))
    for idx, shot_id in enumerate(shot_id_list):
        fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            continue
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        t = M[:, 3 * a - 3] - tdelay
        signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal
        mask = (t >= tmin) & (t <= tmax)
        t_sel = t[mask]
        E_sel = E[mask]

        # 连续小波变换
        wavename = 'cmor1-1'  # 复Morlet小波
        Fc = pywt.central_frequency(wavename)
        totalscal=8192
        c = 2 * Fc * totalscal
        scales = c / np.arange(1, totalscal + 1)
        coefficients, freqs = pywt.cwt(E_sel, scales, wavename, sampling_period=dt)

        im = axs_cwt[idx].imshow(np.abs(coefficients),
                                 extent=[t_sel[0]*1e9, t_sel[-1]*1e9, freqs[0], freqs[-1]],
                                 aspect='auto', cmap='jet', origin='lower')
        axs_cwt[idx].set_ylim(0, 3e9)
        axs_cwt[idx].set_ylabel(f'Shot {shot_id:03d}\nFreq(Hz)')
        axs_cwt[idx].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

    axs_cwt[-1].set_xlabel("t (ns)")
    plt.suptitle(f'CWT {a}')
    fn_img_cwt = os.path.join(save_dir, f'{folder}ch{a}cwt.png')
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    fig_cwt.colorbar(im, ax=axs_cwt.ravel().tolist(), label='|CWT Coeff|')
    plt.savefig(fn_img_cwt, bbox_inches='tight', dpi=600)
    plt.close()