import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from scipy import signal   # <<< 新增，用于寻峰

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
folder = '20Au'
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
shot_id_list = [44,48,50,62,63,65]
a_list = [1]

# 确保保存目录存在
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for a in a_list:
    # =================== 时域信号图 ===================
    fig, axs = plt.subplots(len(shot_id_list), 1, sharex=True, sharey=True, figsize=(8, 12))
    for idx, shot_id in enumerate(shot_id_list):
        fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            print(f"文件不存在: {fn1}")
            sys.exit(0)
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3*a - 2])) * 1e-9
        t = M[:, 3*a - 3] - tdelay
        signal_raw = M[:, 3*a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal_raw
        mask = (t >= -5.5e-9) & (t <= 74.5e-9)
        t_sel = t[mask]
        E_sel = E[mask]
        axs[idx].plot(t_sel, E_sel, label=f'Shot {shot_id:03d}')
        axs[idx].set_ylabel(f'Shot {shot_id:03d}\nE(V/m)')
        axs[idx].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
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

    # =================== FFT + 自动寻峰 ===================
    fig_fft, axs_fft = plt.subplots(len(shot_id_list), 1, sharex=True, figsize=(8, 12))
    for idx, shot_id in enumerate(shot_id_list):
        fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            continue
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3*a - 2])) * 1e-9
        t = M[:, 3*a - 3] - tdelay
        signal_raw = M[:, 3*a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal_raw
        mask = (t >= -5.5e-9) & (t <= 74.5e-9)
        E_sel = E[mask]
        L = len(E_sel)

        # FFT
        Y = np.fft.fft(E_sel)
        P2 = np.abs(Y / L)
        P1 = P2[:L // 2]
        if L > 2:
            P1[1:-1] = 2 * P1[1:-1]
        f = fs * np.arange(0, L // 2) / L

        # ===== 自动寻峰 =====
        # prominence 越大 → 只保留主要峰
        peaks, properties = signal.find_peaks(
            P1,
            prominence=np.max(P1) * 0.05,  # <<< 可调节显著性阈值
            distance=1                    # <<< 相邻峰最小间隔
        )

        # 绘制FFT曲线
        axs_fft[idx].plot(f, P1, 'r', label=f'Shot {shot_id:03d}')
        axs_fft[idx].scatter(f[peaks], P1[peaks], color='blue', s=25, label="Peaks")  # <<< 新增：红点标峰
        axs_fft[idx].set_ylabel(f'Shot {shot_id:03d}\n|P1|')
        axs_fft[idx].yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

        # 格式设置
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axs_fft[idx].yaxis.set_major_formatter(formatter)
        axs_fft[idx].xaxis.set_major_formatter(formatter)

        # 在图上标注峰值频率
        for p in peaks:
            axs_fft[idx].annotate(f"{f[p]/1e9:.2f} GHz",
                                  xy=(f[p], P1[p]),
                                  xytext=(5, 5),
                                  textcoords="offset points",
                                  fontsize=7,
                                  color='blue')

    axs_fft[-1].set_xlabel("Frequency (Hz)")
    plt.xlim(0, 0.5e9)
    plt.suptitle(f'FFT & Peaks {a}')
    fn_img_fft = os.path.join(save_dir, f'{folder}ch{a}fft_peaks.png')
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    # plt.savefig(fn_img_fft, bbox_inches='tight', dpi=600)
    plt.show()
