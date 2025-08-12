import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
read_dir = os.path.join(folder_path, '20Au')
save_dir = read_dir

# 延迟参数
tdelay0 = 198.646 - 45.08 + 13.55
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减常数
attenuate0 = 5

# 示波器采样率
fs = 12.5e9
dt = 1 / fs

# 要处理的shot_id列表
shot_id_list = [5, 10, 44, 48, 50, 57, 62, 63, 65]
a_list = [1, 2, 3, 4, 5, 6]

for a in a_list:
    # 时域图
    fig, axs = plt.subplots(len(shot_id_list), 1, sharex=True, figsize=(8, 16))
    for idx, shot_id in enumerate(shot_id_list):
        fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            print(f"文件不存在: {fn1}")
            continue
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        t = M[:, 3 * a - 3] - tdelay
        signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal
        mask = (t >= 0) & (t <= 6e-8)
        t_sel = t[mask]
        E_sel = E[mask]
        axs[idx].plot(t_sel, E_sel, label=f'Shot {shot_id:03d}')
        axs[idx].set_ylabel(f'Shot {shot_id:03d}\nE(V/m)')
        axs[idx].legend()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axs[idx].yaxis.set_major_formatter(formatter)
    axs[-1].set_xlabel("t(s)")
    plt.xlim(0, 6e-8)
    plt.suptitle(f'CH{a} 时域')
    fn_img = os.path.join(save_dir, f'ch{a:01d}_multishot_subplot.png')
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    plt.savefig(fn_img, bbox_inches='tight', dpi=600)
    plt.close()

    # 傅立叶变换图
    fig_fft, axs_fft = plt.subplots(len(shot_id_list), 1, sharex=True, figsize=(8, 16))
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
        mask = (t >= 0) & (t <= 6e-8)
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
        axs_fft[idx].legend()
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axs_fft[idx].yaxis.set_major_formatter(formatter)
    axs_fft[-1].set_xlabel("Frequency (Hz)")
    plt.xlim(0, 6e9)
    plt.suptitle(f'CH{a} fft')
    fn_img_fft = os.path.join(save_dir, f'ch{a:01d}_multishot_fft_subplot.png')
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    plt.savefig(fn_img_fft, bbox_inches='tight', dpi=600)
    plt.close()