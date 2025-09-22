import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import os
import plot_utils

# 发次号
shot_id = 34

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn1 = os.path.join(folder_path, '20Cudc', f'{shot_id:03d}.csv')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
save_dir = os.path.join(folder_path, 'cudchpf', f'{shot_id:03d}')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取数据
M = pd.read_csv(fn1, skiprows=14, header=None).values

# 延迟
tdelay0 = 198.646 - 45.08 + 13.55  # 13.55是根据信号对时间进行修正
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2,
                    269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减
attenuate0 = 5
attenuate = pd.read_excel(fn2, header=None, usecols="B:G",
                          skiprows=shot_id, nrows=1).values.flatten()

# 示波器采样率
fs = 12.5e9  # 12.5 GHz
dt = 1 / fs

# 参数
cutoff = 2e8    # 高通截止频率 Hz
order = 4      # 滤波器阶数

# 多组处理
for aa in range(1, 2):  # 只处理第1组

    # === Step 1. 时间轴和信号处理 ===
    tdelay = (tdelay1[aa - 1] - tdelay0) * np.ones(len(M[:, 3 * aa - 2])) * 1e-9
    t = M[:, 3 * aa - 3] - tdelay

    # 消除衰减
    sig = M[:, 3 * aa - 2] * (10 ** ((attenuate[aa - 1] + attenuate0) / 20))

    # 电场转换
    E = 27.46 * sig

    # 取 t=0 到 80ns 区间
    mask = (t >= -5e-9) & (t <= 75e-9)
    t_sel = t[mask]
    E_sel = E[mask]

    # 设计 Butterworth 高通滤波器
    b, a = signal.butter(order, cutoff/(0.5*fs), btype='highpass')

    # 进行零相位滤波（离线处理，推荐）
    y = signal.filtfilt(b, a, E_sel)

    # === Step 4. 时域信号可视化 ===
    plt.figure(figsize=(10, 4))
    plt.plot(t_sel * 1e9, E_sel, label="original", alpha=0.5)
    plt.plot(t_sel * 1e9, y, label="lvbo", linewidth=1.2)
    plt.xlabel("times [ns]")
    plt.ylabel("E [V/m]")
    plt.xlim(-5, 75)
    plt.legend()
    plt.title(f"{shot_id}(fc = 300 MHz)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{aa}signal.png'), dpi=600)
    plt.close()

    # === Step 5. FFT计算 ===
    N = len(t_sel)                       # 有效点数
    N_fft = 8192                        # FFT点数（零填充更平滑）
    freqs = np.fft.rfftfreq(N_fft, dt)  # 单边频率
    fft_orig = np.fft.rfft(E_sel, n=N_fft)
    fft_filt = np.fft.rfft(y, n=N_fft)

    # === Step 6. 绘制频谱对比 ===
    plt.figure(figsize=(10, 5))
    plt.plot(freqs / 1e9, np.abs(fft_orig), label="original", alpha=0.5)
    plt.plot(freqs / 1e9, np.abs(fft_filt), label="lvbo", linewidth=1.2)
    plt.xlabel("frequency [GHz]")
    plt.ylabel("amplitude")
    plt.xlim(0, 3)  # 只看低频0~1 GHz
    plt.legend()
    plt.title(f"{shot_id}fft(Δf ≈ {fs/N_fft/1e6:.2f} MHz)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{aa}fft.png'), dpi=600)
    plt.close()

    plot_utils.cwt_plot(y, t, fs, shot_id, shot_id, save_dir=save_dir,xlim=(-5e-9, 75e-9), ylim=(0, 3e9))