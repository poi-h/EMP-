import numpy as np
from scipy import signal
import pandas as pd
import os
import matplotlib.pyplot as plt
import plot_utils

# 发次号
shot_id = 68

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn1 = os.path.join(folder_path, '20Cudc', f'{shot_id:03d}.csv')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
save_dir = os.path.join(folder_path, 'cudclvbo', f'{shot_id:03d}')
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

# 低通滤波参数
fc = 300e6           # 截止频率 200 MHz
numtaps = 725        # 滤波器阶数
beta = 5.65          # Kaiser窗，~60 dB

# 多组处理
for a in range(1, 2):  # 只处理第1组

    # === Step 1. 时间轴和信号处理 ===
    tdelay = (tdelay1[a - 1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
    t = M[:, 3 * a - 3] - tdelay

    # 消除衰减
    sig = M[:, 3 * a - 2] * (10 ** ((attenuate[a - 1] + attenuate0) / 20))

    # 电场转换
    E = 27.46 * sig

    # 取 t=0 到 80ns 区间
    mask = (t >= -5e-9) & (t <= 75e-9)
    t_sel = t[mask]
    E_sel = E[mask]

    # === Step 2. 设计低通滤波器 ===
    taps = signal.firwin(numtaps, fc, fs=fs, window=('kaiser', beta), pass_zero='lowpass')

    # === Step 3. 零填充后滤波 ===
    x_pad = np.pad(E_sel, (1000, 1000), mode='constant')
    y_pad = signal.filtfilt(taps, [1.0], x_pad)
    y = y_pad[1000:-1000]

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
    plt.savefig(os.path.join(save_dir, f'{a}signal.png'), dpi=600)
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
    plt.xlim(0, 0.3)  # 只看低频0~1 GHz
    plt.legend()
    plt.title(f"{shot_id}fft(Δf ≈ {fs/N_fft/1e6:.2f} MHz)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{a}fft.png'), dpi=600)
    plt.close()

    # plot_utils.fft_plot(E_sel, fs, a, shot_id,xlim=(0, 0.3e9))
    # plot_utils.fft_plot(y, fs, f'{a}_filt', shot_id,xlim=(0, 0.3e9))

print(f'{attenuate}')
