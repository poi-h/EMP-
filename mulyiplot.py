import numpy as np
import pandas as pd
import os
import plot_utils
import matplotlib.pyplot as plt
import pywt
import matplotlib.ticker as ticker

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
read_dir = os.path.join(folder_path, '20Au')

# 延迟参数
tdelay0 = 198.646 - 45.08 + 13.55
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减常数
attenuate0 = 5

# 示波器采样率
fs = 12.5e9
dt = 1 / fs

# 要处理的shot_id列表（例如1到10）
shot_id_list = [44]
a_list = [1, 2, 3, 4, 5, 6]

for shot_id in shot_id_list:
    fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
    if not os.path.isfile(fn1):
        print(f"文件不存在: {fn1}")
        continue
    M = pd.read_csv(fn1, skiprows=14, header=None).values
    attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
    fig, axs = plt.subplots(len(a_list), 1, sharex=True, figsize=(8, 12))
    for a in a_list:
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        t = M[:, 3 * a - 3] - tdelay
        signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal
        
        mask = (t >= 0) & (t <= 6e-8)
        t_sel = t[mask]
        E_sel = E[mask]

        axs[a-1].plot(t_sel, E_sel, label=f'{shot_id:03d}CH{a}')
        axs[a-1].legend()
        # 强制科学计数法
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))  # 总是使用科学计数法
        axs[a-1].yaxis.set_major_formatter(formatter)
    axs[3].set_ylabel("E(V/m)")
    axs[len(a_list)-1].set_xlabel("t(s)")
    plt.xlim(0, 6e-8)
    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    plt.savefig("fig.png", bbox_inches='tight', dpi=600)
    plt.show()
