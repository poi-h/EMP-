import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import pywt

import matplotlib as mpl

# 全局字体设置
mpl.rcParams['font.family'] = 'Arial'        # 字体（可以改成 'Times New Roman'、'SimHei'、'Microsoft YaHei' 等）
mpl.rcParams['font.size'] = 22               # 全局字体大小
mpl.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题（尤其是中文）

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
folder = '20Au'
read_dir = os.path.join(folder_path, folder)
save_dir = os.path.join(read_dir, 'cwt_pictures')

# 延迟参数
tdelay0 = 198.646 - 45.08 + 13.55
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减常数
attenuate0 = 5

# 示波器采样率
fs = 12.5e9
dt = 1 / fs

# 要处理的shot_id列表
shot_id_list = [44]
a_list = [1,2,3,4,5,6]  # 处理所有6组数据

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for shot_id in shot_id_list:
    fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
    if not os.path.isfile(fn1):
        print(f"文件不存在: {fn1}")
        break
    M = pd.read_csv(fn1, skiprows=14, header=None).values
    attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
    for a in a_list:
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        t = M[:, 3 * a - 3] - tdelay
        signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal
        mask = (t >= 0e-9) & (t <= 70e-9)
        t_sel = t[mask]
        E_sel = E[mask]

        # CWT
        wavename = 'cmor1-1'
        totalscal = 8192
        Fc = pywt.central_frequency(wavename)
        c = 2 * Fc * totalscal
        scals = c / np.arange(1, totalscal + 1)
        f_cwt = fs * pywt.scale2frequency(wavename, scals)
        coefs, freqs = pywt.cwt(E_sel, scals, wavename)

        fig = plt.figure(figsize=(9.843, 7.382))
        gs = gridspec.GridSpec(2, 2, width_ratios=[25, 1], height_ratios=[1, 1])

        # 上：CWT
        ax_cwt = fig.add_subplot(gs[0, 0], sharex=None)
        im = ax_cwt.imshow(
            np.abs(coefs), aspect='auto',
            extent=[t_sel.min()*1e9, t_sel.max()*1e9, f_cwt.max()*1e-9, f_cwt.min()*1e-9],
            cmap='jet', origin='upper'
        )
        ax_cwt.set_ylabel('Frequency (GHz)', fontsize=24)
        # ax_cwt.set_title(f'Shot {shot_id:03d} CH{a} CWT')
        ax_cwt.set_ylim(0, 2)
        # ax_cwt.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
        ax_cwt.tick_params(labelbottom=False)
        ax_cwt.tick_params(direction='in')

        # colorbar 独立一列
        ax_cbar = fig.add_subplot(gs[0, 1])
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.ax.yaxis.set_offset_position('left')
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax_cbar.yaxis.set_major_formatter(formatter)

        # 下：信号
        ax_sig = fig.add_subplot(gs[1, 0], sharex=ax_cwt)
        ax_sig.plot(t_sel*1e9, E_sel*1e-3, linewidth=2)
        ax_sig.set_xlabel('t(ns)', fontsize=24)
        ax_sig.set_ylabel('E(kV/m)', fontsize=24)
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True) 
        # formatter.set_powerlimits((3, 3))
        # ax_sig.yaxis.set_major_formatter(formatter)
        ax_sig.set_xlim(0, 70)
        ax_sig.tick_params(direction='in')

        plt.subplots_adjust(top=0.95, wspace=0.01, hspace=0)
        fn_img = os.path.join(save_dir, f'{shot_id:03d}CH{a}cwtsignal.png')
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_scientific(True) 
        # formatter.set_powerlimits((0, 0))
        # ax_cwt.yaxis.set_major_formatter(formatter)
        # formatter = ticker.ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-9, -9))
        # ax_cwt.xaxis.set_major_formatter(formatter)
        plt.savefig(fn_img, bbox_inches='tight', dpi=600)
        plt.close()

