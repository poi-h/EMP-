import numpy as np
import pandas as pd
import os
import utils
import matplotlib.pyplot as plt
import pywt
import matplotlib.ticker as ticker
import matplotlib as mpl

mpl.rcParams['font.size'] = 24          # 默认字号（标签）
mpl.rcParams['axes.labelsize'] = 24     # 坐标轴标签
mpl.rcParams['xtick.labelsize'] = 22    # x刻度
mpl.rcParams['ytick.labelsize'] = 22    # y刻度

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
read_dir = os.path.join(folder_path, '20Cudc')
save_dir = read_dir

# 延迟参数
tdelay0 = 198.646 - 45.08 + 13.55
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减常数
attenuate0 = 5

# 示波器采样率
fs = 12.5e9
dt = 1 / fs

# 要处理的shot_id列表（例如1到10）
shot_id_list = [36]
a_list = range(1, 5)

tmin=-5.5e-9
tmax=80e-9

for shot_id in shot_id_list:
    # fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
    # if not os.path.isfile(fn1):
    #     print(f"文件不存在: {fn1}")
    #     continue
    # M = pd.read_csv(fn1, skiprows=14, header=None).values
    # attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()

    # fig, axs = plt.subplots(len(a_list), 1, sharex=True, figsize=(8, 12))
    # for a in a_list:
    #     tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
    #     t = M[:, 3 * a - 3] - tdelay
    #     signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
    #     E = 27.46 * signal
        
    #     mask = (t >= 0) & (t <= 6e-8)
    #     t_sel = t[mask]
    #     E_sel = E[mask]

    #     axs[a-1].plot(t_sel, E_sel, label=f'{shot_id:03d}CH{a}')
    #     axs[a-1].legend()
    #     # 强制科学计数法
    #     formatter = ticker.ScalarFormatter(useMathText=True)
    #     formatter.set_powerlimits((0, 0))  # 总是使用科学计数法
    #     axs[a-1].yaxis.set_major_formatter(formatter)
    # axs[3].set_ylabel("E(V/m)")
    # axs[len(a_list)-1].set_xlabel("t(s)")
    # fn1 = os.path.join(save_dir, f'{shot_id:03d}_mlpt.png')
    # plt.suptitle(f'Shot ID: {shot_id:03d}', fontsize=16)
    # plt.xlim(0, 6e-8)
    # plt.subplots_adjust(top=0.95, wspace=0, hspace=0)
    # plt.savefig(fn1, bbox_inches='tight', dpi=600)
    # plt.show()

    # 小波变换
    fn1 = os.path.join(read_dir, f'{shot_id:03d}.csv')
    if not os.path.isfile(fn1):
        print(f"文件不存在: {fn1}")
        continue

    # ===== 读数据 =====
    M = pd.read_csv(fn1, skiprows=14, header=None).values
    attenuate = pd.read_excel(fn2, header=None,
                              usecols="B:G",
                              skiprows=shot_id,
                              nrows=1).values.flatten()

    # ===== 小波图 =====
    cm = 1/2.54
    fig, axs = plt.subplots(len(a_list), 1, sharex=True, figsize=(25*cm, 37.5*cm))
    # ===== 全局标签 =====
    fig.supylabel('frequency (GHz)', fontsize=24)

    for a in a_list:

        # ===== 数据处理 =====
        tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3*a-2])) * 1e-9
        t = M[:, 3*a-3] - tdelay
        signal = M[:, 3*a-2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        E = 27.46 * signal

        mask = (t >= tmin) & (t <= tmax)
        t_sel = t[mask]
        E_sel = E[mask]

        # ===== 小波 =====
        wavename = 'cmor1-1'
        Fc = pywt.central_frequency(wavename)

        totalscal = 8192
        c = 2 * Fc * totalscal
        scales = c / np.arange(1, totalscal + 1)

        coefficients, freqs = pywt.cwt(E_sel, scales, wavename,
                                       sampling_period=dt)

        # 去掉低频
        # coefficients = coefficients[1:, :]
        # freqs = freqs[1:]

        ax = axs[a-1]

        im = ax.imshow(np.abs(coefficients),
                       extent=[t_sel[0]*1e9, t_sel[-1]*1e9,
                               freqs[0]/1e9, freqs[-1]/1e9],
                       aspect='auto',
                       cmap='jet',
                       origin='lower')

        # ===== ⭐ 坐标轴设置（最终规范版）=====

        # ---- y轴：全部有刻度 ----
        ax.set_ylim(0, 3)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.tick_params(axis='y')
        ax.tick_params(direction='in')

        # # 👉 只在中间保留 y 轴标签
        # if a == (len(a_list)+1)//2:
        #     ax.set_ylabel('frequency (GHz)')
        # else:
        #     ax.set_ylabel('')

        # ---- x轴：全部有刻度 ----
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.tick_params(axis='x')

        # 👉 只在最下面保留 x 轴标签
        if a == len(a_list):
            ax.set_xlabel('time (ns)')
        else:
            ax.set_xlabel('')

    # ===== 布局（留空间给坐标轴）=====
    plt.subplots_adjust(
    left=0.12,   # ⭐ 给ylabel空间
    right=0.90,
    bottom=0.05, # ⭐ 给xlabel空间
    top=0.99,
    hspace=0
)

    # ===== colorbar =====
    cbar = fig.colorbar(im, ax=axs,
                        fraction=0.05,
                        pad=0.02)
    # cbar.set_label('Intensity')

    # ===== 标题 =====
    # plt.suptitle(f'CWT Spectrum (Shot {shot_id:03d})')

    # ===== 保存 =====
    fn_img = os.path.join(save_dir, f'{shot_id:03d}_cwt.png')
    plt.savefig(fn_img, dpi=600)

    plt.close()
