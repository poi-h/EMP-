import numpy as np
import pandas as pd
import os
# import utils_excel as utils
# import plot_utils as pu
# import utils_Bdot as utilsb
import utils

#发次号
# shot_list = [122,123,124,127,129,131,134,135,136,139,140,141]
shot_list = [125,126,128,132,133,137,138]
# shot_list = [125]

b=1
for shot_id in shot_list:
    b=b+1
    # 数据文件路径
    # folder_path = os.path.expanduser('~/Desktop/data_EMP/')
    # fn1 = os.path.join(folder_path, '20Cudc', f'{shot_id:03d}.csv')
    # fn2 = os.path.join(folder_path, 'attenuate.xlsx')
    # save_dir = os.path.join(folder_path, '20Cudcxlsx')
    folder_path = os.path.expanduser('~/Desktop/B-dot')
    fn1 = os.path.join(folder_path, 'ps', f'{shot_id:03d}.csv')
    fn2 = os.path.join(folder_path, 'attenuate.xlsx')
    save_dir = os.path.join(folder_path, 'hBpspro')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取数据
    M = pd.read_csv(fn1, skiprows=14, header=None).values

    # 延迟
    tdelay0 = 198.646 - 45.08 + 13.55 # 13.55是根据信号对时间进行修正
    tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

    # 衰减
    attenuate0 = 5
    attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()#skiprows随发次号修改

    # 示波器采样率
    fs = 12.5e9

    # 时间分辨率
    dt = 1 / fs

    # 多组处理
    for a in [7]:  # 只处理第5组数据

        # 消除延迟
        # tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
        # t = M[:, 3 * a - 3] - tdelay
        if a == 7:
            tdelay = (tdelay1[0] - tdelay0 + 75) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
            t = M[:, 3 * a - 3] - tdelay
        else:
            tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
            t = M[:, 3 * a - 3] - tdelay

        # 消除衰减
        # signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
        if a == 7 :
            signal = M[:, 3 * a - 2] * (10 ** ((attenuate[0] + attenuate0) / 20))
        else:
            signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))

        # 电场转换
        E = 27.46 * signal

        # 取 t=0 到 6e-8 区间
        mask = (t >= -5e-9) & (t <= 70e-9)
        t_sel = t[mask]
        E_sel = E[mask]
        E_sel = utils.highpass_filter(E_sel, fs, 50e6)
        E_sel = utils.process_signal(t_sel, E_sel)
        E_sel = E_sel - np.mean(E_sel[t_sel < 0])

        # 绘制原始信号的时间序列图
        # fn = utils.signal_write(E_sel, t_sel, a, shot_id, save_dir=save_dir, col_index=b)
        # fn = utils.signal_plot(E_sel, t_sel, a, shot_id, save_dir=save_dir, ylim=(-2e-5, 2e-5))

        # FFT
        # fn = utils.fft_write(E_sel, fs, a, shot_id, save_dir=save_dir, col_index=b)
        # fn = utils.fft_plot(E_sel, fs, a, shot_id, save_dir=save_dir, xlim=(0, 6e9))
        
        # 通过A_e修正的FFT
        # fn = utils.fft_plot_Ae(E_sel, fs, a, shot_id, save_dir=save_dir, xlim=(0, 3e9))

        # Wavelet
        fn = utils.cwt_plot(E_sel, t_sel, fs, a, shot_id, save_dir=save_dir, xlim=(t_sel.min(), t_sel.max()))

    print(f'{attenuate}')