import numpy as np
import pandas as pd
import os
import utils

#发次号
# shot_list = [122,123,124,127,129,131,134,135,136,139,140,141]
# shot_list = [125,126,128,132,133,137,138]
shot_list = [1]

b=1
for shot_id in shot_list:
    b=b+1
    # 数据文件路径
    folder_path = os.path.expanduser('~/Desktop/B-dot')
    fn1 = os.path.join(folder_path, f'{shot_id:03d}.csv')
    # fn2 = os.path.join(folder_path, 'attenuate.xlsx')
    save_dir = os.path.join(folder_path, 'hBpro')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取数据
    M = pd.read_csv(fn1, skiprows=12, header=None).values

    # 延迟
    tdelay0 = 0
    tdelay1 = np.array([0,0])

    # 衰减
    attenuate0 = 3
    attenuate = [30,30]
    # attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()#skiprows随发次号修改

    # 示波器采样率
    fs = 12.5e9

    # 时间分辨率
    dt = 1 / fs

    # 多组处理
    for a in [1,2]:  # 只处理第5组数据

        # 消除延迟
        tdelay = (tdelay1[a-1] - tdelay0 + 75) * np.ones(len(M[:,0])) * 1e-9
        t = M[:,0] - tdelay

        # 消除衰减
        signal = M[:,a] * (10 ** ((attenuate[a-1] + attenuate0) / 20))

        # 取 t=0 到 6e-8 区间
        mask = (t >= -50e-9) & (t <= 200e-9)
        t_sel = t[mask]
        signal_sel = signal[mask]

        # 绘制原始信号的时间序列图
        # fn = utils.signal_write(signal_sel, t_sel, a, shot_id, save_dir=save_dir, col_index=b)
        fn = utils.signal_plot(signal_sel, t_sel, a, shot_id, save_dir=save_dir, xlim=(-50e-9, 200e-9), ylim=(-30, 30))
        

        # FFT
        # fn = utils.fft_write(signal_sel, fs, a, shot_id, save_dir=save_dir, col_index=b)
        fn = utils.fft_plot(signal_sel, fs, a, shot_id, save_dir=save_dir, xlim=(0, 6e9))

        # Wavelet
        # fn = utils.cwt_plot(signal_sel, t_sel, fs, a, shot_id, save_dir=save_dir, xlim=(t_sel.min(), t_sel.max()))

        # 去除基线漂移
        signal3 = signal - np.mean(signal[t < -50e-9])
        # fn = utils.signal_plot(signal3, t_sel, a, shot_id, save_dir=save_dir, xlim=(-50e-9, 200e-9), ylim=(-30, 30))

        # 取 t=0 到 6e-8 区间
        mask = (t >= -50e-9) & (t <= 200e-9)
        t_sel = t[mask]
        signal_sel = signal3[mask]

        # 高通滤波
        # signal1 = utils.highpass_filter(signal_sel, fs, 1e6) # 尽量不要用，容易出问题
        # fn = utils.signal_plot(signal1, t_sel, a, shot_id, save_dir=save_dir, xlim=(-50e-9, 200e-9), ylim=(-30, 30))

        # 低通滤波
        signal2 = utils.lowpass_filter(signal_sel, fs, 1e9)
        # fn = utils.signal_plot(signal2, t_sel, a, shot_id, save_dir=save_dir, xlim=(-50e-9, 200e-9), ylim=(-30, 30))

        B = utils.process_signal(t_sel, signal2, A=2e-5)
        fn = utils.B_plot(B, t_sel, a, shot_id, save_dir=save_dir, xlim=(-50e-9, 200e-9), ylim=(-3, 3))
        

    print(f'{attenuate}')