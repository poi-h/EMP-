import numpy as np
import pandas as pd
import os
import csv

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
save_dir = os.path.join(folder_path, '1000Ta')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 延迟参数
tdelay0 = 198.646 - 45.08 + 13.55
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减常数
attenuate0 = 5

# 示波器采样率
fs = 12.5e9
dt = 1 / fs

# 要处理的shot_id列表（例如1到10）
shot_id_list = [33,37,39,55,60,67,81,82,137]

csv_path = os.path.join(save_dir, 'firstpeak.csv')
file_exists = os.path.isfile(csv_path)
with open(csv_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(['shot_id'] + [f'E_max_{a}' for a in range(1, 7)])
    for shot_id in shot_id_list:
        fn1 = os.path.join(save_dir, f'{shot_id:03d}.csv')
        if not os.path.isfile(fn1):
            print(f"文件不存在: {fn1}")
            continue
        M = pd.read_csv(fn1, skiprows=14, header=None).values
        attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()
        E_max_list = []
        for a in range(1, 7):
            tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
            t = M[:, 3 * a - 3] - tdelay
            signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
            E = 27.46 * signal
            mask = (t >= -5e-9) & (t <= 5e-9)
            t_sel = t[mask]
            E_sel = E[mask]
            E_max = np.max(np.abs(E_sel))
            E_max_list.append(E_max)
        writer.writerow([shot_id] + E_max_list)

