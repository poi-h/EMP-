import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import plot_utils

# 发此号
shot_id = 65

# 数据文件路径
folder_path = os.path.expanduser('~/Desktop/data_EMP/')
fn1 = os.path.join(folder_path, '20Au', '{:03d}.csv'.format(shot_id))
fn2 = os.path.join(folder_path, 'attenuate.xlsx')
save_dir = os.path.join(folder_path, '20Au', '{:03d}'.format(shot_id))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 读取数据
M = pd.read_csv(fn1, skiprows=14, header=None).values

# 延迟
tdelay0 = 198.646 - 45.08 + 13.55  # 13.55是根据信号对时间进行修正
tdelay1 = np.array([275.675 / 2, 277.351 / 2, 277.767 / 2, 269.954 / 2, 312.749 / 2, 287.935 / 2])

# 衰减
attenuate0 = 5
attenuate = pd.read_excel(fn2, header=None, usecols="B:G", skiprows=shot_id, nrows=1).values.flatten()  # skiprows随发次号修改

# 示波器采样率
fs = 12.5e9

# 时间分辨率
dt = 1 / fs

# 假设 t 和 E 已经定义（你需要根据实际数据生成 t 和 E）
# 这里举例：假设取第1组数据
a = 1
tdelay = (tdelay1[a-1] - tdelay0) * np.ones(len(M[:, 3 * a - 2])) * 1e-9
t = M[:, 3 * a - 3] - tdelay
signal = M[:, 3 * a - 2] * (10 ** ((attenuate[a-1] + attenuate0) / 20))
E = signal / 0.053

# 取 t=0 到 6e-8 区间
mask = (t >= 0) & (t <= 6e-8)
t_sel = t[mask]
E_sel = E[mask]

# FFT
L = len(t_sel)
Y = np.fft.fft(E_sel)
P2 = np.abs(Y / L)  # 双边谱
P1 = P2[:L // 2]  # 单边谱
if L > 2:
    P1[1:-1] = 2 * P1[1:-1]  # 补偿能量
f = fs * np.arange(0, L // 2) / L

# 物理常数
c = 3e8  # 光速 m/s
G_dBi = 4  # 天线增益（dBi）
G_linear = 10**(G_dBi / 10)

# 频率范围
mask = (f >= 700e6) & (f <= 2700e6)
f = f[mask]
P1 = P1[mask]
wavelengths = c / f

# 有效面积函数 A_e(f)
A_e = G_linear * wavelengths**2 / (4 * np.pi)

# 示例：信号功率谱密度 S(f)
# 可以替换为你实际的谱函数，比如高斯、均匀分布等
# 这里我们用一个中心在 1700 MHz、标准差 300 MHz 的高斯谱
S_f = P1/A_e

plt.figure()
plt.plot(f, S_f, color='r')
plt.xlim([0, 6e9])
plt.title(f'Single-Sided Amplitude Spectrum {a}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|P1|')
plt.show()

# 归一化功率谱（不归一化也可以，只是不会变单位）
S_f_norm = S_f / np.trapz(S_f, f)

# 加权平均有效面积
A_e_avg_weighted = np.trapz(A_e * S_f_norm, f)

print(f"加权平均有效面积为：{A_e_avg_weighted:.4f} m²")

# 可视化（可选）
# plt.figure(figsize=(8,4))
# plt.twinx()
# plt.plot(f/1e6, A_e, label="有效面积 A_e(f)", color='blue')
# plt.plot(f/1e6, S_f_norm, label="功率谱密度 S(f)", color='orange')
# plt.xlabel("频率 (MHz)")
# plt.title("有效面积与功率谱")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()