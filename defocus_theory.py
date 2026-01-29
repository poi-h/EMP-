import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import constants

import numpy as np

# =====================
# 基本参数（按实验改）
# =====================
lambda0 = 0.8e-6      # 激光波长 [m]
w0 = 10e-6             # 焦点光斑半径 (1/e^2) [m]
eta = 0.3           # 能量转换效率
E_laser = 5.6        # 激光能量 [J]
tau = 30e-15         # 脉宽 (FWHM) [s]
P = 0.6 *E_laser / tau           # 峰值功率 [W]  (能量/脉宽)

# =====================
# 计算瑞利长度
# =====================
zR = np.pi * w0**2 / lambda0

# =====================
# 离焦量（单位 m）
# =====================
# z_d = np.array([0, 100e-6, 200e-6])
z_d = np.linspace(0, 200e-6, 1200)

# =====================
# 光斑半径 & 峰值光强
# =====================
# w = w0 * np.sqrt(1 + (z_d / zR)**2)  # 高斯光束模型
w =  0.18 / 1.2 * z_d + w0   # 几何离焦模型
I0 = 1 * P / (np.pi * w**2)     # 峰值光强 [W/m^2]
# a0 = 0.855e-5 * np.sqrt(I0 * lambda0**2)  # 归一化矢量势
a0 = (constants.e * lambda0 / (2 * np.pi * constants.m_e * constants.c**2)) * np.sqrt(2 * I0 / (constants.epsilon_0 * constants.c))  # 归一化矢量势

# =====================
# 洛伦兹因子
# =====================
gamma = np.sqrt(1 + a0**2)

# =====================
# 平均电子动能（J）
# =====================
E_k = (gamma - 1) * constants.m_e * constants.c**2

# =====================
# 电子温度（eV）
# =====================
Te_eV = E_k / constants.e

# =====================
# 输出结果
# =====================
# print(f"zR = {zR*1e6:.1f} µm, P={P:.2e} W")
# for zi, wi, Ii, ai, Te_eVi in zip(z, w, I0, a0, Te_eV):
#     print(f"z = {zi*1e6:6.0f} µm | "
#           f"w = {wi*1e6:6.2f} µm | "
#           f"I0 = {Ii*1e-4:6.2e} W/cm^2 | "
#           f"a0 = {ai:.2e} | "
#           f"Te = {Te_eVi/1e6:6.2f} MeV")

# =====================

# =====================
# 参数
# =====================
a = eta*E_laser      # ~ 激光能量
b = constants.e**2/(constants.epsilon_0*0.16)      # ~ e^2 / C_tar（很小）

# =====================
# 驻点（理论）
# =====================
x_star = np.sqrt(a * b / np.e)
y_star = np.sqrt(a / (b * np.e))

# =====================
# x 取值范围（避开 0）
# =====================
x_vals = np.linspace(np.min(E_k), np.max(E_k), 1200)
x_vals = E_k

# =====================
# Lambert W 解析解
# =====================
z = a * b / x_vals**2
y_vals = (x_vals / b) * lambertw(z).real   # 只取主支实部
zzz = z = a * b / E_k**2
aaa = (E_k / b) * lambertw(z).real

# =====================
# 作图
# =====================
plt.figure()
plt.plot(z_d*1e6, y_vals, label="N_esc")
# plt.scatter(x_star/constants.e, y_star, zorder=5, label="stationary point")
# plt.scatter(Te_eV, aaa, zorder=5, label="defocus")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()