"""
physics.py
==========
物理参数、势场函数与初始波函数定义模块.

本模块负责所有与物理模型相关的定义, 包括:
  - 自然单位制下的物理常数 (ħ = m = 1)
  - 一维空间网格生成
  - 各种势场: 方势垒 / 方势阱 / 谐振子势 / 双势阱势 / 含时振荡势垒
  - 高斯波包初态及其归一化
  - 动能算符与势能算符的期望值计算
  - 理论透射系数 (WKB 近似与精确解)
  - 动量空间傅里叶分析 (创新拓展)

注意: 本模块不包含任何数值演化算法或绘图逻辑, 以保证逻辑解耦.

AI 协作声明:
    本模块由作者在 AI 助手 (GLM, 智谱 AI) 辅助下编写.
    AI 主要协助: WKB/精确透射系数公式的核对, 动量空间 FFT 的实现.
    物理模型设计、参数选取由作者独立完成.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 自然单位制 (atomic units, ħ = m = 1)
# 在此单位制下, 长度单位为玻尔半径 a0, 时间单位为 ħ/Eh, 能量单位为 Hartree.
# 这样选取可使薛定谔方程 i ∂ψ/∂t = -1/2 ∂²ψ/∂x² + V(x) ψ 形式简洁.
# ---------------------------------------------------------------------------
HBAR = 1.0   # 约化普朗克常数 (无量纲)
MASS = 1.0   # 粒子质量 (无量纲)


def build_grid(x_min=-60.0, x_max=60.0, Nx=1201):
    """
    构建一维均匀空间网格.

    Parameters
    ----------
    x_min, x_max : float
        空间区间端点.
    Nx : int
        网格点数 (含两端), 应取奇数以使中心点位于 x=0.

    Returns
    -------
    x : ndarray, shape (Nx,)
        空间坐标数组.
    dx : float
        空间步长.
    """
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]
    return x, dx


# ---------------------------------------------------------------------------
# 势场函数
# ---------------------------------------------------------------------------
def potential_barrier(x, x0=0.0, width=2.0, height=10.0):
    """
    方势垒 V(x) = height,  当 |x - x0| < width/2;  否则 V = 0.

    Parameters
    ----------
    x : ndarray
        空间坐标.
    x0 : float
        势垒中心位置.
    width : float
        势垒宽度.
    height : float
        势垒高度.

    Returns
    -------
    V : ndarray
        势能数组, 与 x 同形状.
    """
    V = np.zeros_like(x)
    V[np.abs(x - x0) < width / 2.0] = height
    return V


def potential_well(x, x0=0.0, width=10.0, depth=10.0):
    """
    方势阱: V(x) = -depth,  当 |x - x0| < width/2;  否则 V = 0.
    """
    V = np.zeros_like(x)
    V[np.abs(x - x0) < width / 2.0] = -depth
    return V


def potential_harmonic(x, k=1.0, x0=0.0):
    """
    谐振子势 V(x) = 0.5 * k * (x - x0)^2.
    在自然单位制下, 谐振子的本征能量为 E_n = (n + 1/2) * ħω, 其中 ω = sqrt(k/m).
    """
    return 0.5 * k * (x - x0) ** 2


def potential_double_well(x, a=2.0, b=1.0, x0=0.0):
    """
    双势阱势 V(x) = b * (x - x0)^4 - a * (x - x0)^2.
    这是一个典型的对称双势阱, 在 |x - x0| = sqrt(a / (2b)) 处有两个极小值,
    中间存在一个势垒, 可用于研究隧穿劈裂等量子现象.
    """
    y = x - x0
    return b * y ** 4 - a * y ** 2


def potential_time_dependent_barrier(x, t, x0=0.0, width=2.0,
                                     height0=15.0, omega=2.0):
    """
    含时振荡势垒 (创新拓展):

        V(x, t) = height0 * [1 + alpha * sin(omega * t)]   当 |x - x0| < width/2

    势垒高度随时间正弦振荡, 振荡频率为 omega.
    这模拟了外部驱动场对势垒的调制, 可研究光辅助隧穿 (photon-assisted
    tunneling) 等量子动力学现象.

    Parameters
    ----------
    x : ndarray
        空间坐标.
    t : float
        当前时间.
    x0 : float
        势垒中心.
    width : float
        势垒宽度.
    height0 : float
        势垒平均高度.
    omega : float
        振荡角频率.

    Returns
    -------
    V : ndarray
        t 时刻的势能数组.
    """
    V = np.zeros_like(x)
    height_t = height0 * (1.0 + 0.3 * np.sin(omega * t))  # 振幅 30%
    V[np.abs(x - x0) < width / 2.0] = height_t
    return V


# ---------------------------------------------------------------------------
# 初始波函数: 高斯波包
# ---------------------------------------------------------------------------
def gaussian_wavepacket(x, x0=-15.0, k0=5.0, sigma=1.5):
    """
    构造归一化的高斯波包作为初态:

        ψ(x, 0) = (2π σ²)^(-1/4) * exp[-(x - x0)² / (4σ²)] * exp(i k0 x)

    其中 σ 为位置空间宽度, k0 为平均波数, 对应平均动量 p0 = ħ k0,
    平均能量约为 E0 ≈ ħ²k0² / (2m) + ħ² / (8mσ²).

    Parameters
    ----------
    x : ndarray
        空间坐标.
    x0 : float
        波包中心位置.
    k0 : float
        平均波数 (决定波包的初始动量与传播方向).
    sigma : float
        波包在位置空间的宽度参数.

    Returns
    -------
    psi : ndarray (complex)
        归一化的复波函数, 满足 ∫|ψ|² dx = 1.
    """
    # 包络 (实高斯)
    envelope = (2.0 * np.pi * sigma ** 2) ** (-0.25) * \
               np.exp(-(x - x0) ** 2 / (4.0 * sigma ** 2))
    # 平面波相位因子
    plane_wave = np.exp(1j * k0 * x)
    psi = envelope * plane_wave
    # 数值归一化 (消除离散化误差)
    psi = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * (x[1] - x[0]))
    return psi


# ---------------------------------------------------------------------------
# 物理量期望值
# ---------------------------------------------------------------------------
def expectation_position(psi, x, dx):
    """计算位置期望值 <x> = ∫ x |ψ|² dx."""
    return np.sum(x * np.abs(psi) ** 2) * dx


def expectation_momentum(psi, x, dx):
    """计算动量期望值 <p> = ∫ ψ* (-iħ ∂/∂x) ψ dx, 用中心差分近似导数."""
    dpsi = np.gradient(psi, dx)
    return -1j * HBAR * np.sum(np.conj(psi) * dpsi) * dx


def expectation_kinetic(psi, x, dx):
    """
    计算动能期望值 <T> = -ħ²/(2m) ∫ ψ* (∂²/∂x²) ψ dx.
    用二阶中心差分近似二阶导数.
    """
    d2psi = np.gradient(np.gradient(psi, dx), dx)
    T = -HBAR ** 2 / (2.0 * MASS) * np.sum(np.conj(psi) * d2psi) * dx
    return T.real


def expectation_potential(psi, V, dx):
    """计算势能期望值 <V> = ∫ |ψ|² V dx."""
    return np.sum(np.abs(psi) ** 2 * V) * dx


def total_energy(psi, x, V, dx):
    """计算总能量期望值 <H> = <T> + <V>."""
    return expectation_kinetic(psi, x, dx) + expectation_potential(psi, V, dx)


def norm(psi, dx):
    """计算波函数的模方, 用于检验归一化与守恒性."""
    return np.sum(np.abs(psi) ** 2) * dx


# ---------------------------------------------------------------------------
# 理论隧穿概率 (WKB 近似, 用于与数值结果对比)
# ---------------------------------------------------------------------------
def wkb_tunneling_probability(E, V0, a, m=MASS, hbar=HBAR):
    """
    计算 WKB 近似下方势垒的透射系数:

        T ≈ exp[-2 ∫_{x1}^{x2} sqrt(2m(V(x) - E)) / ħ dx]

    对于方势垒 V0 (高度), 宽度 a, 入射能量 E < V0, 积分结果为:

        T ≈ exp[-2 a sqrt(2m(V0 - E)) / ħ]

    Parameters
    ----------
    E : float
        入射粒子能量.
    V0 : float
        势垒高度.
    a : float
        势垒宽度.
    m, hbar : float
        粒子质量与约化普朗克常数.

    Returns
    -------
    T : float
        WKB 透射系数 (0 ~ 1). 当 E >= V0 时返回 1 (经典完全透射).
    """
    if E >= V0:
        return 1.0
    kappa = np.sqrt(2.0 * m * (V0 - E)) / hbar
    return np.exp(-2.0 * kappa * a)


def exact_tunneling_probability_rectangular(E, V0, a, m=MASS, hbar=HBAR):
    """
    方势垒的精确透射系数 (E < V0 情形):

        T = [1 + (V0² sinh²(κ a)) / (4 E (V0 - E))]^(-1)

    其中 κ = sqrt(2m(V0 - E)) / ħ. 这是量子力学教科书的标准结果,
    可作为数值模拟的基准.

    Parameters
    ----------
    E : float
        入射粒子能量 (E < V0).
    V0 : float
        势垒高度.
    a : float
        势垒宽度.
    """
    if E >= V0:
        # E > V0 情形, 用 k' = sqrt(2m(E - V0))/ħ
        if E == V0:
            return 1.0 / (1.0 + m * V0 * a ** 2 / (2.0 * hbar ** 2))
        kp = np.sqrt(2.0 * m * (E - V0)) / hbar
        return 1.0 / (1.0 + V0 ** 2 * np.sin(kp * a) ** 2 / (4.0 * E * (E - V0)))
    kappa = np.sqrt(2.0 * m * (V0 - E)) / hbar
    sinh_term = np.sinh(kappa * a)
    return 1.0 / (1.0 + V0 ** 2 * sinh_term ** 2 / (4.0 * E * (V0 - E)))


# ---------------------------------------------------------------------------
# 动量空间分析 (傅里叶变换)
# ---------------------------------------------------------------------------
def momentum_space_wavefunction(psi, x, dx, hbar=HBAR):
    """
    用快速傅里叶变换 (FFT) 计算波函数的动量空间表示:

        φ(p) = (1/√(2πħ)) ∫ ψ(x) exp(-ipx/ħ) dx

    在离散网格上, 利用 numpy.fft.fft 实现, 并做归一化与频率轴映射.

    Parameters
    ----------
    psi : ndarray (complex)
        位置空间波函数.
    x : ndarray
        空间坐标.
    dx : float
        空间步长.

    Returns
    -------
    p : ndarray
        动量轴 (已 fftshift, 中心为 p=0).
    phi : ndarray (complex)
        动量空间波函数 (已 fftshift 并归一化).
    """
    Nx = len(x)
    # 动量网格: dp = 2πħ/(Nx*dx), 范围 [-πħ/dx, πħ/dx]
    dp = 2.0 * np.pi * hbar / (Nx * dx)
    p = np.fft.fftfreq(Nx, d=dx / (2.0 * np.pi * hbar))
    p = np.fft.fftshift(p)
    # FFT 并归一化: φ(p) = dx/√(2πħ) * FFT[ψ]
    phi = np.fft.fftshift(np.fft.fft(psi)) * dx / np.sqrt(2.0 * np.pi * hbar)
    return p, phi


def momentum_distribution(psi, x, dx, hbar=HBAR):
    """
    计算动量概率分布 |φ(p)|², 用于分析波包的能量成分.

    Returns
    -------
    p : ndarray
        动量轴.
    rho_p : ndarray
        动量概率密度 |φ(p)|².
    """
    p, phi = momentum_space_wavefunction(psi, x, dx, hbar)
    rho_p = np.abs(phi) ** 2
    return p, rho_p


def energy_distribution(psi, x, dx, m=MASS, hbar=HBAR):
    """
    计算能量分布 ρ(E), 其中 E = p²/(2m).

    这对于理解隧穿现象至关重要: 波包具有能量展宽,
    只有 E > V0 的成分可以经典透射, E < V0 的成分发生隧穿.

    Returns
    -------
    E : ndarray
        能量轴 (≥ 0).
    rho_E : ndarray
        能量概率密度.
    """
    p, rho_p = momentum_distribution(psi, x, dx, hbar)
    E = p ** 2 / (2.0 * m)
    # 变量代换: ρ(E) dE = ρ(p) dp, dp/dE = m/p = m/√(2mE)
    # 只取 p > 0 部分 (因为 E = p²/2m 是偶函数, 需合并 ±p)
    mask = p > 0
    E_pos = E[mask]
    rho_p_pos = rho_p[mask]
    # 合并 ±p 的贡献
    rho_p_combined = rho_p_pos + rho_p[::-1][mask]
    dp_dE = m / np.sqrt(2.0 * m * E_pos)
    rho_E = rho_p_combined * dp_dE
    return E_pos, rho_E
