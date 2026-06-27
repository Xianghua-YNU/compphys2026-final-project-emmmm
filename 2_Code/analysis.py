"""
analysis.py
===========
物理量分析与后处理模块.

本模块负责从原始波函数数据中提取物理可观测量:
  - 隧穿概率 (透射系数) 与反射系数
  - 概率密度分布
  - 能量与概率守恒检验
  - 波包质心轨迹与展宽
  - 隧穿概率随势垒参数的变化扫描

注意: 本模块不包含绘图逻辑, 仅返回数值结果.

AI 协作声明:
    本模块由作者在 AI 助手 (GLM, 智谱 AI) 辅助下编写.
    AI 主要协助: 守恒检验的离散哈密顿量计算, 参数扫描的组织结构.
    物理量定义、误差分析由作者独立完成.
"""

import numpy as np
from physics import (
    norm,
    total_energy,
    expectation_position,
    expectation_kinetic,
    expectation_potential,
    wkb_tunneling_probability,
    exact_tunneling_probability_rectangular,
)


def compute_transmission_reflection(psi, x, barrier_x0=0.0, dx=None):
    """
    根据最终波函数计算透射系数 T 与反射系数 R.

    定义: 在势垒右侧 (x > barrier_x0 + a/2) 的概率为透射部分,
          左侧 (x < barrier_x0 - a/2) 的概率为反射部分.
    这里简化为以势垒中心为分界, 适用于波包已远离势垒的情形.

    Parameters
    ----------
    psi : ndarray
        最终时刻的波函数.
    x : ndarray
        空间坐标.
    barrier_x0 : float
        势垒中心位置.
    dx : float, optional
        空间步长, 若为 None 则自动计算.

    Returns
    -------
    T : float
        透射系数 (右侧概率).
    R : float
        反射系数 (左侧概率).
    """
    if dx is None:
        dx = x[1] - x[0]
    prob_density = np.abs(psi) ** 2 * dx
    T = np.sum(prob_density[x > barrier_x0])
    R = np.sum(prob_density[x < barrier_x0])
    return T, R


def scan_barrier_height(psi0, x, dx, dt, Nt, heights, barrier_width=2.0,
                        barrier_x0=0.0, solver_class=None):
    """
    扫描势垒高度, 计算每个高度下的透射系数.

    Parameters
    ----------
    psi0 : ndarray
        初始波函数 (每次扫描前会重新归一化).
    x, dx, dt : 网格与时间参数.
    Nt : int
        演化步数.
    heights : ndarray
        势垒高度数组.
    barrier_width : float
        势垒宽度 (固定).
    barrier_x0 : float
        势垒中心.
    solver_class : class, optional
        求解器类, 默认为 CrankNicolsonSolver.

    Returns
    -------
    results : dict
        包含 'heights', 'T_numerical', 'T_wkb', 'T_exact' 等数组.
    """
    if solver_class is None:
        from solvers import CrankNicolsonSolver
        solver_class = CrankNicolsonSolver
    from physics import potential_barrier

    # 入射能量 (由初始波包计算)
    E_incident = expectation_kinetic(psi0, x, dx) + \
                 expectation_potential(psi0, np.zeros_like(x), dx)
    # 简化: 取动能作为入射能量 (势垒外 V=0)
    E_incident = expectation_kinetic(psi0, x, dx)

    T_num = np.zeros(len(heights))
    T_wkb = np.zeros(len(heights))
    T_exact = np.zeros(len(heights))

    for i, h in enumerate(heights):
        V = potential_barrier(x, x0=barrier_x0, width=barrier_width, height=h)
        solver = solver_class(V, dx, dt)
        psi_final = psi0.copy()
        for _ in range(Nt):
            psi_final = solver.step(psi_final)
        T, R = compute_transmission_reflection(psi_final, x, barrier_x0, dx)
        T_num[i] = T
        T_wkb[i] = wkb_tunneling_probability(E_incident, h, barrier_width)
        T_exact[i] = exact_tunneling_probability_rectangular(
            E_incident, h, barrier_width)

    return {
        'heights': heights,
        'T_numerical': T_num,
        'T_wkb': T_wkb,
        'T_exact': T_exact,
        'E_incident': E_incident,
    }


def scan_barrier_width(psi0, x, dx, dt, Nt, widths, barrier_height=10.0,
                       barrier_x0=0.0, solver_class=None):
    """
    扫描势垒宽度, 计算每个宽度下的透射系数.

    Parameters
    ----------
    widths : ndarray
        势垒宽度数组.
    barrier_height : float
        势垒高度 (固定).

    Returns
    -------
    results : dict
        同 scan_barrier_height.
    """
    if solver_class is None:
        from solvers import CrankNicolsonSolver
        solver_class = CrankNicolsonSolver
    from physics import potential_barrier

    E_incident = expectation_kinetic(psi0, x, dx)

    T_num = np.zeros(len(widths))
    T_wkb = np.zeros(len(widths))
    T_exact = np.zeros(len(widths))

    for i, w in enumerate(widths):
        V = potential_barrier(x, x0=barrier_x0, width=w, height=barrier_height)
        solver = solver_class(V, dx, dt)
        psi_final = psi0.copy()
        for _ in range(Nt):
            psi_final = solver.step(psi_final)
        T, R = compute_transmission_reflection(psi_final, x, barrier_x0, dx)
        T_num[i] = T
        T_wkb[i] = wkb_tunneling_probability(E_incident, barrier_height, w)
        T_exact[i] = exact_tunneling_probability_rectangular(
            E_incident, barrier_height, w)

    return {
        'widths': widths,
        'T_numerical': T_num,
        'T_wkb': T_wkb,
        'T_exact': T_exact,
        'E_incident': E_incident,
    }


def check_conservation(psi_history, x, V, dx):
    """
    检验概率守恒与能量守恒.

    能量计算使用与 Crank-Nicolson 求解器相同的离散哈密顿量矩阵,
    因此 <ψ|H|ψ> 应在演化过程中严格守恒 (CN 格式是幺正的).

    Parameters
    ----------
    psi_history : ndarray, shape (Nt, Nx)
        波函数历史.
    x : ndarray
        空间坐标.
    V : ndarray
        势能数组.
    dx : float
        空间步长.

    Returns
    -------
    norms : ndarray
        每个时刻的归一化系数.
    energies : ndarray
        每个时刻的总能量.
    """
    from solvers import build_hamiltonian_matrix
    Nt = psi_history.shape[0]
    norms = np.zeros(Nt)
    energies = np.zeros(Nt)
    # 构建离散哈密顿量矩阵 (与求解器一致)
    H = build_hamiltonian_matrix(V, dx)
    for n in range(Nt):
        psi = psi_history[n]
        norms[n] = norm(psi, dx)
        # <ψ|H|ψ> = ψ† H ψ * dx (注意 H 已包含 1/dx² 因子, 需乘 dx 积分)
        Hpsi = H.dot(psi)
        energies[n] = np.real(np.sum(np.conj(psi) * Hpsi)) * dx
    return norms, energies


def wavepacket_centroid_std(psi_history, x, dx):
    """
    计算波包质心 <x> 与位置标准差 σ_x 随时间的演化.

    Parameters
    ----------
    psi_history : ndarray, shape (Nt, Nx)
    x : ndarray
    dx : float

    Returns
    -------
    centroids : ndarray
        质心位置 <x>(t).
    stds : ndarray
        位置标准差 σ_x(t) = sqrt(<x²> - <x>²).
    """
    Nt = psi_history.shape[0]
    centroids = np.zeros(Nt)
    stds = np.zeros(Nt)
    for n in range(Nt):
        psi = psi_history[n]
        rho = np.abs(psi) ** 2
        rho_norm = rho * dx
        x_mean = np.sum(x * rho_norm)
        x2_mean = np.sum(x ** 2 * rho_norm)
        centroids[n] = x_mean
        stds[n] = np.sqrt(max(x2_mean - x_mean ** 2, 0.0))
    return centroids, stds
