"""
solvers.py
==========
一维含时与定态薛定谔方程的数值求解器.

本模块实现三类核心算法:
  1. Crank-Nicolson 隐式差分格式 —— 用于求解含时薛定谔方程 (TDSE)
  2. 含时势场 Crank-Nicolson 求解器 —— 处理 V(x,t) 显含时间的情况 (创新拓展)
  3. 有限差分矩阵对角化方法 —— 用于求解定态薛定谔方程 (TISE) 的本征值与本征态

Crank-Nicolson 格式具有二阶时间精度 O(Δt²) 与二阶空间精度 O(Δx²),
且对线性薛定谔方程而言是无条件概率守恒 (幺正) 的, 因此特别适合
长时演化问题.

核心物理方程:
    含时薛定谔方程: iħ ∂ψ/∂t = Ĥψ
    哈密顿量: Ĥ = -(ħ²/2m)∂²/∂x² + V(x)
    CN 离散: ψⁿ⁺¹ = (I + iHΔt/2ħ)⁻¹(I - iHΔt/2ħ)ψⁿ
    (Cayley 形式逼近, 严格幺正 → 概率守恒)

注意: 本模块不包含绘图逻辑, 仅返回数值结果.

AI 协作声明:
    本模块由作者在 AI 助手 (GLM, 智谱 AI) 辅助下编写.
    AI 主要协助: CN 格式的稀疏矩阵化实现, 含时势场求解器的半隐式处理.
    算法选择、精度分析由作者独立完成.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from physics import HBAR, MASS


# ---------------------------------------------------------------------------
# 构建二阶导数稀疏矩阵 (周期性边界除外, 这里采用 Dirichlet 边界)
# ---------------------------------------------------------------------------
def build_laplacian_matrix(Nx, dx):
    """
    构建一维二阶导数的稀疏有限差分矩阵 (Dirichlet 边界):

        D2[i, i]   = -2 / dx²
        D2[i, i±1] =  1 / dx²

    边界点 (i=0, i=Nx-1) 仅保留对角项, 等价于 ψ(边界)=0 的硬壁边界条件.

    Returns
    -------
    D2 : scipy.sparse.csr_matrix, shape (Nx, Nx)
    """
    main_diag = -2.0 * np.ones(Nx) / dx ** 2
    off_diag = np.ones(Nx - 1) / dx ** 2
    D2 = sp.diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1],
                  format="csr", dtype=complex)
    return D2


def build_hamiltonian_matrix(V, dx, hbar=HBAR, m=MASS):
    """
    构建哈密顿量矩阵 H = T + V:

        H = -(ħ² / 2m) D2 + diag(V)

    其中 D2 为二阶导数有限差分矩阵.

    Parameters
    ----------
    V : ndarray, shape (Nx,)
        势能数组.
    dx : float
        空间步长.

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (Nx, Nx)
        哈密顿量稀疏矩阵 (复数类型, 便于后续复数运算).
    """
    Nx = len(V)
    D2 = build_laplacian_matrix(Nx, dx)
    H = -(hbar ** 2 / (2.0 * m)) * D2 + sp.diags(V, offsets=0, format="csr", dtype=complex)
    return H


# ---------------------------------------------------------------------------
# Crank-Nicolson 演化器 (单步与多步)
# ---------------------------------------------------------------------------
class CrankNicolsonSolver:
    """
    Crank-Nicolson 隐式格式求解一维含时薛定谔方程:

        i ħ ∂ψ/∂t = H ψ

    离散化后, 时间演化算符为

        ψ(t + Δt) = (I + i H Δt / (2ħ))^(-1) (I - i H Δt / (2ħ)) ψ(t)

    该算符严格幺正, 因此概率 ∫|ψ|² dx 在演化过程中严格守恒
    (在数值精度范围内), 这是 CN 格式相对于显式 Euler 格式的最大优势.

    Attributes
    ----------
    H : sparse matrix
        哈密顿量矩阵.
    dt : float
        时间步长.
    Nx : int
        空间网格点数.
    """

    def __init__(self, V, dx, dt, hbar=HBAR, m=MASS):
        """
        Parameters
        ----------
        V : ndarray
            势能数组.
        dx : float
            空间步长.
        dt : float
            时间步长.
        """
        self.Nx = len(V)
        self.dx = dx
        self.dt = dt
        self.hbar = hbar
        # 构建哈密顿量
        self.H = build_hamiltonian_matrix(V, dx, hbar=hbar, m=m)
        # 演化矩阵的系数: α = i H Δt / (2ħ)
        alpha = 1j * dt / (2.0 * hbar)
        I = sp.eye(self.Nx, format="csr", dtype=complex)
        # A = I + α H  (需要求逆/解线性方程)
        self.A = (I + alpha * self.H).tocsc()
        # B = I - α H  (显式部分)
        self.B = (I - alpha * self.H).tocsc()
        # 预计算 LU 分解, 加速每步求解
        self.A_lu = spla.splu(self.A)

    def step(self, psi):
        """
        执行单步时间演化: ψ(t+Δt) = A^(-1) B ψ(t).

        Parameters
        ----------
        psi : ndarray (complex), shape (Nx,)
            当前时刻的波函数.

        Returns
        -------
        psi_next : ndarray (complex)
            下一时刻的波函数.
        """
        rhs = self.B.dot(psi)
        psi_next = self.A_lu.solve(rhs)
        return psi_next

    def evolve(self, psi0, Nt, save_every=1):
        """
        多步演化, 记录中间状态.

        Parameters
        ----------
        psi0 : ndarray
            初始波函数.
        Nt : int
            总时间步数.
        save_every : int
            每隔多少步保存一次波函数 (节省内存).

        Returns
        -------
        psi_history : ndarray, shape (N_saved, Nx)
            保存的波函数历史.
        t_history : ndarray
            对应的时间数组.
        """
        psi = psi0.copy()
        n_saved = Nt // save_every + 1
        psi_history = np.zeros((n_saved, self.Nx), dtype=complex)
        t_history = np.zeros(n_saved)
        psi_history[0] = psi
        t_history[0] = 0.0
        idx = 1
        for n in range(1, Nt + 1):
            psi = self.step(psi)
            if n % save_every == 0:
                psi_history[idx] = psi
                t_history[idx] = n * self.dt
                idx += 1
        return psi_history[:idx], t_history[:idx]


class TimeDependentCNSolver:
    """
    含时势场的 Crank-Nicolson 求解器 (创新拓展).

    当势能 V(x, t) 随时间变化时, 哈密顿量 H(t) 也是含时的.
    此时在每个时间步内, 用当前时刻的 H(t) 构造 CN 演化算符:

        ψ(t+Δt) = [I + iH(t)Δt/(2ħ)]⁻¹ [I - iH(t)Δt/(2ħ)] ψ(t)

    这是 "半隐式" 处理: 在每步内 H 视为常数, 但步与步之间 H 更新.
    对于缓慢变化的势场, 这种近似是二阶精度的.

    注意: 含时势场下能量不再守恒 (因 H 显含时间), 但概率仍守恒
    (CN 算符仍为幺正).

    Parameters
    ----------
    V_func : callable
        势能函数 V(x, t) -> ndarray.
    x : ndarray
        空间网格.
    dx : float
        空间步长.
    dt : float
        时间步长.
    """

    def __init__(self, V_func, x, dx, dt, hbar=HBAR, m=MASS):
        self.V_func = V_func
        self.x = x
        self.Nx = len(x)
        self.dx = dx
        self.dt = dt
        self.hbar = hbar
        self.m = m
        # 预计算动能矩阵部分 (不随时间变化)
        self.D2 = build_laplacian_matrix(self.Nx, dx)
        self.T_mat = -(hbar ** 2 / (2.0 * m)) * self.D2
        self.I = sp.eye(self.Nx, format="csr", dtype=complex)

    def step(self, psi, t):
        """单步演化, t 为当前时刻."""
        V_t = self.V_func(self.x, t)
        H_t = self.T_mat + sp.diags(V_t, offsets=0, format="csr", dtype=complex)
        alpha = 1j * self.dt / (2.0 * self.hbar)
        A = (self.I + alpha * H_t).tocsc()
        B = (self.I - alpha * H_t).tocsc()
        rhs = B.dot(psi)
        psi_next = spla.spsolve(A, rhs)
        return psi_next

    def evolve(self, psi0, Nt, save_every=1):
        """多步演化."""
        psi = psi0.copy()
        n_saved = Nt // save_every + 1
        psi_history = np.zeros((n_saved, self.Nx), dtype=complex)
        t_history = np.zeros(n_saved)
        psi_history[0] = psi
        t_history[0] = 0.0
        idx = 1
        for n in range(1, Nt + 1):
            t = (n - 0.5) * self.dt  # 用步中时刻的 H
            psi = self.step(psi, t)
            if n % save_every == 0:
                psi_history[idx] = psi
                t_history[idx] = n * self.dt
                idx += 1
        return psi_history[:idx], t_history[:idx]


# ---------------------------------------------------------------------------
# 定态薛定谔方程求解: 有限差分矩阵对角化
# ---------------------------------------------------------------------------
def solve_stationary_states(V, dx, num_states=10, hbar=HBAR, m=MASS):
    """
    用有限差分矩阵对角化方法求解定态薛定谔方程:

        H φ_n = E_n φ_n

    将哈密顿量离散化为稀疏矩阵后, 用 Lanczos 算法 (scipy eigsh)
    求解最低的若干个本征值与本征态.

    Parameters
    ----------
    V : ndarray
        势能数组.
    dx : float
        空间步长.
    num_states : int
        需要计算的本征态数目.
    hbar, m : float
        物理常数.

    Returns
    -------
    energies : ndarray, shape (num_states,)
        本征能量数组 (升序).
    eigenstates : ndarray, shape (num_states, Nx)
        对应的本征波函数 (已归一化).
    """
    H = build_hamiltonian_matrix(V, dx, hbar=hbar, m=m)
    # 转换为实对称矩阵 (因为 H 在 Dirichlet 边界下是实对称的)
    H_real = H.real
    # 求解最小的 num_states 个本征值 (用 'SA' = Smallest in Algebraic)
    # 注意: eigsh 要求矩阵为对称/厄米, 这里 H_real 满足
    energies, eigenstates = spla.eigsh(H_real, k=num_states, which='SA')
    # 排序
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    eigenstates = eigenstates[:, sort_idx].T  # shape (num_states, Nx)
    # 归一化
    for i in range(num_states):
        norm = np.sqrt(np.sum(np.abs(eigenstates[i]) ** 2) * dx)
        eigenstates[i] = eigenstates[i] / norm
    return energies, eigenstates
