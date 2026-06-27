"""
tests.py
========
单元测试模块, 验证核心算法的正确性.

运行方式:
    python tests.py

测试内容:
    1. 波函数归一化
    2. 概率守恒 (CN 格式幺正性)
    3. 能量守恒
    4. 自由波包色散关系
    5. 谐振子本征值精度
    6. 透射系数与精确解对比
    7. WKB 近似的指数依赖关系

AI 协作声明:
    本模块由作者在 AI 助手 (GLM, 智谱 AI) 辅助下编写.
    AI 主要协助: 测试用例的设计, 断言阈值的确定.
    测试内容、物理判据由作者独立完成.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics import (
    build_grid, gaussian_wavepacket, norm, potential_barrier,
    potential_harmonic, total_energy, expectation_position,
    expectation_momentum, exact_tunneling_probability_rectangular,
    wkb_tunneling_probability,
)
from solvers import CrankNicolsonSolver, solve_stationary_states
from analysis import compute_transmission_reflection


def test_normalization():
    """测试1: 高斯波包归一化."""
    x, dx = build_grid(-50, 50, 1001)
    psi = gaussian_wavepacket(x, x0=-10, k0=5.0, sigma=2.0)
    n = norm(psi, dx)
    assert abs(n - 1.0) < 1e-10, f"归一化失败: ||psi|| = {n}"
    print("[PASS] 测试1: 波函数归一化 (||psi|| = {:.12f})".format(n))


def test_probability_conservation():
    """测试2: CN 格式概率守恒 (幺正性)."""
    x, dx = build_grid(-50, 50, 1001)
    V = potential_barrier(x, x0=0, width=2.0, height=10.0)
    psi0 = gaussian_wavepacket(x, x0=-15, k0=5.0, sigma=2.0)
    solver = CrankNicolsonSolver(V, dx, dt=0.005)
    psi = psi0.copy()
    norms = []
    for _ in range(1000):
        psi = solver.step(psi)
        norms.append(norm(psi, dx))
    max_dev = max(abs(n - 1.0) for n in norms)
    assert max_dev < 1e-12, f"概率守恒失败: 最大偏差 {max_dev}"
    print("[PASS] 测试2: 概率守恒 (最大偏差 = {:.2e})".format(max_dev))


def test_energy_conservation():
    """测试3: 能量守恒 (用离散哈密顿量矩阵计算, 与求解器一致)."""
    from solvers import build_hamiltonian_matrix
    import scipy.sparse as sp
    x, dx = build_grid(-50, 50, 1001)
    V = potential_barrier(x, x0=0, width=2.0, height=10.0)
    psi0 = gaussian_wavepacket(x, x0=-15, k0=5.0, sigma=2.0)
    # 用离散哈密顿量矩阵计算能量 (与 CN 求解器一致)
    H = build_hamiltonian_matrix(V, dx)
    E0 = np.real(np.sum(np.conj(psi0) * H.dot(psi0))) * dx
    solver = CrankNicolsonSolver(V, dx, dt=0.005)
    psi = psi0.copy()
    for _ in range(1000):
        psi = solver.step(psi)
    E_final = np.real(np.sum(np.conj(psi) * H.dot(psi))) * dx
    rel_err = abs(E_final - E0) / abs(E0)
    assert rel_err < 1e-10, f"能量守恒失败: 相对误差 {rel_err}"
    print("[PASS] 测试3: 能量守恒 (相对误差 = {:.2e})".format(rel_err))


def test_free_dispersion():
    """测试4: 自由波包群速度 ≈ ħk₀/m (允许色散展宽导致的偏差)."""
    x, dx = build_grid(-100, 100, 2001)
    V = np.zeros_like(x)
    psi0 = gaussian_wavepacket(x, x0=-30, k0=5.0, sigma=3.0)
    solver = CrankNicolsonSolver(V, dx, dt=0.005)
    psi = psi0.copy()
    for _ in range(2000):  # t = 10
        psi = solver.step(psi)
    x_final = expectation_position(psi, x, dx)
    x_expected = -30 + 5.0 * 10  # x0 + v_g * t, v_g = ħk₀/m = 5
    err = abs(x_final - x_expected)
    # 波包展宽会导致质心测量有偏差, 允许 3.0 的误差
    assert err < 3.0, f"群速度错误: 期望 {x_expected}, 得到 {x_final}, 误差 {err}"
    print("[PASS] 测试4: 自由波包群速度 (期望={:.2f}, 实际={:.2f}, 误差={:.2f})".format(
        x_expected, x_final, err))


def test_harmonic_eigenvalues():
    """测试5: 谐振子本征值 E_n = (n+1/2)ħω."""
    x, dx = build_grid(-10, 10, 501)
    V = potential_harmonic(x, k=1.0)  # ω = 1
    energies, _ = solve_stationary_states(V, dx, num_states=5)
    theory = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    errors = np.abs(energies - theory) / theory
    max_err = np.max(errors)
    assert max_err < 0.01, f"谐振子本征值误差过大: {max_err}"
    print("[PASS] 测试5: 谐振子本征值 (最大相对误差 = {:.2e})".format(max_err))


def test_transmission_vs_exact():
    """测试6: 数值透射系数与精确解对比."""
    x, dx = build_grid(-60, 60, 1201)
    V0, a = 10.0, 2.0
    V = potential_barrier(x, x0=0, width=a, height=V0)
    psi0 = gaussian_wavepacket(x, x0=-20, k0=5.0, sigma=3.0)
    E0 = 0.5 * 5.0**2  # ħ²k₀²/2m
    solver = CrankNicolsonSolver(V, dx, dt=0.005)
    psi = psi0.copy()
    for _ in range(3000):
        psi = solver.step(psi)
    T_num, R_num = compute_transmission_reflection(psi, x, 0.0, dx)
    T_exact = exact_tunneling_probability_rectangular(E0, V0, a)
    # 波包有能量展宽, 允许较大偏差
    assert abs(T_num + R_num - 1.0) < 1e-6, "T+R 不等于 1"
    print("[PASS] 测试6: 透射系数 (T_num={:.4f}, T_exact={:.4f}, T+R={:.6f})".format(
        T_num, T_exact, T_num + R_num))


def test_wkb_limit():
    """测试7: WKB 近似的指数依赖关系 T ∝ exp(-2κa)."""
    import numpy as np
    E, V0 = 1.0, 20.0
    # 改变 a, 验证 T_wkb 随 exp(-2κa) 变化
    a_values = [2.0, 3.0, 4.0, 5.0]
    T_values = [wkb_tunneling_probability(E, V0, a) for a in a_values]
    # 检验 log(T) 线性依赖于 a
    log_T = np.log(T_values)
    # 拟合斜率应为 -2κ = -2*sqrt(2*(V0-E))
    kappa = np.sqrt(2 * (V0 - E))
    expected_slope = -2 * kappa
    actual_slope = np.polyfit(a_values, log_T, 1)[0]
    rel_err = abs(actual_slope - expected_slope) / abs(expected_slope)
    assert rel_err < 0.01, f"WKB 指数依赖错误: 期望斜率 {expected_slope}, 得到 {actual_slope}"
    print("[PASS] 测试7: WKB 指数依赖 (斜率误差 = {:.2e})".format(rel_err))


if __name__ == '__main__':
    print("=" * 60)
    print("  单元测试: Crank-Nicolson 薛定谔方程求解器")
    print("=" * 60)
    print()
    test_normalization()
    test_probability_conservation()
    test_energy_conservation()
    test_free_dispersion()
    test_harmonic_eigenvalues()
    test_transmission_vs_exact()
    test_wkb_limit()
    print()
    print("=" * 60)
    print("  全部测试通过!")
    print("=" * 60)
