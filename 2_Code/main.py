"""
main.py
=======
项目主入口: 一维含时薛定谔方程的 Crank-Nicolson 数值模拟.

本脚本组织整个模拟流程, 包括:
  1. 设置物理参数与网格
  2. 演化高斯波包在方势垒中的传播 (含反射与隧穿)
  3. 扫描势垒高度/宽度, 计算透射系数
  4. 检验概率与能量守恒
  5. 求解谐振子与双势阱的定态本征值问题
  6. 生成所有论文图片

运行方式:
    python main.py

输出:
    所有图片保存至 ../1_Paper/assets/ 目录
    (相对于本脚本所在目录)

AI 协作声明:
    本代码由作者在 AI 助手 (GLM) 辅助下编写, AI 主要协助:
    - Crank-Nicolson 格式的矩阵化实现
    - WKB 与精确透射系数公式的核对
    - Matplotlib 可视化代码的优化
    物理模型设计、参数选取、结果分析由作者完成.
"""

import os
import sys
import time
import numpy as np

# 添加当前目录到路径 (便于作为脚本运行)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics import (
    build_grid,
    potential_barrier,
    potential_harmonic,
    potential_double_well,
    potential_time_dependent_barrier,
    gaussian_wavepacket,
    expectation_kinetic,
    momentum_distribution,
    energy_distribution,
    total_energy,
    exact_tunneling_probability_rectangular,
    HBAR,
)
from solvers import CrankNicolsonSolver, TimeDependentCNSolver, solve_stationary_states
from analysis import (
    compute_transmission_reflection,
    scan_barrier_height,
    scan_barrier_width,
    check_conservation,
    wavepacket_centroid_std,
)
from plot_utils import (
    plot_initial_state,
    plot_wavepacket_evolution,
    plot_transmission_vs_height,
    plot_transmission_vs_width,
    plot_conservation,
    plot_harmonic_eigenstates,
    plot_double_well_eigenstates,
    plot_centroid_trajectory,
    plot_final_states_comparison,
    plot_momentum_space,
    plot_energy_distribution,
    plot_transmission_error,
    plot_time_dependent_barrier,
    plot_convergence_analysis,
)


# ===========================================================================
# 全局参数 (集中定义, 避免魔法数字)
# ===========================================================================
PARAMS = {
    # 空间网格 (扩大区间以避免边界反射影响)
    'x_min': -80.0,
    'x_max': 80.0,
    'Nx': 1601,

    # 时间步长与总步数
    'dt': 0.005,
    'Nt_main': 3000,        # 主演化步数 (总时间 15, 波包不会到达边界)
    'save_every': 15,       # 每隔多少步保存一次

    # 初始波包参数
    'psi_x0': -20.0,        # 初始位置 (左移以增加演化空间)
    'psi_k0': 5.0,          # 初始波数 (动量 p = ħ k0)
    'psi_sigma': 2.0,       # 波包宽度 (增大以减小动量展宽, 更接近平面波)

    # 势垒参数 (主模拟) — E_inc ≈ 12.5, V0 = 15 > E, 隧穿区域
    'barrier_x0': 0.0,
    'barrier_width': 1.5,
    'barrier_height': 15.0,

    # 扫描参数
    'height_scan': np.linspace(3.0, 30.0, 28),
    'width_scan': np.linspace(0.3, 4.0, 20),

    # 谐振子参数
    'harmonic_k': 1.0,

    # 双势阱参数
    'dw_a': 2.0,
    'dw_b': 1.0,

    # 含时势垒参数 (创新拓展)
    'td_omega': 3.0,       # 振荡频率
    'td_amplitude': 0.3,   # 振荡振幅 (相对值)
}


def section(title):
    """打印分节标题."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_main_simulation(assets_dir, params):
    """主模拟: 高斯波包在方势垒中的传播."""
    section("1. 主模拟: 高斯波包在方势垒中的传播")

    # 构建网格与势场
    x, dx = build_grid(params['x_min'], params['x_max'], params['Nx'])
    V = potential_barrier(x, x0=params['barrier_x0'],
                          width=params['barrier_width'],
                          height=params['barrier_height'])
    # 初始波包
    psi0 = gaussian_wavepacket(x, x0=params['psi_x0'], k0=params['psi_k0'],
                               sigma=params['psi_sigma'])

    # 入射能量 (动能)
    E_inc = expectation_kinetic(psi0, x, dx)
    print(f"  网格: Nx = {params['Nx']}, dx = {dx:.4f}, 区间 = [{params['x_min']}, {params['x_max']}]")
    print(f"  时间步长 dt = {params['dt']}, 总步数 Nt = {params['Nt_main']}")
    print(f"  入射动能 E_inc = {E_inc:.4f}")
    print(f"  势垒: V0 = {params['barrier_height']}, a = {params['barrier_width']}")
    print(f"  E_inc / V0 = {E_inc / params['barrier_height']:.4f} (隧穿区域)")

    # 演化
    t_start = time.time()
    solver = CrankNicolsonSolver(V, dx, params['dt'])
    psi_history, t_history = solver.evolve(psi0, params['Nt_main'],
                                            save_every=params['save_every'])
    t_elapsed = time.time() - t_start
    print(f"  演化完成, 耗时 {t_elapsed:.2f} s, 保存 {len(t_history)} 个时刻")

    # 最终透射/反射
    psi_final = psi_history[-1]
    T, R = compute_transmission_reflection(psi_final, x,
                                            barrier_x0=params['barrier_x0'], dx=dx)
    print(f"  透射系数 T = {T:.4f}, 反射系数 R = {R:.4f}, T + R = {T + R:.4f}")

    # 绘图
    plot_initial_state(x, psi0, V,
                       os.path.join(assets_dir, 'fig1_initial_state.png'),
                       barrier_x0=params['barrier_x0'],
                       barrier_width=params['barrier_width'])
    print("  [图1] 初始波包与势垒 -> fig1_initial_state.png")

    plot_wavepacket_evolution(x, t_history, psi_history, V,
                              os.path.join(assets_dir, 'fig2_evolution.png'))
    print("  [图2] 波包时空演化 -> fig2_evolution.png")

    # 守恒检验
    norms, energies = check_conservation(psi_history, x, V, dx)
    plot_conservation(t_history, norms, energies,
                      os.path.join(assets_dir, 'fig5_conservation.png'))
    print(f"  [图5] 守恒检验 -> fig5_conservation.png")
    print(f"       归一化偏差: {np.max(np.abs(norms - 1.0)):.2e}")
    print(f"       能量偏差: {np.max(np.abs(energies - energies[0])):.2e}")

    # 质心轨迹
    centroids, stds = wavepacket_centroid_std(psi_history, x, dx)
    plot_centroid_trajectory(t_history, centroids, stds,
                             os.path.join(assets_dir, 'fig8_centroid.png'),
                             barrier_x0=params['barrier_x0'])
    print("  [图8] 质心轨迹 -> fig8_centroid.png")

    return x, dx, psi0, V, psi_history, t_history


def run_height_scan(assets_dir, params, x, dx, psi0):
    """扫描势垒高度."""
    section("2. 扫描势垒高度")
    print(f"  高度范围: {params['height_scan'][0]:.1f} ~ {params['height_scan'][-1]:.1f}")
    print(f"  固定宽度: {params['barrier_width']}")

    # 扫描时使用较少步数 (波包穿过势垒即可)
    Nt_scan = 2500
    results = scan_barrier_height(
        psi0, x, dx, params['dt'], Nt_scan,
        heights=params['height_scan'],
        barrier_width=params['barrier_width'],
        barrier_x0=params['barrier_x0'])
    print(f"  入射能量 E_inc = {results['E_incident']:.4f}")
    print(f"  数值 T 范围: {results['T_numerical'].min():.2e} ~ {results['T_numerical'].max():.4f}")

    plot_transmission_vs_height(results,
                                os.path.join(assets_dir, 'fig3_T_vs_height.png'))
    print("  [图3] 透射系数 vs 势垒高度 -> fig3_T_vs_height.png")
    return results


def run_width_scan(assets_dir, params, x, dx, psi0):
    """扫描势垒宽度."""
    section("3. 扫描势垒宽度")
    print(f"  宽度范围: {params['width_scan'][0]:.2f} ~ {params['width_scan'][-1]:.2f}")
    print(f"  固定高度: {params['barrier_height']}")

    Nt_scan = 2500
    results = scan_barrier_width(
        psi0, x, dx, params['dt'], Nt_scan,
        widths=params['width_scan'],
        barrier_height=params['barrier_height'],
        barrier_x0=params['barrier_x0'])
    print(f"  入射能量 E_inc = {results['E_incident']:.4f}")
    print(f"  数值 T 范围: {results['T_numerical'].min():.2e} ~ {results['T_numerical'].max():.4f}")

    plot_transmission_vs_width(results,
                               os.path.join(assets_dir, 'fig4_T_vs_width.png'))
    print("  [图4] 透射系数 vs 势垒宽度 -> fig4_T_vs_width.png")
    return results


def run_final_states_comparison(assets_dir, params, x, dx, psi0):
    """对比不同势垒参数下的最终波函数."""
    section("4. 不同势垒参数下最终波函数对比")

    configs = [
        ('V0=10, a=1.5', 10.0, 1.5),
        ('V0=15, a=1.5', 15.0, 1.5),
        ('V0=20, a=1.5', 20.0, 1.5),
        ('V0=15, a=3.0', 15.0, 3.0),
    ]
    psi_list = []
    V_list = []
    labels = []
    Nt_comp = 2800
    for label, h, w in configs:
        V = potential_barrier(x, x0=params['barrier_x0'], width=w, height=h)
        solver = CrankNicolsonSolver(V, dx, params['dt'])
        psi_f = psi0.copy()
        for _ in range(Nt_comp):
            psi_f = solver.step(psi_f)
        psi_list.append(psi_f)
        V_list.append(V)
        labels.append(label)
        T, R = compute_transmission_reflection(psi_f, x, params['barrier_x0'], dx)
        print(f"  {label}: T = {T:.4f}, R = {R:.4f}")

    plot_final_states_comparison(x, psi_list, V_list, labels,
                                 os.path.join(assets_dir, 'fig9_final_comparison.png'))
    print("  [图9] 最终波函数对比 -> fig9_final_comparison.png")


def run_harmonic_eigenstates(assets_dir, params):
    """谐振子定态本征值问题."""
    section("5. 谐振子定态本征值问题")
    # 谐振子用更窄的网格
    x_h, dx_h = build_grid(x_min=-10.0, x_max=10.0, Nx=801)
    V_h = potential_harmonic(x_h, k=params['harmonic_k'])
    energies_h, states_h = solve_stationary_states(V_h, dx_h, num_states=6)
    print("  谐振子本征能量 (数值 vs 理论 (n+1/2)ħω):")
    omega = np.sqrt(params['harmonic_k'])
    for n, E in enumerate(energies_h):
        E_theory = (n + 0.5) * HBAR * omega
        print(f"    n={n}: E_num = {E:.6f}, E_theory = {E_theory:.6f}, "
              f"误差 = {abs(E - E_theory):.2e}")

    plot_harmonic_eigenstates(x_h, V_h, energies_h, states_h,
                              os.path.join(assets_dir, 'fig6_harmonic.png'),
                              n_show=5)
    print("  [图6] 谐振子本征态 -> fig6_harmonic.png")
    return energies_h


def run_double_well_eigenstates(assets_dir, params):
    """双势阱定态本征值问题."""
    section("6. 双势阱定态本征值问题")
    x_d, dx_d = build_grid(x_min=-5.0, x_max=5.0, Nx=601)
    V_d = potential_double_well(x_d, a=params['dw_a'], b=params['dw_b'])
    energies_d, states_d = solve_stationary_states(V_d, dx_d, num_states=8)
    print("  双势阱最低 8 个本征能量:")
    for n, E in enumerate(energies_d):
        print(f"    n={n}: E = {E:.6f}")
    # 计算基态与第一激发态的劈裂
    if len(energies_d) >= 2:
        splitting = energies_d[1] - energies_d[0]
        print(f"  基态-第一激发态能级劈裂 ΔE = {splitting:.6f}")

    plot_double_well_eigenstates(x_d, V_d, energies_d, states_d,
                                 os.path.join(assets_dir, 'fig7_double_well.png'),
                                 n_show=6)
    print("  [图7] 双势阱本征态 -> fig7_double_well.png")
    return energies_d


def run_momentum_energy_analysis(assets_dir, params, x, dx, psi0, psi_final):
    """动量空间与能量分布分析 (深化物理洞见)."""
    section("7. 动量空间与能量分布分析")

    # 动量空间分布
    p, rho_p_initial = momentum_distribution(psi0, x, dx)
    _, rho_p_final = momentum_distribution(psi_final, x, dx)
    plot_momentum_space(p, rho_p_initial, rho_p_final,
                        os.path.join(assets_dir, 'fig10_momentum_space.png'),
                        k0=params['psi_k0'], V0=params['barrier_height'])
    print("  [图10] 动量空间分布 -> fig10_momentum_space.png")

    # 能量分布
    E, rho_E = energy_distribution(psi0, x, dx)
    E_mean = total_energy(psi0, x,
                          potential_barrier(x, x0=params['barrier_x0'],
                                            width=params['barrier_width'],
                                            height=params['barrier_height']),
                          dx)
    plot_energy_distribution(E, rho_E,
                             os.path.join(assets_dir, 'fig11_energy_distribution.png'),
                             V0=params['barrier_height'], E_mean=E_mean)
    print("  [图11] 能量分布 -> fig11_energy_distribution.png")

    # 计算能量分布中 E > V0 和 E < V0 的比例
    mask_tunnel = E < params['barrier_height']
    mask_classical = E >= params['barrier_height']
    dE = E[1] - E[0]
    P_tunnel = np.sum(rho_E[mask_tunnel]) * dE
    P_classical = np.sum(rho_E[mask_classical]) * dE
    print(f"  能量分布分析:")
    print(f"    E < V0 (隧穿成分) 比例: {P_tunnel:.4f}")
    print(f"    E >= V0 (经典透射成分) 比例: {P_classical:.4f}")
    print(f"  这解释了为何数值透射系数高于单能 WKB 预测")


def run_error_analysis(assets_dir, params, x, dx, psi0, height_results, width_results):
    """透射系数误差分析 (数值 vs 理论)."""
    section("8. 透射系数误差分析")

    E_inc = height_results['E_incident']
    V0 = params['barrier_height']
    a = params['barrier_width']

    # 高度扫描的误差分析
    T_exact_height = np.array([
        exact_tunneling_probability_rectangular(E_inc, h, a)
        for h in height_results['heights']
    ])
    plot_transmission_error(
        height_results['heights'],
        height_results['T_numerical'],
        T_exact_height,
        os.path.join(assets_dir, 'fig12_error_height.png'),
        xlabel=r'势垒高度 $V_0$',
        title='透射系数误差分析 (扫描势垒高度)')
    print("  [图12] 误差分析 (高度) -> fig12_error_height.png")

    # 宽度扫描的误差分析
    T_exact_width = np.array([
        exact_tunneling_probability_rectangular(E_inc, V0, w)
        for w in width_results['widths']
    ])
    plot_transmission_error(
        width_results['widths'],
        width_results['T_numerical'],
        T_exact_width,
        os.path.join(assets_dir, 'fig13_error_width.png'),
        xlabel=r'势垒宽度 $a$',
        title='透射系数误差分析 (扫描势垒宽度)')
    print("  [图13] 误差分析 (宽度) -> fig13_error_width.png")

    # 打印关键误差统计
    rel_err_height = np.abs(height_results['T_numerical'] - T_exact_height) / np.maximum(T_exact_height, 1e-15)
    rel_err_width = np.abs(width_results['T_numerical'] - T_exact_width) / np.maximum(T_exact_width, 1e-15)
    print(f"  高度扫描: 平均相对误差 = {np.mean(rel_err_height):.4f}, 最大 = {np.max(rel_err_height):.4f}")
    print(f"  宽度扫描: 平均相对误差 = {np.mean(rel_err_width):.4f}, 最大 = {np.max(rel_err_width):.4f}")


def run_time_dependent_barrier(assets_dir, params, x, dx, psi0):
    """含时振荡势垒模拟 (创新拓展)."""
    section("9. 含时振荡势垒模拟 (创新拓展)")

    # 静态势垒 (对照)
    V_static = potential_barrier(x, x0=params['barrier_x0'],
                                 width=params['barrier_width'],
                                 height=params['barrier_height'])
    solver_static = CrankNicolsonSolver(V_static, dx, params['dt'])
    psi_hist_static, t_hist = solver_static.evolve(
        psi0, params['Nt_main'], save_every=params['save_every'])

    # 含时振荡势垒
    def V_func(x_arr, t):
        return potential_time_dependent_barrier(
            x_arr, t, x0=params['barrier_x0'],
            width=params['barrier_width'],
            height0=params['barrier_height'],
            omega=params['td_omega'])

    solver_td = TimeDependentCNSolver(V_func, x, dx, params['dt'])
    psi_hist_td, t_hist_td = solver_td.evolve(
        psi0, params['Nt_main'], save_every=params['save_every'])

    # 最终透射系数对比
    T_static, _ = compute_transmission_reflection(
        psi_hist_static[-1], x, params['barrier_x0'], dx)
    T_td, _ = compute_transmission_reflection(
        psi_hist_td[-1], x, params['barrier_x0'], dx)
    print(f"  静态势垒最终透射系数: T = {T_static:.4f}")
    print(f"  含时振荡势垒最终透射系数: T = {T_td:.4f}")
    print(f"  振荡频率 omega = {params['td_omega']}, 振幅 = {params['td_amplitude']*100}%")
    print(f"  透射增强因子: {T_td / T_static:.2f}")

    plot_time_dependent_barrier(
        x, psi_hist_static, psi_hist_td, t_hist,
        os.path.join(assets_dir, 'fig14_time_dependent.png'),
        barrier_x0=params['barrier_x0'])
    print("  [图14] 含时势垒结果 -> fig14_time_dependent.png")


def run_convergence_analysis(assets_dir, params):
    """数值收敛性分析 (验证 O(Δt²) 和 O(Δx²) 收敛阶)."""
    section("10. 数值收敛性分析")

    # 收敛性分析专用参数 (局部定义, 避免污染全局 PARAMS)
    CONV_PARAMS = {
        'x_range': (-40.0, 40.0),
        'total_time': 10.0,           # 固定总演化时间
        'dt_values': [0.02, 0.01, 0.005, 0.0025, 0.001],  # 时间步长序列
        'dt_fixed': 0.002,            # 空间收敛性分析用的固定 dt
        'Nx_values': [201, 401, 801, 1601],  # 空间网格点数序列
        'Nx_reference': 3201,         # 参考解的最细网格
        'barrier_width': 1.5,
        'barrier_height': 15.0,
        'psi_x0': -15.0,
        'psi_k0': 5.0,
        'psi_sigma': 2.0,
    }

    # 时间收敛性: 固定 dx, 改变 dt
    x_conv, dx_conv = build_grid(CONV_PARAMS['x_range'][0], CONV_PARAMS['x_range'][1], 801)
    V_conv = potential_barrier(x_conv, x0=0.0,
                               width=CONV_PARAMS['barrier_width'],
                               height=CONV_PARAMS['barrier_height'])
    psi0_conv = gaussian_wavepacket(x_conv,
                                    x0=CONV_PARAMS['psi_x0'],
                                    k0=CONV_PARAMS['psi_k0'],
                                    sigma=CONV_PARAMS['psi_sigma'])
    E0 = total_energy(psi0_conv, x_conv, V_conv, dx_conv)

    dt_values = CONV_PARAMS['dt_values']
    energy_errors = []
    total_time = CONV_PARAMS['total_time']
    for dt in dt_values:
        solver = CrankNicolsonSolver(V_conv, dx_conv, dt)
        psi_f = psi0_conv.copy()
        Nt = int(total_time / dt)  # 固定总时间
        for _ in range(Nt):
            psi_f = solver.step(psi_f)
        E_final = total_energy(psi_f, x_conv, V_conv, dx_conv)
        energy_errors.append(abs(E_final - E0))

    # 空间收敛性: 固定 dt, 改变 dx
    dt_fixed = CONV_PARAMS['dt_fixed']
    dx_values = []
    T_errors = []
    # 参考解 (最细网格)
    x_ref, dx_ref = build_grid(CONV_PARAMS['x_range'][0], CONV_PARAMS['x_range'][1],
                               CONV_PARAMS['Nx_reference'])
    V_ref = potential_barrier(x_ref, x0=0.0,
                              width=CONV_PARAMS['barrier_width'],
                              height=CONV_PARAMS['barrier_height'])
    psi0_ref = gaussian_wavepacket(x_ref,
                                   x0=CONV_PARAMS['psi_x0'],
                                   k0=CONV_PARAMS['psi_k0'],
                                   sigma=CONV_PARAMS['psi_sigma'])
    solver_ref = CrankNicolsonSolver(V_ref, dx_ref, dt_fixed)
    psi_ref = psi0_ref.copy()
    for _ in range(int(total_time / dt_fixed)):
        psi_ref = solver_ref.step(psi_ref)
    T_ref, _ = compute_transmission_reflection(psi_ref, x_ref, 0.0, dx_ref)

    for Nx in CONV_PARAMS['Nx_values']:
        x_t, dx_t = build_grid(CONV_PARAMS['x_range'][0], CONV_PARAMS['x_range'][1], Nx)
        V_t = potential_barrier(x_t, x0=0.0,
                                width=CONV_PARAMS['barrier_width'],
                                height=CONV_PARAMS['barrier_height'])
        psi0_t = gaussian_wavepacket(x_t,
                                     x0=CONV_PARAMS['psi_x0'],
                                     k0=CONV_PARAMS['psi_k0'],
                                     sigma=CONV_PARAMS['psi_sigma'])
        solver_t = CrankNicolsonSolver(V_t, dx_t, dt_fixed)
        psi_t = psi0_t.copy()
        for _ in range(int(total_time / dt_fixed)):
            psi_t = solver_t.step(psi_t)
        T_t, _ = compute_transmission_reflection(psi_t, x_t, 0.0, dx_t)
        dx_values.append(dx_t)
        T_errors.append(abs(T_t - T_ref))

    plot_convergence_analysis(
        np.array(dt_values), np.array(energy_errors),
        np.array(dx_values), np.array(T_errors),
        os.path.join(assets_dir, 'fig15_convergence.png'))
    print("  [图15] 收敛性分析 -> fig15_convergence.png")
    print(f"  时间收敛: dt 从 {dt_values[0]} 到 {dt_values[-1]}, "
          f"误差从 {energy_errors[0]:.2e} 到 {energy_errors[-1]:.2e}")
    print(f"  空间收敛: dx 从 {dx_values[0]:.3f} 到 {dx_values[-1]:.3f}, "
          f"误差从 {T_errors[0]:.2e} 到 {T_errors[-1]:.2e}")


def main():
    """主函数: 组织所有模拟流程."""
    t_total_start = time.time()

    # 输出目录 (相对于本脚本所在目录的 ../1_Paper/assets/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(script_dir, '..', '1_Paper', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    print(f"图片输出目录: {os.path.abspath(assets_dir)}")

    # 1. 主模拟
    x, dx, psi0, V, psi_history, t_history = run_main_simulation(assets_dir, PARAMS)

    # 2. 势垒高度扫描
    height_results = run_height_scan(assets_dir, PARAMS, x, dx, psi0)

    # 3. 势垒宽度扫描
    width_results = run_width_scan(assets_dir, PARAMS, x, dx, psi0)

    # 4. 最终波函数对比
    run_final_states_comparison(assets_dir, PARAMS, x, dx, psi0)

    # 5. 谐振子本征态
    run_harmonic_eigenstates(assets_dir, PARAMS)

    # 6. 双势阱本征态
    run_double_well_eigenstates(assets_dir, PARAMS)

    # 7. 动量空间与能量分布分析 (新增)
    run_momentum_energy_analysis(assets_dir, PARAMS, x, dx, psi0, psi_history[-1])

    # 8. 透射系数误差分析 (新增)
    run_error_analysis(assets_dir, PARAMS, x, dx, psi0, height_results, width_results)

    # 9. 含时振荡势垒模拟 (创新拓展, 新增)
    run_time_dependent_barrier(assets_dir, PARAMS, x, dx, psi0)

    # 10. 数值收敛性分析 (新增)
    run_convergence_analysis(assets_dir, PARAMS)

    t_total = time.time() - t_total_start
    section("全部模拟完成")
    print(f"  总耗时: {t_total:.2f} s")
    print(f"  所有图片已保存至: {os.path.abspath(assets_dir)}")


if __name__ == '__main__':
    main()
