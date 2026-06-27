"""
plot_utils.py
=============
可视化模块.

本模块负责生成论文所需的所有图表, 包括:
  - 波包传播时空演化图
  - 隧穿概率随势垒高度/宽度变化曲线
  - 谐振子与双势阱的本征态图
  - 概率/能量守恒检验图
  - 波包质心轨迹图
  - 动量空间分布与能量分布 (创新)
  - CN vs Euler 格式对比 (理论深度)
  - 数值收敛性分析 (创新)
  - 含时势场结果 (创新拓展)

所有图表均采用 Matplotlib 绘制, 中文字体使用 LXGW WenKai,
并启用 constrained_layout 以避免元素重叠.

注意: 本模块不包含任何物理计算逻辑, 仅接收数据并绘图.

AI 协作声明:
    本模块由作者在 AI 助手 (GLM, 智谱 AI) 辅助下编写.
    AI 主要协助: Matplotlib 字体设置, constrained_layout 布局优化, colormap 设计.
    图表设计、标注内容由作者独立完成.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# ---------------------------------------------------------------------------
# 字体设置: 尝试加载中文字体, 失败则回退到系统默认
# ---------------------------------------------------------------------------
_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/lxgw-wenkai/LXGWWenKai-Regular.ttf',
    '/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
]

_font_loaded = False
for _font_path in _FONT_CANDIDATES:
    if os.path.exists(_font_path):
        try:
            fm.fontManager.addfont(_font_path)
            _font_loaded = True
            break
        except Exception:
            continue

# 尝试加载 DejaVu Sans 作为符号回退
_dejavu_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
if os.path.exists(_dejavu_path):
    try:
        fm.fontManager.addfont(_dejavu_path)
    except Exception:
        pass

plt.rcParams['font.sans-serif'] = ['LXGW WenKai', 'Noto Sans SC', 'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 自定义 colormap 用于 |ψ|² 时空图
PSI_CMAP = LinearSegmentedColormap.from_list(
    "psi_cmap", ["#ffffff", "#a6cee3", "#1f78b4", "#08306b"], N=256)


# ---------------------------------------------------------------------------
# 图1: 初始波包与势垒
# ---------------------------------------------------------------------------
def plot_initial_state(x, psi0, V, save_path, barrier_x0=0.0, barrier_width=2.0):
    """绘制初始高斯波包概率密度与势垒形状."""
    fig, ax1 = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    color1 = '#08306b'
    ax1.plot(x, np.abs(psi0) ** 2, color=color1, lw=2.0, label=r'$|\psi(x,0)|^2$')
    ax1.set_xlabel(r'$x\;/\;a_0$')
    ax1.set_ylabel(r'概率密度 $|\psi|^2$', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(-35, 15)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.plot(x, V, color=color2, lw=2.0, label=r'$V(x)$')
    ax2.fill_between(x, V, alpha=0.15, color=color2)
    ax2.set_ylabel(r'势能 $V(x)\;/\;E_{\mathrm{h}}$', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(V.max() * 1.5, 1.0))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.9)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图2: 波包传播时空演化 (2D 热图 + 关键时刻快照)
# ---------------------------------------------------------------------------
def plot_wavepacket_evolution(x, t_history, psi_history, V, save_path,
                              snapshot_times=None, n_snapshots=4):
    """
    绘制波包演化图: 上方为 |ψ|² 时空热图, 下方为若干关键时刻的快照.
    """
    prob_density = np.abs(psi_history) ** 2
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # 上图: 时空热图
    T_grid, X_grid = np.meshgrid(t_history, x, indexing='ij')
    pcm = ax1.pcolormesh(T_grid, X_grid, prob_density,
                         cmap=PSI_CMAP, shading='auto')
    # 叠加势垒轮廓
    ax1.fill_between(x, V / V.max() * (x.max() * 0.95),
                     x.min(), alpha=0.0)  # 占位
    # 在势垒位置画半透明带
    barrier_mask = V > 0
    if barrier_mask.any():
        x_barrier = x[barrier_mask]
        ax1.axvspan(x_barrier.min(), x_barrier.max(),
                    color='#d62728', alpha=0.25, label='势垒')
        ax1.legend(loc='upper right')
    ax1.set_xlabel(r'时间 $t\;/\;\hbar E_{\mathrm{h}}^{-1}$')
    ax1.set_ylabel(r'位置 $x\;/\;a_0$')
    ax1.set_title(r'波包概率密度 $|\psi(x,t)|^2$ 的时空演化')
    cbar = fig.colorbar(pcm, ax=ax1, pad=0.01)
    cbar.set_label(r'$|\psi|^2$')

    # 下图: 关键时刻快照
    if snapshot_times is None:
        snapshot_indices = np.linspace(0, len(t_history) - 1, n_snapshots, dtype=int)
    else:
        snapshot_indices = [np.argmin(np.abs(t_history - st)) for st in snapshot_times]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(snapshot_indices)))
    for idx, c in zip(snapshot_indices, colors):
        t_val = t_history[idx]
        ax2.plot(x, prob_density[idx], color=c, lw=1.6,
                 label=fr'$t = {t_val:.2f}$')
    # 势垒
    ax2b = ax2.twinx()
    ax2b.plot(x, V, color='#d62728', lw=1.2, ls='--', alpha=0.7)
    ax2b.set_ylabel(r'$V(x)$', color='#d62728')
    ax2b.tick_params(axis='y', labelcolor='#d62728')
    ax2b.set_ylim(0, max(V.max() * 1.5, 1.0))
    ax2.set_xlabel(r'位置 $x\;/\;a_0$')
    ax2.set_ylabel(r'概率密度 $|\psi|^2$')
    ax2.set_xlim(-45, 30)
    ax2.legend(loc='upper right', ncol=2, framealpha=0.9)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图3: 隧穿概率随势垒高度变化
# ---------------------------------------------------------------------------
def plot_transmission_vs_height(results, save_path):
    """绘制透射系数随势垒高度变化曲线, 含数值/WKB/精确解对比."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    h = results['heights']
    ax.semilogy(h, results['T_numerical'], 'o-', color='#08306b',
                label='数值模拟 (Crank-Nicolson)', markersize=6, lw=1.5)
    ax.semilogy(h, np.clip(results['T_exact'], 1e-10, 1.0), '--',
                color='#d62728', label='精确解析解', lw=1.8)
    ax.semilogy(h, np.clip(results['T_wkb'], 1e-10, 1.0), ':',
                color='#2ca02c', label='WKB 近似', lw=1.8)
    E = results['E_incident']
    ax.axvline(E, color='gray', ls='-.', alpha=0.7,
               label=fr'入射能量 $E_0 = {E:.2f}$')
    ax.set_xlabel(r'势垒高度 $V_0\;/\;E_{\mathrm{h}}$')
    ax.set_ylabel(r'透射系数 $T$')
    ax.set_title(r'透射系数随势垒高度的变化 (固定宽度 $a = 2.0\,a_0$)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_ylim(1e-6, 2)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图4: 隧穿概率随势垒宽度变化
# ---------------------------------------------------------------------------
def plot_transmission_vs_width(results, save_path):
    """绘制透射系数随势垒宽度变化曲线."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    w = results['widths']
    ax.semilogy(w, results['T_numerical'], 's-', color='#08306b',
                label='数值模拟 (Crank-Nicolson)', markersize=6, lw=1.5)
    ax.semilogy(w, np.clip(results['T_exact'], 1e-10, 1.0), '--',
                color='#d62728', label='精确解析解', lw=1.8)
    ax.semilogy(w, np.clip(results['T_wkb'], 1e-10, 1.0), ':',
                color='#2ca02c', label='WKB 近似', lw=1.8)
    E = results['E_incident']
    ax.set_xlabel(r'势垒宽度 $a\;/\;a_0$')
    ax.set_ylabel(r'透射系数 $T$')
    ax.set_title(fr'透射系数随势垒宽度的变化 (固定高度 $V_0 = 10\,E_{{\mathrm{{h}}}}, E_0 = {E:.2f}$)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', ls=':', alpha=0.4)
    ax.set_ylim(1e-6, 2)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图5: 概率与能量守恒检验
# ---------------------------------------------------------------------------
def plot_conservation(t_history, norms, energies, save_path):
    """绘制归一化系数与总能量随时间的演化, 检验守恒性."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5),
                                    constrained_layout=True, sharex=True)
    ax1.plot(t_history, norms, color='#08306b', lw=1.5)
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.7)
    ax1.set_ylabel(r'归一化 $\int|\psi|^2 dx$')
    ax1.set_title('概率守恒检验')
    ax1.grid(True, ls=':', alpha=0.4)
    # 标注最大偏差
    dev = np.max(np.abs(norms - 1.0))
    ax1.text(0.02, 0.92, fr'最大偏差: $\Delta N = {dev:.2e}$',
             transform=ax1.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax2.plot(t_history, energies, color='#d62728', lw=1.5)
    E0 = energies[0]
    ax2.axhline(E0, color='gray', ls='--', alpha=0.7)
    ax2.set_xlabel(r'时间 $t$')
    ax2.set_ylabel(r'总能量 $\langle H \rangle$')
    ax2.set_title('能量守恒检验')
    ax2.grid(True, ls=':', alpha=0.4)
    dev_E = np.max(np.abs(energies - E0))
    ax2.text(0.02, 0.92, fr'最大偏差: $\Delta E = {dev_E:.2e}$',
             transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图6: 谐振子本征态
# ---------------------------------------------------------------------------
def plot_harmonic_eigenstates(x, V, energies, eigenstates, save_path, n_show=5):
    """绘制谐振子势与最低几个本征态波函数."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    constrained_layout=True)
    # 左图: 势 + 能级
    ax1.plot(x, V, color='black', lw=1.8, label=r'$V(x) = \frac{1}{2}kx^2$')
    for i in range(min(n_show, len(energies))):
        ax1.axhline(energies[i], color=f'C{i}', ls='--', alpha=0.6)
        ax1.text(x.max() * 0.7, energies[i], fr'$E_{i} = {energies[i]:.3f}$',
                 color=f'C{i}', fontsize=9, va='bottom')
    ax1.set_xlabel(r'$x\;/\;a_0$')
    ax1.set_ylabel(r'能量 $E\;/\;E_{\mathrm{h}}$')
    ax1.set_title('谐振子势与能级')
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(0, energies[n_show - 1] + 1.5)
    ax1.legend(loc='upper right')
    ax1.grid(True, ls=':', alpha=0.4)

    # 右图: 本征波函数
    for i in range(min(n_show, len(eigenstates))):
        # 偏移以便观察
        offset = energies[i]
        ax2.plot(x, eigenstates[i] + offset, color=f'C{i}', lw=1.5,
                 label=fr'$\phi_{i}$, $E_{i}={energies[i]:.3f}$')
        ax2.axhline(offset, color=f'C{i}', ls=':', alpha=0.3)
    ax2.set_xlabel(r'$x\;/\;a_0$')
    ax2.set_ylabel(r'$\phi_n(x)$ (偏移至能级位置)')
    ax2.set_title('谐振子本征波函数')
    ax2.set_xlim(-6, 6)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, ls=':', alpha=0.4)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图7: 双势阱本征态
# ---------------------------------------------------------------------------
def plot_double_well_eigenstates(x, V, energies, eigenstates, save_path, n_show=6):
    """绘制双势阱势与最低几个本征态."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    constrained_layout=True)
    ax1.plot(x, V, color='black', lw=1.8, label=r'$V(x) = bx^4 - ax^2$')
    for i in range(min(n_show, len(energies))):
        ax1.axhline(energies[i], color=f'C{i}', ls='--', alpha=0.6)
        ax1.text(x.max() * 0.55, energies[i], fr'$E_{i} = {energies[i]:.3f}$',
                 color=f'C{i}', fontsize=9, va='bottom')
    ax1.set_xlabel(r'$x\;/\;a_0$')
    ax1.set_ylabel(r'能量 $E\;/\;E_{\mathrm{h}}$')
    ax1.set_title('双势阱势与能级')
    ax1.set_xlim(-4, 4)
    V_max_plot = max(energies[n_show - 1] + 1.0, V.max() * 0.4)
    ax1.set_ylim(V.min() - 0.5, V_max_plot)
    ax1.legend(loc='upper right')
    ax1.grid(True, ls=':', alpha=0.4)

    for i in range(min(n_show, len(eigenstates))):
        offset = energies[i]
        ax2.plot(x, eigenstates[i] * 1.5 + offset, color=f'C{i}', lw=1.5,
                 label=fr'$\phi_{i}$, $E_{i}={energies[i]:.3f}$')
        ax2.axhline(offset, color=f'C{i}', ls=':', alpha=0.3)
    ax2.set_xlabel(r'$x\;/\;a_0$')
    ax2.set_ylabel(r'$\phi_n(x)$ (偏移至能级位置)')
    ax2.set_title('双势阱本征波函数')
    ax2.set_xlim(-4, 4)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, ls=':', alpha=0.4)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图8: 波包质心轨迹与展宽
# ---------------------------------------------------------------------------
def plot_centroid_trajectory(t_history, centroids, stds, save_path, barrier_x0=0.0):
    """绘制波包质心位置与宽度随时间的演化."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(t_history, centroids, color='#08306b', lw=1.8,
            label=r'质心 $\langle x \rangle$')
    ax.fill_between(t_history, centroids - stds, centroids + stds,
                    color='#08306b', alpha=0.2, label=r'宽度 $\langle x \rangle \pm \sigma_x$')
    ax.axhline(barrier_x0, color='#d62728', ls='--', lw=1.2,
               label=fr'势垒位置 $x = {barrier_x0}$')
    ax.set_xlabel(r'时间 $t$')
    ax.set_ylabel(r'位置 $x\;/\;a_0$')
    ax.set_title('波包质心轨迹与展宽')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, ls=':', alpha=0.4)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图9: 不同势垒高度下的最终波函数对比
# ---------------------------------------------------------------------------
def plot_final_states_comparison(x, psi_list, V_list, labels, save_path):
    """绘制不同势垒参数下最终波函数概率密度的对比."""
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(psi_list)))
    for psi, V, lab, c in zip(psi_list, V_list, labels, colors):
        ax.plot(x, np.abs(psi) ** 2, color=c, lw=1.5, label=lab)
    ax.set_xlabel(r'位置 $x\;/\;a_0$')
    ax.set_ylabel(r'概率密度 $|\psi|^2$')
    ax.set_title('不同势垒参数下最终波函数对比')
    ax.set_xlim(-50, 40)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, ls=':', alpha=0.4)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图10: 动量空间分布 (初始 vs 最终)
# ---------------------------------------------------------------------------
def plot_momentum_space(p, rho_p_initial, rho_p_final, save_path,
                        k0=None, V0=None):
    """
    绘制动量空间概率分布 |φ(p)|² 的对比 (初始 vs 隧穿后).

    这揭示了隧穿现象的动量空间物理图像:
      - 初始波包在 p = ħk₀ 处有峰值
      - 隧穿后, 透射部分保持正向动量, 反射部分动量反转
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(p, rho_p_initial, color='#1f77b4', lw=1.8, alpha=0.8,
            label=r'初始 $|\phi(p, 0)|^2$')
    ax.plot(p, rho_p_final, color='#d62728', lw=1.8, alpha=0.8,
            label=r'最终 $|\phi(p, t_f)|^2$')
    if k0 is not None:
        ax.axvline(k0, color='gray', ls='--', alpha=0.6,
                   label=fr'初始动量 $p_0 = \hbar k_0 = {k0}$')
        ax.axvline(-k0, color='gray', ls=':', alpha=0.6,
                   label=fr'反射动量 $-p_0 = {-k0}$')
    ax.set_xlabel(r'动量 $p\;/\;(\hbar/a_0)$')
    ax.set_ylabel(r'动量概率密度 $|\phi(p)|^2$')
    ax.set_title('动量空间分布: 隧穿与反射的动量图像')
    ax.set_xlim(-12, 12)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图11: 能量分布与势垒关系
# ---------------------------------------------------------------------------
def plot_energy_distribution(E, rho_E, save_path, V0=None, E_mean=None,
                             E_kinetic=None):
    """
    绘制波包的能量分布 ρ(E), 并标注势垒高度 V₀.

    物理意义: 波包具有能量展宽. E > V₀ 的成分经典透射,
    E < V₀ 的成分发生量子隧穿. 这解释了为何数值透射系数
    与单能理论值存在偏差.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.fill_between(E, rho_E, alpha=0.4, color='#1f77b4')
    ax.plot(E, rho_E, color='#1f77b4', lw=1.8,
            label=r'能量分布 $\rho(E)$')
    if V0 is not None:
        ax.axvline(V0, color='#d62728', ls='--', lw=1.5,
                   label=fr'势垒高度 $V_0 = {V0}$')
        # 标注隧穿区域
        ax.axvspan(0, V0, alpha=0.08, color='#d62728',
                   label='隧穿区域 ($E < V_0$)')
    if E_mean is not None:
        ax.axvline(E_mean, color='green', ls=':', lw=1.5,
                   label=fr'平均能量 $\langle E \rangle = {E_mean:.2f}$')
    ax.set_xlabel(r'能量 $E\;/\;E_{\mathrm{h}}$')
    ax.set_ylabel(r'能量概率密度 $\rho(E)$')
    ax.set_title('波包能量分布与势垒高度的关系')
    ax.set_xlim(0, max(E.max(), V0 * 1.5 if V0 else E.max()))
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图12: 透射系数误差分析 (数值 vs 理论)
# ---------------------------------------------------------------------------
def plot_transmission_error(scan_values, T_num, T_theory, save_path,
                            xlabel='势垒高度 $V_0$', title='透射系数误差分析'):
    """
    绘制数值透射系数与理论值的对比及相对误差.

    上图: T vs 参数 (数值 + 理论曲线)
    下图: 相对误差 |T_num - T_theory| / T_theory
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True,
                                    sharex=True)
    # 上图: 对比
    ax1.semilogy(scan_values, T_num, 'o-', color='#1f77b4', lw=1.5, ms=5,
                 label='数值结果 (CN)')
    ax1.semilogy(scan_values, T_theory, 's--', color='#d62728', lw=1.5, ms=5,
                 label='理论值 (精确解)')
    ax1.set_ylabel(r'透射系数 $T$')
    ax1.set_title(title)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, ls=':', alpha=0.4, which='both')
    # 下图: 相对误差
    rel_err = np.abs(T_num - T_theory) / np.maximum(T_theory, 1e-15)
    ax2.semilogy(scan_values, rel_err, 'o-', color='#2ca02c', lw=1.5, ms=5)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(r'相对误差 $|T_{\mathrm{num}} - T_{\mathrm{th}}|/T_{\mathrm{th}}$')
    ax2.grid(True, ls=':', alpha=0.4, which='both')
    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图13: 含时振荡势垒结果 (创新拓展)
# ---------------------------------------------------------------------------
def plot_time_dependent_barrier(x, psi_history_static, psi_history_dynamic,
                                t_history, save_path, barrier_x0=0.0):
    """
    对比静态势垒与含时振荡势垒下的波包演化 (创新拓展).

    上图: 时空演化热图 (静态)
    中图: 时空演化热图 (含时振荡)
    下图: 透射概率随时间的演化对比
    """
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

    # 静态势垒
    rho_static = np.abs(psi_history_static) ** 2
    im1 = axes[0].pcolormesh(x, t_history, rho_static,
                              cmap=PSI_CMAP, shading='auto')
    axes[0].axvline(barrier_x0, color='red', ls='--', lw=1, alpha=0.7)
    axes[0].set_ylabel(r'时间 $t$')
    axes[0].set_title('(a) 静态势垒: $V_0 = 15$ (常数)')
    axes[0].set_xlim(-50, 40)
    plt.colorbar(im1, ax=axes[0], label=r'$|\psi|^2$')

    # 含时势垒
    rho_dynamic = np.abs(psi_history_dynamic) ** 2
    im2 = axes[1].pcolormesh(x, t_history, rho_dynamic,
                              cmap=PSI_CMAP, shading='auto')
    axes[1].axvline(barrier_x0, color='red', ls='--', lw=1, alpha=0.7)
    axes[1].set_ylabel(r'时间 $t$')
    axes[1].set_title(r'(b) 含时振荡势垒: $V(t) = V_0[1 + 0.3\sin(\omega t)]$')
    axes[1].set_xlim(-50, 40)
    plt.colorbar(im2, ax=axes[1], label=r'$|\psi|^2$')

    # 透射概率随时间
    dx = x[1] - x[0]
    T_static = np.array([np.sum(np.abs(psi)[x > barrier_x0] ** 2) * dx
                         for psi in psi_history_static])
    T_dynamic = np.array([np.sum(np.abs(psi)[x > barrier_x0] ** 2) * dx
                          for psi in psi_history_dynamic])
    axes[2].plot(t_history, T_static, color='#1f77b4', lw=1.8,
                 label='静态势垒')
    axes[2].plot(t_history, T_dynamic, color='#d62728', lw=1.8,
                 label='含时振荡势垒')
    axes[2].set_xlabel(r'时间 $t$')
    axes[2].set_ylabel(r'透射概率 $T(t)$')
    axes[2].set_title('(c) 透射概率随时间的演化')
    axes[2].legend(loc='lower right', framealpha=0.9)
    axes[2].grid(True, ls=':', alpha=0.4)

    plt.savefig(save_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图14: 数值收敛性分析 (创新拓展)
# ---------------------------------------------------------------------------
def plot_convergence_analysis(dt_values, energy_errors, dx_values, T_errors,
                              save_path):
    """
    绘制数值收敛性分析:
      左图: 能量守恒误差 vs 时间步长 Δt (验证 O(Δt²) 收敛)
      右图: 透射系数误差 vs 空间步长 Δx (验证 O(Δx²) 收敛)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    constrained_layout=True)
    # 左图: 时间收敛性
    ax1.loglog(dt_values, energy_errors, 'o-', color='#1f77b4', lw=1.5, ms=7,
               label='数值误差')
    # 参考线 O(Δt²)
    ref = energy_errors[0] * (dt_values / dt_values[0]) ** 2
    ax1.loglog(dt_values, ref, '--', color='gray', lw=1.2,
               label=r'$\propto \Delta t^2$ (二阶参考)')
    ax1.set_xlabel(r'时间步长 $\Delta t$')
    ax1.set_ylabel(r'能量守恒最大偏差')
    ax1.set_title('时间收敛性 (Crank-Nicolson)')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, ls=':', alpha=0.4, which='both')

    # 右图: 空间收敛性
    ax2.loglog(dx_values, T_errors, 's-', color='#d62728', lw=1.5, ms=7,
               label='数值误差')
    ref2 = T_errors[0] * (dx_values / dx_values[0]) ** 2
    ax2.loglog(dx_values, ref2, '--', color='gray', lw=1.2,
               label=r'$\propto \Delta x^2$ (二阶参考)')
    ax2.set_xlabel(r'空间步长 $\Delta x$')
    ax2.set_ylabel(r'透射系数相对误差')
    ax2.set_title('空间收敛性 (中心差分)')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, ls=':', alpha=0.4, which='both')

    plt.savefig(save_path)
    plt.close(fig)
