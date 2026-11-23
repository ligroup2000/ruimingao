# -*- coding: utf-8 -*-
"""2D 周期非线性薛定谔方程：i*u_t + u_xx + 2*|u|^2*u = 0。
区间为正方形区域, |x|+|y|<=\sqrt{2}L, L=10
N=128, 将u=a+ib处理为 a,b分开计算
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

def configure(threads, seed, device):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                        else 'cpu' if device == 'auto' else device)


def initial_state(grid):
    """直接给定 u=a+bi 的两个实值初始分量 a0、b0。"""
    x, y = grid.physical_coordinates()
    a0 = torch.zeros_like(x)
    b0 = torch.zeros_like(x)
    for k in range(2):
        shifted = x-(-1)**k*2
        amplitude = torch.exp(-shifted.square()-y.square()+y)
        phase = grid.nu*shifted
        a0 = a0 + amplitude*torch.cos(phase)
        b0 = b0 + amplitude*torch.sin(phase)
    return torch.stack([a0, b0])

class PeriodicGrid:
    """Rotated periodic square: q=(x+y)/sqrt(2), r=(y-x)/sqrt(2).

    |q|,|r| <= L maps exactly to |x|+|y| <= sqrt(2)*L.
    The rotation has unit Jacobian and preserves the Laplacian.
    """
    def __init__(self, n, length, device, nu=.1):
        if n < 4 or n % 2:
            raise ValueError('n must be even and at least 4')
        if not math.isfinite(length) or length <= 0 or not math.isfinite(nu):
            raise ValueError('length must be positive and parameters finite')
        self.n, self.length, self.nu = n, length, nu
        self.dx = length/n
        self.cell_volume = self.dx**2
        self.x = -length/2+self.dx*torch.arange(n, device=device)
        k = 2*math.pi*torch.fft.fftfreq(n, d=self.dx, device=device)
        k[n//2] = 0.
        self.k1, self.k2 = k[:, None], k[None, :]
        self.k_squared = self.k1.square()+self.k2.square()

    def physical_coordinates(self):
        q, r = torch.meshgrid(self.x, self.x, indexing='ij')
        return (q-r)/math.sqrt(2), (q+r)/math.sqrt(2)

    def coordinate_arrays(self):
        x, y = self.physical_coordinates()
        return dict(q=self.x.cpu().numpy(), r=self.x.cpu().numpy(),
                    x1=x.cpu().numpy(), x2=y.cpu().numpy())

    @staticmethod
    def fft(f):
        return torch.fft.fftn(f, dim=(-2, -1))

    @staticmethod
    def ifft(f):
        return torch.fft.ifftn(f, dim=(-2, -1))

    def dq(self, field):
        """实值分量对计算坐标 q 的导数；FFT 中间系数为复数。"""
        return self.ifft(1j*self.k1*self.fft(field)).real

    def dr(self, field):
        """实值分量对计算坐标 r 的导数。"""
        return self.ifft(1j*self.k2*self.fft(field)).real

    def dx1(self, field):
        """物理 x 导数：d/dx=(d/dq-d/dr)/sqrt(2)，同样适用于 a、b。"""
        return (self.dq(field)-self.dr(field))/math.sqrt(2)

    def dy1(self, field):
        """物理 y 导数：d/dy=(d/dq+d/dr)/sqrt(2)。"""
        return (self.dq(field)+self.dr(field))/math.sqrt(2)

    def laplacian(self, f):
        return self.ifft(-self.k_squared*self.fft(f)).real

    def gradient_squared(self, f):
        return self.dx1(f).square()+self.dy1(f).square()

    def rhs(self, state):
        a, b = state.unbind(dim=-3)
        rho = a.square()+b.square()
        return torch.stack([-self.laplacian(b)-2*rho*b,
                             self.laplacian(a)+2*rho*a], dim=-3)

    def mass(self, state):
        return .5*self.cell_volume*state.square().sum(dim=(-3, -2, -1))

    def energy(self, state):
        a, b = state.unbind(dim=-3)
        rho = a.square()+b.square()
        density = .5*(self.gradient_squared(a)+self.gradient_squared(b)-rho.square())
        return self.cell_volume*density.sum(dim=(-2, -1))

    def momentum(self, state):
        a, b = state.unbind(dim=-3)
        return .5*self.cell_volume*torch.stack([
            (a*D(b)-b*D(a)).sum(dim=(-2, -1))
            for D in (self.dx1, self.dy1)], dim=-1)

    def invariant_increment(self, state, change):
        """Exact polynomial increments of M, E, Px, Py."""
        a, b = state.unbind(dim=-3)
        da, db = change.unbind(dim=-3)
        rho = a.square()+b.square()
        drho = 2*a*da+da.square()+2*b*db+db.square()
        de = -.5*(2*rho*drho+drho.square())
        for D in (self.dx1, self.dy1):
            old, delta = D(state), D(change)
            de = de + .5*(2*old*delta+delta.square()).sum(dim=-3)
        dp = []
        for D in (self.dx1, self.dy1):
            value = a*D(db)+da*D(b)+da*D(db)-b*D(da)-db*D(a)-db*D(da)
            dp.append(.5*value.sum(dim=(-2, -1)))
        return self.cell_volume*torch.stack([
            .5*drho.sum(dim=(-2, -1)), de.sum(dim=(-2, -1)), *dp], dim=-1)

class NewtonFailure(RuntimeError):
    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = dict(diagnostic or {})
        self.diagnostic.setdefault('reason',message)

class SpectralConv2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(2, width, width, modes, modes, 2)
                                   /(width*math.sqrt(2)))

    def forward(self, z):
        zhat = torch.fft.rfft2(z)
        m1 = min(self.modes, z.shape[-2]//2)
        m2 = min(self.modes, zhat.shape[-1])
        result = torch.zeros_like(zhat)
        weights = torch.view_as_complex(self.weight.contiguous())
        for index, rows in enumerate((slice(0, m1), slice(-m1, None))):
            result[..., rows, :m2] = torch.einsum(
                'bixy,ioxy->boxy', zhat[..., rows, :m2], weights[index, ..., :m1, :m2])
        return torch.fft.irfft2(result, s=z.shape[-2:])


class FourierCNN(nn.Module):
    def __init__(self, channels, width, modes):
        super().__init__()
        self.lift = nn.Conv2d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(2)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(2)])
        self.project = nn.Conv2d(width, 2, 1)
        nn.init.zeros_(self.project.weight); nn.init.zeros_(self.project.bias)

    def forward(self, features):
        z = self.lift(features)
        for spectral, local in zip(self.spectral, self.local):
            z = torch.tanh(spectral(z)+local(z))
        # 无边界因子、无端点投影；保留有序周期网格及逆 FFT 的周期解释。
        return self.project(z)


SNAPSHOT_TIMES = (0., .5, 1., 1.5)
SOLUTION_SAVE_END = 5.

TABLEAU_METHODS = {
    'gauss_legendre': ('Gauss-Legendre', 3, 6),
    'lobatto_iiia': ('Lobatto IIIA', 4, 6),
}


def gauss_legendre_tableau(state):
    """三阶段、六阶 Gauss-Legendre 配置法与一套嵌入权重。

    c 是 [0,1] 上的三个 Gauss 点。
    A_ij = integral_0^{c_i} L_j(s) ds，b1_j = integral_0^1 L_j(s) ds，
    其中 L_j 为这三个节点上的 Lagrange 基函数。
    保留当前 b2=(1/6,1/6,2/3)，sum(b2)=1，仅保证一阶。
    配置主表来源：https://juliagni.github.io/RungeKutta.jl/stable/gauss/
    """
    r = math.sqrt(15.)
    c = state.new_tensor([.5-r/10, .5, .5+r/10])
    a = state.new_tensor([
        [5/36,       2/9-r/15, 5/36-r/30],
        [5/36+r/24,  2/9,      5/36-r/24],
        [5/36+r/30,  2/9+r/15, 5/36],
    ])
    b1 = state.new_tensor([5/18, 4/9, 5/18])
    b2 = state.new_tensor([1/6, 1/6, 2/3])
    return a, b1, b2, c


def lobatto_iiia_tableau(state):
    """四阶段六阶 Lobatto IIIA 主表；六阶指未松弛的主 RK 法。

    来源：https://juliagni.github.io/RungeKutta.jl/stable/lobatto/
    b2 为独立的一阶辅助权重，用于现有双参数 MRRK，不是六阶权重。
    """
    r = math.sqrt(5.)
    c = state.new_tensor([0., .5-r/10, .5+r/10, 1.])
    a = state.new_tensor([
        [0., 0., 0., 0.],
        [(11+r)/120, (25-r)/120, (25-13*r)/120, (r-1)/120],
        [(11-r)/120, (25+13*r)/120, (25+r)/120, -(1+r)/120],
        [1/12, 5/12, 5/12, 1/12],
    ])
    b1 = state.new_tensor([1/12, 5/12, 5/12, 1/12])
    b2 = state.new_tensor([1/4, 1/4, 1/4, 1/4])
    return a, b1, b2, c

def save_snapshot_surfaces(output_directory, x, snapshots):
    """在完整菱形区域绘制四个指定时刻的 |u| 三维曲面。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    Q, R = np.meshgrid(x, x, indexing='ij')
    X1, X2 = (Q-R)/np.sqrt(2), (Q+R)/np.sqrt(2)
    available = [t for t in SNAPSHOT_TIMES if t in snapshots]
    if not available:
        return
    for name, label, component in (
        ('amplitude', r'$|u|$', lambda state: np.sqrt(state[0]**2+state[1]**2)),
    ):
        fields = {t: component(np.asarray(snapshots[t])) for t in available}
        lower = min(float(field.min()) for field in fields.values())
        upper = max(float(field.max()) for field in fields.values())
        padding = .05*(upper-lower) if upper > lower else .05*max(abs(upper), 1.)
        fig = plt.figure(figsize=(13, 10), layout='constrained')
        active_axes = []
        for index, t in enumerate(SNAPSHOT_TIMES, 1):
            ax = fig.add_subplot(2, 2, index, projection='3d')
            if t not in fields:
                ax.set_axis_off()
                ax.set_title(f't = {t:g} (unavailable)')
                continue
            surface = ax.plot_surface(X1, X2, fields[t], cmap='viridis',
                                      vmin=lower, vmax=upper, rstride=1, cstride=1,
                                      linewidth=0, antialiased=True)
            ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$', zlabel=label,
                   title=f't = {t:g}', zlim=(lower-padding, upper+padding))
            ax.view_init(elev=28, azim=-55)
            active_axes.append(ax)
        fig.colorbar(surface, ax=active_axes, shrink=.65, pad=.06, label=label)
        fig.suptitle(f'{label}: time snapshots')
        fig.savefig(output_directory / f'time_snapshots_{name}_3d.png', dpi=200)
        plt.close(fig)


def save_plots(output_directory, x, times, states, history, snapshots, state_times):
    """守恒图与演化图；指定时刻使用阶段修正后的端点作 Hermite 插值。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Q, R = np.meshgrid(x, x, indexing='ij')
    X1, X2 = (Q-R)/np.sqrt(2), (Q+R)/np.sqrt(2)
    times = np.asarray(times)
    state_times = np.asarray(state_times)
    states = np.asarray(states)

    # 能量只使用接受步记录，包含初始时刻的零误差。
    relative_errors = np.array([0.] + [row['relative_energy_error'] for row in history])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, relative_errors, linewidth=1.4)
    ax.set(xlabel='Physical time t', ylabel=r'$|E(t)-E(0)|/|E(0)|$',
           title='Relative energy conservation error')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_directory / 'energy_relative_error.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, [0.] + [row['energy_error'] for row in history])
    axes[0].set(title='Energy conservation', ylabel='E(t) - E(0)')
    axes[1].plot(times, [0.] + [row['mass_error'] for row in history])
    axes[1].set(title='Mass conservation', ylabel='M(t) - M(0)')
    for ax in axes:
        ax.set_xlabel('Physical time t')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_directory / 'mass_energy_conservation.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    momentum_errors = np.asarray([[0., 0.]] + [row['momentum_error'] for row in history])
    for j in range(2):
        ax.plot(times, momentum_errors[:, j], label=f'P{j+1}(t) - P{j+1}(0)')
    ax.set(xlabel='Physical time t', ylabel='Momentum error', title='Momentum conservation')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_directory / 'momentum_conservation.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    # 对 x2 求平均，得到二维解沿 x1 的时间演化概览。
    amplitude = np.sqrt(np.sum(states[:, :2]**2, axis=1)).mean(axis=-1)
    mesh = ax.pcolormesh(x, state_times, amplitude, shading='auto', cmap='viridis')
    fig.colorbar(mesh, ax=ax, label='mean of |u| over r')
    ax.set(xlabel='q', ylabel='Physical time t', title='Evolution of mean |u| (0–5 s)')
    fig.tight_layout()
    fig.savefig(output_directory / 'long_time_evolution.png', dpi=200)
    plt.close(fig)

    available_times = [t for t in SNAPSHOT_TIMES if t in snapshots]
    snapshot_states = [np.asarray(snapshots[t]) for t in available_times]
    amplitudes = [np.hypot(state[0], state[1]) for state in snapshot_states]
    if amplitudes:
        lower = min(float(field.min()) for field in amplitudes)
        upper = max(float(field.max()) for field in amplitudes)
        if upper <= lower:
            upper = lower+max(abs(lower), 1.)*.05
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True,
                                 layout='constrained')
        active_axes = []
        for ax, requested_time in zip(axes.flat, SNAPSHOT_TIMES):
            if requested_time not in available_times:
                ax.set_axis_off()
                ax.set_title(f't = {requested_time:g} (unavailable)')
                continue
            field = amplitudes[available_times.index(requested_time)]
            mesh = ax.pcolormesh(X1, X2, field, shading='auto', cmap='viridis',
                                 vmin=lower, vmax=upper)
            ax.set(title=f't = {requested_time:g}', xlabel='x', ylabel='y', aspect='equal')
            active_axes.append(ax)
        fig.colorbar(mesh, ax=active_axes, shrink=.8, label=r'$|u|$')
        fig.suptitle(r'$|u|$: spatial snapshots')
        fig.savefig(output_directory / 'time_snapshots.png', dpi=200)
        plt.close(fig)
    np.savez_compressed(
        output_directory / 'time_snapshots.npz', q=x, r=x, x1=X1, x2=X2,
        t=np.asarray(available_times), states=np.asarray(snapshot_states),
        amplitude=np.asarray(amplitudes),
        state_components=np.asarray(['a', 'b']),
        interpolation='cubic Hermite',
    )
    save_snapshot_surfaces(output_directory, x,
                           dict(zip(available_times, snapshot_states)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(times[1:], np.maximum([row['pde_rms'] for row in history], 1e-17), label='Validation RMS')
    ax.semilogy(times[1:], np.maximum([row['pde_max_time_rms'] for row in history], 1e-17), label='Maximum time-slice RMS')
    ax.set(xlabel='Physical time t', ylabel='PDE residual', title='Hermite PDE residual')
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_directory / 'pde_residual.png', dpi=200)
    plt.close(fig)


class RKStep(nn.Module):
    """当前一步的 RK 方程；每个数学量都保留自己的名称。

    y_n.shape = (2, N, N)：当前 Re(u)、Im(u) 状态。
    K[i].shape = (2, N, N)：网络预测的第 i 个阶段斜率。
    Y[i].shape = (2, N, N)：由所有 K_j 和 RK 系数重建的第 i 个阶段状态。
    i = 0, 1, 2 对应数学下标 i = 1, 2, 3。
    """

    def __init__(self, grid, y_n, h, width=8, modes=4, direction_count=2,
                 tableau='lobatto_iiia'):
        super().__init__()
        self.grid = grid
        self.h = h
        self.register_buffer('y_n', y_n.detach().clone())

        # 只在这里选择时间离散的 Butcher 表；FNO 和空间导数不变。
        self.direction_count = direction_count
        if direction_count == 2:
            if tableau == 'gauss_legendre':
                a, b1, b2, c = gauss_legendre_tableau(y_n)
            elif tableau == 'lobatto_iiia':
                a, b1, b2, c = lobatto_iiia_tableau(y_n)
            else:
                raise ValueError(f'Unknown tableau: {tableau}')
        elif direction_count == 4:
            a, b1, b2, b3, b4, c = inverse_lobatto_iiia_tableau(y_n)
        else:
            raise ValueError('direction_count must be 2 or 4')
        self.stage_count = len(c)
        self.register_buffer('a', a)
        self.register_buffer('b1', b1)
        self.register_buffer('b2', b2)
        if direction_count == 4:
            self.register_buffer('b3', b3)
            self.register_buffer('b4', b4)
        self.register_buffer('c', c)

        # 输入：两个状态分量与阶段 c_i；输出两个分量的斜率。
        # 模态数不超过当前网格可表示的实数 Fourier 频率数。
        self.network = FourierCNN(channels=3, width=width, modes=min(modes, grid.n // 2))

    def predict_slopes(self):
        """K_i = FNO_theta(y_n, c_i)，逐阶段计算。

        FNO 要求输入形状为 (batch, channels, N, N)。这里只有一个样本，
        所以用 unsqueeze(0) 加上 batch 维度，输出后再 squeeze(0) 去掉。
        """
        K = []
        for i in range(self.stage_count):
            time_channel = self.c[i] * torch.ones_like(self.y_n[:1])
            input_channels = torch.cat([self.y_n, time_channel], dim=0)
            network_input = input_channels.unsqueeze(0)
            network_output = self.network(network_input)
            K_i = network_output.squeeze(0)
            K.append(K_i)

        return K

    def stage_states(self, K):
        """Y_i = y_n + h * sum_j a_ij K_j，由预测斜率重建。"""
        Y = []
        for i in range(self.stage_count):
            weighted_sum = torch.zeros_like(self.y_n)
            for j in range(self.stage_count):
                weighted_sum = weighted_sum + self.a[i, j] * K[j]
            Y.append(self.y_n + self.h * weighted_sum)
        return Y

    def stage_residuals(self, Y, K):
        """R_i = K_i - f(Y_i)，训练预测斜率满足阶段方程。"""
        R = []
        for i in range(self.stage_count):
            R_i = K[i] - self.grid.rhs(Y[i])
            R.append(R_i)
        return R

    def rk_loss(self):
        """L_RK = (1/3) * sum_i mean(|R_i|^2)。"""
        K = self.predict_slopes()
        Y = self.stage_states(K)
        R = self.stage_residuals(Y, K)

        loss = self.y_n.new_tensor(0.)
        for i in range(self.stage_count):
            loss = loss + torch.mean(R[i] ** 2)

        return loss / self.stage_count

    def rk_directions(self, K):
        """Configured directions from the same three stage slopes, without h."""
        stages = torch.stack(K)
        return tuple(torch.einsum('i,icxy->cxy', getattr(self, f'b{k}'), stages)
                     for k in range(1, self.direction_count+1))

    def rk_endpoint(self, K):
        """普通 RK 终点，使用第一套权重。"""
        d1, *_ = self.rk_directions(K)
        return self.y_n + self.h * d1


def train_rk(step, adam_iterations, lbfgs_iterations, learning_rate):
    """用优化器求解三个阶段方程，不使用参考解作标签。"""
    if adam_iterations:
        optimizer = torch.optim.Adam(step.network.parameters(), lr=learning_rate)
        for iteration in range(adam_iterations):
            optimizer.zero_grad()
            loss = step.rk_loss()
            if not torch.isfinite(loss):
                raise RuntimeError('RK 训练损失不是有限数。')
            loss.backward()
            optimizer.step()

    if lbfgs_iterations:
        optimizer = torch.optim.LBFGS(
            step.network.parameters(),
            max_iter=lbfgs_iterations,
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
            line_search_fn='strong_wolfe',
        )

        def closure():
            optimizer.zero_grad()
            loss = step.rk_loss()
            if not torch.isfinite(loss):
                raise RuntimeError('RK 训练损失不是有限数。')
            loss.backward()
            return loss

        optimizer.step(closure)


@torch.no_grad()
def polish_rk_stages(step, K, max_iterations, tolerance):
    """以 FNO 为初值精修所选 RK 阶段方程；频域隐式解线性项。

    非线性项使用 Picard 迭代，不作端点投影。返回实际使用的阶段斜率，
    它们在启用精修时不再是纯网络输出。零次迭代表示禁用。
    """
    grid, h, A = step.grid, step.h, step.a

    def residual_rms(slopes):
        states = step.stage_states(slopes)
        residuals = torch.stack(step.stage_residuals(states, slopes))
        return float(residuals.square().mean().sqrt())

    raw_rms = residual_rms(K)
    info = dict(fno_rk_rms=raw_rms, stage_polish_iterations=0)
    if not max_iterations or raw_rms <= tolerance:
        return K, raw_rms, info

    y = step.y_n
    # 分别对实值 a、b 作 FFT，不构造复值解 u=a+ib。
    a0_hat, b0_hat = grid.fft(y[0]), grid.fft(y[1])
    identity = torch.eye(step.stage_count, dtype=y.dtype, device=y.device)
    coupling = h*grid.k_squared[..., None, None]*A
    identity = identity.expand_as(coupling)
    # a_i - h*sum_j A_ij*k^2*b_j = a0 + h*sum_j A_ij*N_a,j
    # b_i + h*sum_j A_ij*k^2*a_j = b0 + h*sum_j A_ij*N_b,j
    block_matrix = torch.cat([
        torch.cat([identity, -coupling], dim=-1),
        torch.cat([coupling, identity], dim=-1)], dim=-2)

    def mix(fields):
        return torch.einsum('ij,jxy->ixy', A.to(fields.dtype), fields)

    stages = torch.stack(step.stage_states(K))
    try:
        # FFT 系数是复数，a 和 b 仍是两个独立的实值场。
        factors = torch.linalg.lu_factor(block_matrix.to(a0_hat.dtype))
        for iteration in range(1, max_iterations + 1):
            a_stages, b_stages = stages[:, 0], stages[:, 1]
            rho = a_stages.square()+b_stages.square()
            nonlinear_a_hat = grid.fft(-2*rho*b_stages)
            nonlinear_b_hat = grid.fft(2*rho*a_stages)
            rhs_a = a0_hat[None]+h*mix(nonlinear_a_hat)
            rhs_b = b0_hat[None]+h*mix(nonlinear_b_hat)
            rhs = torch.cat([rhs_a, rhs_b], dim=0).movedim(0, -1).unsqueeze(-1)
            solution = torch.linalg.lu_solve(*factors, rhs).squeeze(-1).movedim(-1, 0)
            a_hat, b_hat = solution.split(step.stage_count, dim=0)
            a_stages = grid.ifft(a_hat).real
            b_stages = grid.ifft(b_hat).real
            stages = torch.stack([a_stages, b_stages], dim=1)
            K = list(grid.rhs(stages).unbind(0))
            rms = residual_rms(K)
            info.update(stage_polish_iterations=iteration, polished_rk_rms=rms)
            if not math.isfinite(rms):
                raise NewtonFailure('nonfinite_stage_polish_residual', info)
            if rms <= tolerance:
                return K, rms, info
    except torch.linalg.LinAlgError as error:
        raise NewtonFailure('stage_polish_linear_solve_failed', info) from error
    raise NewtonFailure('stage_polish_tolerance_not_met', info)


class HermiteExtension(nn.Module):
    """在实际松弛时间区间内插值，端点斜率取半离散方程右端。"""

    def __init__(self, grid, state_start, state_end, duration):
        super().__init__()
        self.grid = grid
        self.duration = duration
        self.register_buffer('state_start', state_start)
        self.register_buffer('state_end', state_end)
        self.register_buffer('slope_start', grid.rhs(state_start))
        self.register_buffer('slope_end', grid.rhs(state_end))

    def fields(self, s):
        """s 为归一化时间；返回状态和物理时间导数。"""
        s = s.reshape(-1, 1, 1, 1)
        H = self.duration
        # 使用端点差值，避免将两个接近的状态分别除以 H。
        delta = self.state_end - self.state_start
        f0, f1 = self.slope_start, self.slope_end
        blend = s.square()*(3.-2.*s)
        state = ((1.-blend)*self.state_start + blend*self.state_end
                 + H*s*(1.-s)*((1.-s)*f0-s*f1))
        slope = (6.*s*(1.-s)*(delta/H)
                 + (3.*s.square()-4.*s+1.)*f0 + (3.*s.square()-2.*s)*f1)
        return state, slope

    def interior_residual(self, interior_s):
        state, slope = self.fields(interior_s)
        return slope - self.grid.rhs(state)

    def endpoint_residual(self, change=None):
        """用一阶后向差分近似右端点时间导数。"""
        if change is None:
            change = self.state_end-self.state_start
        return change/self.duration - self.slope_end

    def residual_info(self, interior_s):
        residual = self.interior_residual(interior_s)
        time_mse = residual.square().mean(dim=(-3, -2, -1))
        mse = time_mse.mean()
        endpoint_residual = self.endpoint_residual()
        return dict(pde_mse=float(mse),
                    pde_rms=float(mse.sqrt()),
                    pde_max_time_rms=float(time_mse.max().sqrt()),
                    pde_interior_points=interior_s.numel(),
                    endpoint_backward_pde_rms=float(endpoint_residual.square().mean().sqrt()))


def differentiable_relaxation(step, K, targets, config):
    """前向用 Newton 解，反向用守恒方程的隐式导数。"""
    grid, y_n, h = step.grid, step.y_n, step.h
    d1, d2 = step.rk_directions(K)
    _, gamma2, info = solve_mrrk_parameters(step, d1.detach(), d2.detach(), targets, config)
    parameters = y_n.new_tensor([info['time_factor'], gamma2])
    D1, D2 = h*d1, h*d2
    difference = D2-D1
    change = parameters[0]*D1 + parameters[1]*difference
    F = (invariant_values(grid, y_n) - targets + grid.invariant_increment(y_n, change))[:2]

    with torch.no_grad():
        tolerance = y_n.new_tensor([config.mass_tol, config.energy_tol])
        J = torch.stack([torch.stack([(grad*direction).sum() for direction in (D1, difference)])
                         for grad in invariant_gradients(grid, y_n+change)[:2]]) / tolerance[:, None]
        scale = torch.linalg.vector_norm(J, dim=0).clamp_min(torch.finfo(y_n.dtype).tiny)
        J = J / scale
        if not bool(torch.isfinite(J).all()):
            raise NewtonFailure('nonfinite_endpoint_training_jacobian')
        singular_values = torch.linalg.svdvals(J)
        if float(singular_values[-1]) <= 1e-12*float(singular_values[0]):
            raise NewtonFailure('singular_endpoint_training_jacobian')

    # 前向修正恰为零。反向给出 dp = -J_original^{-1} dF。
    parameters = parameters - torch.linalg.solve(J, (F-F.detach())/tolerance) / scale
    time_factor, gamma2 = parameters.unbind()
    primary_change = time_factor*D1
    secondary_change = gamma2*difference
    return y_n+primary_change+secondary_change, h*time_factor, primary_change+secondary_change


class StageCorrector(nn.Module):
    """输入精修后的全部阶段、未松弛端点和配点标签，输出阶段修正。"""

    def __init__(self, polished_stages, polished_endpoint, nodes, width, modes):
        super().__init__()
        self.register_buffer('polished_stages', polished_stages.detach())
        self.register_buffer('polished_endpoint', polished_endpoint.detach())
        self.register_buffer('nodes', nodes)
        channels = (len(nodes)+1)*polished_stages.shape[1]+1
        self.network = FourierCNN(channels=channels, width=width, modes=modes)

    def forward(self):
        s, _, nx, ny = self.polished_stages.shape
        context = torch.cat([self.polished_stages.flatten(0, 1), self.polished_endpoint], dim=0)
        context = context.unsqueeze(0).expand(s, -1, -1, -1)
        labels = self.nodes.reshape(s, 1, 1, 1).expand(-1, 1, nx, ny)
        correction = self.network(torch.cat([context, labels], dim=1))
        # 对 c_i=0 的配点，阶段值就是左端点。
        return self.polished_stages + (labels != 0.)*correction


def correct_stages(step, polished_stages, polished_endpoint, baseline, targets, config):
    """用端点和 Hermite 步内 PDE loss 训练第二个 FNO。"""
    y_base, H_base, _, _, _, _ = baseline
    interior_s = torch.linspace(0., 1., config.corrector_points+2,
                                device=step.y_n.device, dtype=step.y_n.dtype)[1:-1]

    def residual_mses(y_end, duration, change=None):
        curve = HermiteExtension(step.grid, step.y_n, y_end, duration)
        return torch.stack([curve.endpoint_residual(change).square().mean(),
                            curve.interior_residual(interior_s).square().mean()])

    base_mses = residual_mses(y_base, H_base)
    # 按 Picard+Newton 基线的残差平方归一化，防止一阶端点差分项主导训练。
    scales = base_mses.detach().clamp_min(1e-24)
    weights = step.y_n.new_tensor([1., config.corrector_interior_weight])
    best_loss = float((weights*(base_mses/scales)).sum())
    endpoint_rms, interior_rms = base_mses.sqrt().tolist()
    info = dict(corrector_updates=0, corrector_selected_epoch=None,
                corrector_pde_before_rms=endpoint_rms, corrector_pde_after_rms=endpoint_rms,
                corrector_interior_before_rms=interior_rms, corrector_interior_after_rms=interior_rms,
                corrector_loss_before=best_loss, corrector_loss_after=best_loss,
                corrector_interior_points=config.corrector_points)
    best_stages, best_result = polished_stages, baseline
    if not config.corrector_adam:
        return best_stages, best_result, info

    model = StageCorrector(polished_stages, polished_endpoint, step.c,
                           config.width, min(config.modes, step.grid.n//2))
    model = model.to(step.y_n)
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.corrector_lr)

    def evaluate():
        stages = model()
        K = list(step.grid.rhs(stages).unbind(0))
        y_end, H, change = differentiable_relaxation(step, K, targets, config)
        mses = residual_mses(y_end, H, change)
        loss = (weights*(mses/scales)).sum()
        if not bool(torch.isfinite(loss)):
            raise NewtonFailure('nonfinite_corrector_loss')
        return stages, K, loss, mses

    try:
        _, _, loss, _ = evaluate()
    except (NewtonFailure, torch.linalg.LinAlgError) as error:
        info['corrector_stop_reason'] = str(error)
        return best_stages, best_result, info

    # Adam 给出更新方向。若该更新过大导致松弛失败，沿同一方向缩步。
    info['corrector_backtracks'] = 0
    for epoch in range(1, config.corrector_adam + 1):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 100.)
        if not bool(torch.isfinite(grad_norm)):
            info['corrector_stop_reason'] = 'nonfinite_corrector_gradient'
            break
        before = [value.detach().clone() for value in parameters]
        optimizer.step()
        direction = [value.detach()-old for value, old in zip(parameters, before)]
        for backtrack in range(13):
            fraction = 2.**(-backtrack)
            if backtrack:
                info['corrector_backtracks'] += 1
                with torch.no_grad():
                    for value, old, delta in zip(parameters, before, direction):
                        value.copy_(old + fraction*delta)
            try:
                stages, K, candidate_loss, mses = evaluate()
                score = float(candidate_loss.detach())
                if score >= best_loss:
                    reason = 'corrector_pde_loss_did_not_decrease'
                    continue
                candidate = multiple_relaxation(step, targets, config, K=[ki.detach() for ki in K])
            except (NewtonFailure, torch.linalg.LinAlgError) as error:
                reason = str(error)
                continue
            best_stages, best_result, best_loss = stages.detach(), candidate, score
            loss = candidate_loss
            optimizer.param_groups[0]['lr'] *= fraction
            endpoint_rms, interior_rms = mses.detach().sqrt().tolist()
            info.update(corrector_selected_epoch=epoch, corrector_pde_after_rms=endpoint_rms,
                        corrector_interior_after_rms=interior_rms, corrector_loss_after=best_loss,
                        corrector_updates=epoch, corrector_final_lr=optimizer.param_groups[0]['lr'])
            break
        else:
            info['corrector_stop_reason'] = reason
            break
    return best_stages, best_result, info


def invariant_values(grid, state):
    """固定顺序 M、E、P1、P2；不将向量动量压成单个标量。"""
    return torch.cat([torch.stack([grid.mass(state), grid.energy(state)], dim=-1),
                      grid.momentum(state)], dim=-1)


def invariant_tolerances(state, config):
    """质量和能量的 Newton 接受容差。"""
    return state.new_tensor([config.mass_tol, config.energy_tol])


def invariant_gradients(grid, state):
    """Gradients of M, E, Px, Py in the physical coordinates."""
    a, b = state.unbind(dim=-3)
    rho = a.square()+b.square()
    grad_mass = grid.cell_volume*state
    grad_energy = grid.cell_volume*torch.stack([
        -grid.laplacian(a)-2*rho*a, -grid.laplacian(b)-2*rho*b], dim=-3)
    grad_momentum = [grid.cell_volume*torch.stack([D(b), -D(a)], dim=-3)
                     for D in (grid.dx1, grid.dy1)]
    return grad_mass, grad_energy, *grad_momentum

@torch.no_grad()
def solve_mrrk_parameters(step, d1, d2, targets, config):
    """沿两个 RK 方向联立求解两个守恒约束。

    y_next=y_RK+sum_k gamma_k*Dk，Dk=h*dk。
    两个初值均为零，时间因子为 1+sum_k gamma_k。
    Jacobian 为 2x2；使用 SVD 最小二乘 Newton 步。
    两项均满足容差才算收敛；权重独立不保证 Jacobian 满秩或有根。
    """
    grid, y_n = step.grid, step.y_n
    D1, D2 = (step.h*d for d in (d1, d2))
    tolerance = invariant_tolerances(y_n, config)
    # 等价坐标 (s,gamma2)，s 为两个 gamma 之和。
    difference2 = D2 - D1
    directions = (D1, difference2)
    parameters = y_n.new_zeros(2)
    trace = []
    diagnostic = dict(trace=trace, parameter_bound=config.gamma_bound, h=step.h,
                      direction_difference_norm=float(torch.linalg.vector_norm(difference2)),
                      invariant_order=['mass', 'energy'],
                      rk_increment_norm=float(torch.linalg.vector_norm(D1)))

    def fail(reason):
        diagnostic.update(status='failed', reason=reason)
        raise NewtonFailure(reason, diagnostic)

    # 对全部守恒量作精确差值展开，减少消减误差。
    offsets = (invariant_values(grid, y_n)-targets)[:2]

    def equation(parameters):
        total_gamma, gamma2 = parameters
        change = (1. + total_gamma)*D1 + gamma2*difference2
        F = offsets+grid.invariant_increment(y_n, change)[:2]
        gradients = invariant_gradients(grid, y_n + change)[:2]
        J = torch.stack([torch.stack([(grad * direction).sum() for direction in directions])
                         for grad in gradients])
        return F, J

    for iteration in range(config.newton_iters + 1):
        F, J = equation(parameters)
        if not bool(torch.isfinite(F).all()) or not bool(torch.isfinite(J).all()):
            fail('nonfinite_mrrk_equation')
        total_gamma, gamma2 = parameters.cpu().tolist()
        gamma1 = total_gamma - gamma2
        trace.append(dict(iteration=iteration, gamma1=gamma1, gamma2=gamma2,
                          gamma_sum=total_gamma,
                          mass_error=float(F[0]), energy_error=float(F[1])))
        if bool((F.abs() <= tolerance).all()):
            # 两个原始守恒函数逐项复核。
            y_next = y_n + (1. + total_gamma)*D1 + gamma2*difference2
            actual = (invariant_values(grid, y_next) - targets)[:2]
            if not bool(torch.isfinite(actual).all()) or not bool((actual.abs() <= 2*tolerance).all()):
                fail('invariants_recheck_failed')
            diagnostic.update(status='converged', iterations=iteration,
                              gamma1=gamma1, gamma2=gamma2,
                              time_factor=1.+total_gamma,
                              mass_error=float(actual[0]), energy_error=float(actual[1]))
            return gamma1, gamma2, diagnostic
        if iteration == config.newton_iters:
            fail('max_newton_iterations')

        # 两个守恒量分别按容差缩放，回溯检查全部误差。
        scaled_F = F / tolerance
        scaled_J = J / tolerance[:, None]
        # 按列缩放，避免方向差很小被误判为没有独立方向。
        column_scale = torch.linalg.vector_norm(scaled_J, dim=0).clamp_min(torch.finfo(y_n.dtype).tiny)
        normalized_J = scaled_J / column_scale[None, :]
        try:
            U, singular_values, Vh = torch.linalg.svd(normalized_J, full_matrices=False)
        except RuntimeError:
            fail('jacobian_factorization_failed')
        active = singular_values > singular_values[0] * 1e-12
        rank = int(active.sum())
        trace[-1]['jacobian_rank'] = rank
        trace[-1]['scaled_jacobian_singular_values'] = singular_values.cpu().tolist()
        if rank == 0:
            fail('zero_mrrk_jacobian')
        projected = U.T @ scaled_F
        inverse = torch.zeros_like(singular_values)
        inverse[active] = 1. / singular_values[active]
        scaled_delta = Vh.T @ (inverse * projected)
        trace[-1]['linearized_unresolved_max'] = float(
            (normalized_J @ scaled_delta-scaled_F).abs().max())
        delta = scaled_delta / column_scale
        if not bool(torch.isfinite(delta).all()):
            fail('nonfinite_newton_update')

        merit = float(torch.linalg.vector_norm(scaled_F))
        damping = 1.
        parameter_blocks = 0
        time_blocks = 0
        best_candidate_merit = None
        trace[-1]['newton_delta'] = delta.cpu().tolist()
        for backtrack in range(31):
            candidate = parameters - damping * delta
            candidate_total, candidate_gamma2 = candidate.cpu().tolist()
            candidate_gamma1 = candidate_total - candidate_gamma2
            time_factor = 1. + candidate_total
            within_parameters = max(abs(candidate_gamma1), abs(candidate_gamma2)) <= config.gamma_bound
            within_time = .5 <= time_factor <= 1.5
            parameter_blocks += int(not within_parameters)
            time_blocks += int(not within_time)
            if within_parameters and within_time:
                candidate_F, _ = equation(candidate)
                candidate_merit = float(torch.linalg.vector_norm(candidate_F / tolerance))
                if math.isfinite(candidate_merit):
                    best_candidate_merit = candidate_merit if best_candidate_merit is None else min(best_candidate_merit, candidate_merit)
                if math.isfinite(candidate_merit) and candidate_merit < merit:
                    parameters = candidate
                    trace[-1]['damping'] = damping
                    break
            damping *= .5
        else:
            trace[-1].update(parameter_bound_blocks=parameter_blocks, time_bound_blocks=time_blocks,
                             best_candidate_merit=best_candidate_merit, current_merit=merit)
            if parameter_blocks == 31:

                fail('mrrk_parameter_bound_blocked')
            if time_blocks == 31:
                fail('mrrk_time_bound_blocked')
            fail('mrrk_line_search_failed')
    fail('max_newton_iterations')


@torch.no_grad()
def multiple_relaxation(step, targets, config, K=None):
    """预测 K -> 两个 RK 方向 -> 联立求两个 gamma -> 状态/时间更新。"""
    if K is None:
        K = step.predict_slopes()
    d1, d2 = step.rk_directions(K)
    gamma1, gamma2, diagnostic = solve_mrrk_parameters(step, d1, d2, targets, config)
    # 与求根器采用相同的差方向写法，减少大系数相消误差。
    time_factor = diagnostic['time_factor']
    D1, D2 = (step.h*d for d in (d1, d2))
    y_next = step.y_n + time_factor*D1 + gamma2*(D2-D1)
    physical_time_step = time_factor * step.h
    actual = (invariant_values(step.grid, y_next) - targets)[:2]
    tolerance = invariant_tolerances(y_next, config)
    if not bool(torch.isfinite(actual).all()) or not bool((actual.abs() <= 2*tolerance).all()):
        raise NewtonFailure('最终状态的质量/能量复核未通过', diagnostic)
    return y_next, physical_time_step, gamma1, gamma2, K, diagnostic


@torch.no_grad()
def zero_gamma_fallback(step, K, targets, error):
    """接受有限的普通 RK 端点；保留相对初始守恒量的真实误差。"""
    if not bool(torch.isfinite(torch.stack(K)).all()):
        raise NewtonFailure('nonfinite_zero_gamma_slopes') from error
    d1, *_ = step.rk_directions(K)
    y_next = step.y_n + step.h*d1
    if not bool(torch.isfinite(y_next).all()):
        raise NewtonFailure('nonfinite_zero_gamma_endpoint') from error
    actual = (invariant_values(step.grid, y_next)-targets)[:2]
    if not bool(torch.isfinite(actual).all()):
        raise NewtonFailure('nonfinite_zero_gamma_invariants') from error
    diagnostic = dict(error.diagnostic)
    trace = diagnostic.get('trace', [])
    diagnostic.update(status='zero_gamma_fallback', gamma_fallback=True,
                      fallback_reason=str(error), gamma1=0., gamma2=0., time_factor=1.,
                      iterations=trace[-1]['iteration'] if trace else 0,
                      mass_error=float(actual[0]), energy_error=float(actual[1]),
                      momentum_error=(step.grid.momentum(y_next)-targets[2:]).cpu().tolist())
    return y_next, step.h, 0., 0., K, diagnostic


def fit_step(step, targets, config, step_number, t, output_directory):
    """守恒或 PDE 验证失败时加训、减半；显式 zero 模式允许普通 RK 回退。"""
    total_refinements = 0
    for halving in range(config.max_step_halvings + 1):
        for refinement in range(config.max_refinements + 1):
            train_rk(step, config.adam, config.lbfgs, config.lr)
            with torch.no_grad():
                rk_rms = math.sqrt(float(step.rk_loss()))
            if not math.isfinite(rk_rms):
                raise RuntimeError('RK 残差不是有限数。')
            try:
                phase = 'stage_prediction'
                with torch.no_grad():
                    K = step.predict_slopes()
                    if config.stage_polish_iters:
                        phase = 'stage_polish'
                        K, rk_rms, stage_info = polish_rk_stages(
                            step, K, config.stage_polish_iters, config.stage_polish_tol,
                        )
                    else:
                        residuals = torch.stack(step.stage_residuals(step.stage_states(K), K))
                        rk_rms = float(residuals.square().mean().sqrt())
                        stage_info = dict(fno_rk_rms=rk_rms, stage_polish_iterations=0)
                phase = 'rk'
                if rk_rms > config.rk_tol:
                    raise NewtonFailure('rk_residual_exceeds_tolerance', dict(rk_rms=rk_rms))
                phase = 'relaxation'
                try:
                    result = multiple_relaxation(step, targets, config, K=K)
                except NewtonFailure as error:
                    if config.gamma_failure != 'zero':
                        raise
                    result = zero_gamma_fallback(step, K, targets, error)
                with torch.no_grad():
                    polished_stages = torch.stack(step.stage_states(K))
                    d1, _ = step.rk_directions(K)
                    polished_endpoint = step.y_n+step.h*d1
                stages = polished_stages
                corrector_info = {}
                phase = 'stage_corrector'
                if not result[-1].get('gamma_fallback', False):
                    stages, result, corrector_info = correct_stages(
                        step, polished_stages, polished_endpoint, result, targets, config)
                else:
                    corrector_info = dict(corrector_updates=0, corrector_selected_epoch=None,
                                          corrector_stop_reason='picard_endpoint_relaxation_failed')
                y_next, duration, gamma1, gamma2, K, diagnostic = result
                with torch.no_grad():
                    rebuilt = torch.stack(step.stage_states(K))
                    residuals = torch.stack(step.stage_residuals(list(rebuilt.unbind(0)), K))
                    corrector_info.update(
                        corrector_rk_rms=float(residuals.square().mean().sqrt()),
                        corrector_stage_defect_rms=float((stages-rebuilt).square().mean().sqrt()),
                        corrector_change_rms=float((stages-polished_stages).square().mean().sqrt()))
                    stage_outputs = dict(Y_picard=polished_stages, state_picard=polished_endpoint,
                                         Y=stages, Y_rk_reconstructed=rebuilt)
                    curve = HermiteExtension(step.grid, step.y_n, y_next, duration).to(step.y_n)
                    validation = torch.linspace(0., 1., 2*config.pde_points-1,
                                                device=step.y_n.device, dtype=step.y_n.dtype)
                    pde_info = curve.residual_info(validation)
                    pde_info.update(pde_validation_points=len(validation))
                    if not math.isfinite(pde_info['pde_max_time_rms']):
                        raise NewtonFailure('nonfinite_hermite_pde_validation', pde_info)
                    if pde_info['pde_max_time_rms'] > config.pde_tol:
                        raise NewtonFailure('hermite_pde_tolerance_not_met', pde_info)
                diagnostic.update(stage_info)
                diagnostic.update(corrector_info)
                if diagnostic.get('gamma_fallback', False):
                    print(f'FALLBACK step={step_number}, h={step.h:.6g}, gamma1=gamma2=0, '
                          f'reason={diagnostic["fallback_reason"]}', flush=True)
                return result, rk_rms, total_refinements, halving, curve, pde_info, stage_outputs
            except NewtonFailure as error:
                failure = dict(step=step_number, t=t, h=step.h, halving=halving,
                               refinement=refinement, rk_rms=rk_rms, phase=phase,
                               reason=str(error), diagnostic=error.diagnostic)
                with (output_directory / 'retry_history.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(failure, ensure_ascii=False) + '\n')
                if refinement < config.max_refinements:
                    total_refinements += 1
                    continue
                next_h = .5 * step.h
                if halving < config.max_step_halvings and next_h >= config.min_dt:
                    print(f'RETRY step={step_number}, h={step.h:.6g} -> {next_h:.6g}, '
                          f'reason={error}', flush=True)
                    # y_n 不变。h 进入阶段方程，因此减半后必须重新训练。
                    step.h = next_h
                    break
                failure.update(min_dt=config.min_dt, max_step_halvings=config.max_step_halvings)
                (output_directory / 'failure.json').write_text(
                    json.dumps(failure, indent=2, ensure_ascii=False), encoding='utf-8',
                )
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=32, help='每方向网格点数；二维共 n^2 个点，偶数且至少为 4')
    parser.add_argument('--L', type=float, default=10., help='Diamond: |x|+|y| <= sqrt(2)*L')
    parser.add_argument('--length', type=float, default=None, help='Rotated square side length; overrides 2*L')
    parser.add_argument('--nu', type=float, default=0.1, help='初始复相位的系数')
    parser.add_argument('--t-end', type=float, default=5., help='最少推进到的物理时间')
    parser.add_argument('--mass-tol', type=float, default=1e-14)
    parser.add_argument('--dt', type=float, default=0.1, help='名义 RK 步长 h')
    parser.add_argument('--steps', type=int, default=500, help='最少接受步数；不足 t-end 时继续推进')
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--modes', type=int, default=4, help='二维 FNO 每个方向保留的低频模态数')
    parser.add_argument('--adam', type=int, default=400)
    parser.add_argument('--lbfgs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--rk-tol', type=float, default=1e-2, help='阶段斜率残差 RMS')
    parser.add_argument('--stage-polish-iters', type=int, default=30,
                        help='默认最多 30 次阶段精修；设为 0 可关闭 polish')
    parser.add_argument('--stage-polish-tol', type=float, default=1e-10,
                        help='数值精修后的阶段残差 RMS 容差')
    parser.add_argument('--energy-tol', type=float, default=1e-12)
    parser.add_argument('--momentum-tol', type=float, default=1e-12, help='两个动量分量分别使用的绝对守恒容差')
    parser.add_argument('--newton-iters', type=int, default=10)
    parser.add_argument('--gamma-bound', type=float, default=0.5, help='两个参数分别的近零搜索半径')
    parser.add_argument('--gamma-failure', choices=['zero', 'stop'], default='stop',
                        help='stop 重试并最终报错；zero 允许两个 gamma 均为零的非守恒回退')
    parser.add_argument('--corrector-adam', type=int, default=10, help='第二 FNO 的 Adam 次数；0 关闭阶段修正')
    parser.add_argument('--corrector-lr', type=float, default=1e-4)
    parser.add_argument('--corrector-points', type=int, default=9, help='第二网络 Hermite 内部训练点数')
    parser.add_argument('--corrector-interior-weight', type=float, default=1.)
    parser.add_argument('--pde-points', type=int, default=9, help='Hermite PDE 验证点数，含两端')
    parser.add_argument('--pde-tol', type=float, default=1e-2, help='验证点中最差时间切片的 PDE RMS 容差')
    parser.add_argument('--max-refinements', type=int, default=3)
    parser.add_argument('--max-step-halvings', type=int, default=2, help='当前步失败后的最大减半次数')
    parser.add_argument('--min-dt', type=float, default=None, help='最小名义步长，默认 dt/64')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', default='nls2d_fno_mrrk_output')
    parser.add_argument('--method', choices=['gauss', 'lobatto', 'both'], default='both',
                        help='Gauss: 3 stages, order 6; Lobatto IIIA: 4 stages, order 6; both: run separately')
    config = parser.parse_args(argv)
    if config.length is None:
        config.length = 2*config.L
    config.L = config.length/2

    if min(config.dt, config.length, config.mass_tol, config.steps, config.width, config.modes,
           config.rk_tol, config.energy_tol, config.momentum_tol, config.newton_iters,
           config.gamma_bound, config.lr, config.threads, config.corrector_lr,
           config.corrector_points, config.pde_tol, config.stage_polish_tol) <= 0:
        parser.error('步长、区间长度、步数、宽度、模态数、容差、学习率和线程数必须为正。')
    if min(config.adam, config.lbfgs, config.corrector_adam,
           config.max_refinements, config.max_step_halvings, config.stage_polish_iters) < 0:
        parser.error('优化次数和加训次数不能为负。')
    if not math.isfinite(config.corrector_interior_weight) or config.corrector_interior_weight < 0:
        parser.error('--corrector-interior-weight 必须为有限非负数。')
    if config.t_end < 0:
        parser.error('--t-end 不能为负。')
    if config.pde_points < 3:
        parser.error('--pde-points 至少为 3，覆盖起点、内部和终点。')
    if config.min_dt is None:
        config.min_dt = config.dt / 64.
    if not 0. < config.min_dt <= config.dt:
        parser.error('--min-dt 必须为正且不大于 --dt。')

    summaries = {}
    methods = ('gauss', 'lobatto') if config.method == 'both' else (config.method,)
    for choice in methods:
        method = {'gauss': 'gauss_legendre', 'lobatto': 'lobatto_iiia'}[choice]
        method_config = argparse.Namespace(**vars(config))
        method_config.method = choice
        method_config.output = str(Path(config.output) / method)
        print(f'Running {method}: output={method_config.output}', flush=True)
        summaries[method] = run_method(method_config, method)
    return summaries


def run_method(config, method):
    method_label, stage_count, main_order = TABLEAU_METHODS[method]
    tableau_factory = (gauss_legendre_tableau if method == 'gauss_legendre'
                       else lobatto_iiia_tableau)
    device = configure(config.threads, config.seed, config.device)
    grid = PeriodicGrid(config.n, length=config.length, device=device, nu=config.nu)

    y = initial_state(grid).detach()
    mass_target = grid.mass(y).detach()
    energy_target = grid.energy(y).detach()
    momentum_target = grid.momentum(y).detach()
    targets = invariant_values(grid, y).detach()
    # 相对能量误差以初值的绝对值为分母。
    initial_energy = float(energy_target)
    energy_scale = max(abs(initial_energy), torch.finfo(y.dtype).eps)
    output_directory = Path(config.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    step_directory = output_directory / 'steps'
    step_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(step_directory / 'step_00000000.npz',
                        **grid.coordinate_arrays(),
                        t=0., state=y.cpu().numpy(), momentum=momentum_target.cpu().numpy(),
                        energy=float(energy_target), mass=float(mass_target),
                        complex_state=(y[0]+1j*y[1]).cpu().numpy())

    t = 0.
    times = [t]
    state_times = [t]
    states = [y.cpu().numpy()]
    snapshots = {0.: states[0].copy()}
    history = []
    previous_network = None
    next_h = config.dt
    step_number = 0
    # 将实际使用的表保存到输出目录，避免不同方法的结果混淆。
    a, b1, b2, c_nodes = tableau_factory(y)
    (output_directory / 'butcher_tableau.json').write_text(
        json.dumps(dict(method=f'{method_label} collocation, {stage_count} stages',
                        main_order=main_order, embedded_order=1,
                        a=a.cpu().tolist(), b1=b1.cpu().tolist(), b2=b2.cpu().tolist(),
                        c=c_nodes.cpu().tolist()), indent=2), encoding='utf-8',
    )

    # 自适应减小 h 后不能再用初始 dt 推算最大步数，直接按实际 t 判断。
    while step_number < config.steps or t < config.t_end:
        step_number += 1
        step = RKStep(grid, y, h=next_h, width=config.width, modes=config.modes,
                      tableau=method).to(device)
        if previous_network is not None:
            step.network.load_state_dict(previous_network)

        result, rk_rms, refinements, halvings, curve, pde_info, stage_outputs = fit_step(
            step, targets, config, step_number, t, output_directory,
        )
        y_next, time_step, gamma1, gamma2, K, diagnostic = result
        next_h = step.h

        mass_error = float(grid.mass(y_next) - mass_target)
        gamma_fallback = diagnostic.get('gamma_fallback', False)
        actual_errors = (invariant_values(grid, y_next)-targets)[:2]
        if (not bool(torch.isfinite(actual_errors).all()) or
                (not gamma_fallback and not bool((actual_errors.abs() <=
                                                 2*invariant_tolerances(y_next, config)).all()))):
            raise NewtonFailure('invariants_recheck_failed', dict(errors=actual_errors.cpu().tolist()))
        t_start = t
        y = y_next.detach()
        t = t + time_step
        previous_network = {
            name: value.detach().clone()
            for name, value in step.network.state_dict().items()
        }
        energy_error = float(grid.energy(y) - energy_target)
        with torch.no_grad():
            d1, d2 = step.rk_directions(K)
            grad_M, grad_E, *grad_P = invariant_gradients(grid, y)
            rhs_value = grid.rhs(y)
            mass_rhs_rate = float((grad_M * rhs_value).sum())
            energy_rhs_rate = float((grad_E * rhs_value).sum())
            momentum_rhs_rate = [float((grad*rhs_value).sum()) for grad in grad_P]
            correction = step.h*((diagnostic['time_factor']-1.)*d1 + gamma2*(d2-d1))
            correction_norm = float(torch.linalg.vector_norm(correction))
            increment_norm = float(torch.linalg.vector_norm(step.h*d1))
        row = {
            'step': step_number,
            't': t,
            'h': step.h,
            'physical_time_step': time_step,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'gamma_fallback': gamma_fallback,
            'gamma_fallback_reason': diagnostic.get('fallback_reason'),
            'max_abs_gamma': max(abs(gamma1), abs(gamma2)),
            'time_factor': diagnostic['time_factor'],
            'rk_rms': rk_rms,
            'fno_rk_rms': diagnostic['fno_rk_rms'],
            'stage_polish_iterations': diagnostic['stage_polish_iterations'],
            'mass': float(grid.mass(y)),
            'mass_error': mass_error,
            'energy': initial_energy + energy_error,
            'energy_error': energy_error,
            'momentum': grid.momentum(y).cpu().tolist(),
            'momentum_error': (grid.momentum(y)-momentum_target).cpu().tolist(),
            'max_momentum_error': float((grid.momentum(y)-momentum_target).abs().max()),
            'momentum_constraint_components': [],
            'relative_energy_error': abs(energy_error) / energy_scale,
            'mass_rhs_rate': mass_rhs_rate,
            'energy_rhs_rate': energy_rhs_rate,
            'momentum_rhs_rate': momentum_rhs_rate,
            'correction_norm': correction_norm,
            'relative_correction': correction_norm / max(increment_norm, torch.finfo(y.dtype).eps),
            'newton_iterations': diagnostic['iterations'],
            'refinements': refinements,
            'step_halvings': halvings,
            **pde_info,
            **{name: value for name, value in diagnostic.items() if name.startswith('corrector_')},
        }
        history.append(row)
        times.append(t)
        save_full_state = t <= SOLUTION_SAVE_END
        if save_full_state:
            state_times.append(t)
            states.append(y.cpu().numpy())
        print(
            f'step={step_number:02d}, t={t:.6f}, h={step.h:.4g}, '
            f'gamma1={gamma1:+.4e}, gamma2={gamma2:+.4e}, '
            f'RK={rk_rms:.2e}, dE={row["energy_error"]:+.4e}, '
            f'dM={mass_error:+.4e}, max|dP|={row["max_momentum_error"]:.2e}, '
            f'PDE={pde_info["pde_max_time_rms"]:.2e}',
            flush=True,
        )

        with torch.no_grad():
            if save_full_state:
                dense_s = torch.linspace(0., 1., 2*config.pde_points-1, device=y.device, dtype=y.dtype)
                dense_u, _ = curve.fields(dense_s)
                dense_states = dense_u.cpu().numpy()
            # 指定物理时刻从当前区间的三次 Hermite 曲线取值。
            for requested_time in SNAPSHOT_TIMES:
                if requested_time not in snapshots and t_start <= requested_time <= t:
                    s = y.new_tensor([(requested_time-t_start)/time_step])
                    snapshot_u, _ = curve.fields(s)
                    snapshots[requested_time] = snapshot_u[0].cpu().numpy()
        if save_full_state:
            torch.save(dict(state_dict=curve.state_dict(), duration=time_step,
                            interpolation='cubic Hermite', width=config.width, modes=config.modes,
                            n=grid.n, length=grid.length,
                            coefficients=dict(nonlinearity=2., nu=grid.nu),
                            t_start=t_start, t_end=t, pde_info=pde_info),
                       step_directory / f'continuous_{step_number:08d}.pt')
            np.savez_compressed(
                step_directory / f'step_{step_number:08d}.npz',
                **grid.coordinate_arrays(),
                t=t, t_start=t-time_step, h=step.h,
                state_start=step.y_n.cpu().numpy(), state=y.cpu().numpy(),
                rk_state=(step.y_n+step.h*d1).cpu().numpy(),
                K=torch.stack(K).cpu().numpy(),
                **{name: value.cpu().numpy() for name, value in stage_outputs.items()},
                d1=d1.cpu().numpy(), d2=d2.cpu().numpy(),
                b1=step.b1.cpu().numpy(), b2=step.b2.cpu().numpy(),
                a=step.a.cpu().numpy(), c=step.c.cpu().numpy(),
                gamma1=gamma1, gamma2=gamma2, time_factor=diagnostic['time_factor'],
                gamma_fallback=gamma_fallback,
                energy=float(grid.energy(y)), complex_state=(y[0]+1j*y[1]).cpu().numpy(),
                mass=float(grid.mass(y)), mass_error=mass_error,
                energy_error=energy_error,
                momentum=grid.momentum(y).cpu().numpy(), momentum_error=(grid.momentum(y)-momentum_target).cpu().numpy(),
                dense_s=dense_s.cpu().numpy(), dense_t=t_start+time_step*dense_s.cpu().numpy(),
                dense_states=dense_states,
                pde_rms=pde_info['pde_rms'], pde_max_time_rms=pde_info['pde_max_time_rms'],
            )
            # 保存后释放步内批量采样，避免在后续长时间推进中持续占用内存/显存。
            del dense_s, dense_u, dense_states
        (step_directory / f'newton_{step_number:08d}.json').write_text(
            json.dumps(diagnostic, indent=2), encoding='utf-8',
        )
        (output_directory / 'history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8',
        )

    summary = {
        'config': vars(config),
        'final_time': t,
        'steps': len(history),
        'max_energy_error': max(abs(row['energy_error']) for row in history),
        'initial_energy': initial_energy,
        'initial_mass': float(mass_target),
        'initial_momentum': momentum_target.cpu().tolist(),
        'max_mass_error': max(abs(row['mass_error']) for row in history),
        'max_momentum_error': max(row['max_momentum_error'] for row in history),
        'invariant_definitions': dict(mass='0.5*integral(|u|^2)',
            energy='0.5*integral(|grad(u)|^2-|u|^4)',
            momentum='0.5*Im(integral(conj(u)*grad(u)))'),
        'equations': ['i*u_t + Delta(u) + 2*|u|^2*u = 0'],
        'nonlinearity': dict(f='2*rho', F='rho^2', rho='|u|^2'),
        'state_components': ['real_u', 'imag_u'],
        'spatial_axis_order': ['q', 'r'],
        'coordinate_transform': 'q=(x+y)/sqrt(2), r=(y-x)/sqrt(2)',
        'domain': '|x|+|y| <= sqrt(2)*L',
        'boundary': 'periodic in rotated coordinates q,r on [-L,L)^2',
        'initial_conditions': dict(
            u='sum_k exp(-(x-(-1)^k*2)^2-y^2)*exp(i*nu*(x-(-1)^k*2)+y), k=0,1'),
        'snapshot_times': list(SNAPSHOT_TIMES),
        'solution_save_end': SOLUTION_SAVE_END,
        'saved_state_count': len(states),
        'last_saved_state_time': state_times[-1],
        'method': ('2D FNO stages'
                   + (f' + numerical {method_label} stage polish' if config.stage_polish_iters else '')
                   + f' + two-parameter {method_label} MRRK + independent cubic Hermite'),
        'tableau': method,
        'stage_count': stage_count,
        'main_rk_order': main_order,
        'embedded_rk_order': 1,
        'weight_matrix_rank': 2,
        'structurally_rank_deficient': False,
        'relaxation_parameters': 2,
        'momentum_constraint_components': [],
        'momentum_diagnostic_components': ['P1', 'P2'],
        'invariant_order': ['mass', 'energy'],
        'relaxation_update': 'y_next=y_n+h*((1+gamma1)*d1+gamma2*d2)',
        'time_update': 't_next=t_n+h*(1+gamma1+gamma2)',
        'constraint_solver': 'damped column-scaled SVD minimum-norm Newton, 2 constraints / 2 parameters',
        'max_abs_gamma': max(row['max_abs_gamma'] for row in history),
        'gamma_fallback_steps': sum(row['gamma_fallback'] for row in history),
        'all_steps_passed_invariant_checks': not any(row['gamma_fallback'] for row in history),
        'spatial_step': grid.dx,
        'max_mass_rhs_rate': max(abs(row['mass_rhs_rate']) for row in history),
        'max_energy_rhs_rate': max(abs(row['energy_rhs_rate']) for row in history),
        'max_momentum_rhs_rate': max(abs(rate) for row in history for rate in row['momentum_rhs_rate']),
        'max_relative_correction': max(row['relative_correction'] for row in history),
        'relative_energy_denominator': energy_scale,
        'max_relative_energy_error': max(row['relative_energy_error'] for row in history),
        'max_rk_rms': max(row['rk_rms'] for row in history),
        'max_pde_rms': max(row['pde_rms'] for row in history),
        'max_pde_time_rms': max(row['pde_max_time_rms'] for row in history),
        'min_accepted_h': min(row['h'] for row in history),
        'total_step_halvings': sum(row['step_halvings'] for row in history),
    }
    (output_directory / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8',
    )
    save_plots(output_directory, grid.x.cpu().numpy(), times, states, history, snapshots, state_times)
    return summary


if __name__ == '__main__':
    main()
