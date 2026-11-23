# -*- coding: utf-8 -*-
"""二维周期 Schrödinger--Klein--Gordon 耦合算例4.3.1。

i*u_t + kappa1/2*Delta(u) + xi*u*v = 0，
v_tt - kappa2*Delta(v) + mu^2*v - xi*|u|^2 = 0。
默认区域 [-20,20)^2，不重复右端点；state=(Re(u),Im(u),v,w)，w=v_t。

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
    """给定的二维初值，返回 (4,N,N)，空间轴顺序为 (x1,x2)。"""
    x1, x2 = torch.meshgrid(grid.x, grid.x, indexing='ij')
    a, b, v = torch.zeros_like(x1), torch.zeros_like(x1), torch.zeros_like(x1)
    for k in range(2):
        center = (-1)**k * 2.
        amplitude = torch.exp(-(x1-center).square()-x2.square()+x2)
        phase = grid.mu*(x1-center)
        a = a + amplitude*torch.cos(phase)
        b = b + amplitude*torch.sin(phase)
        for j in range(2):
            v = v - torch.exp(-(x1-center).square()-(x2-(-1)**j*2.).square())
    w = torch.exp(-x1.square()-x2.square())
    return torch.stack([a, b, v, w], dim=0)


class PeriodicGrid:
    def __init__(self, n, length, device, kappa1=-.4, kappa2=.1, xi=.2, mu=.1):
        if n < 32 or n % 2:
            raise ValueError('n 需为不小于32的偶数')
        self.n, self.length = n, length
        self.dx = length/n
        self.cell_area = self.dx**2
        self.kappa1, self.kappa2, self.xi, self.mu = kappa1, kappa2, xi, mu
        self.x = -length/2+self.dx*torch.arange(n, device=device)
        k = 2*math.pi*torch.fft.fftfreq(n, d=self.dx, device=device)
        # 沿用 Nyquist 置零约定，Delta=D_x1^2+D_x2^2，保证离散分部积分。
        k[n//2] = 0.
        self.k1, self.k2 = k[:, None], k[None, :]
        self.k_squared = self.k1.square()+self.k2.square()

    def dx1(self, f):
        return torch.fft.ifft2(1j*self.k1*torch.fft.fft2(f)).real

    def dy1(self, f):
        return torch.fft.ifft2(1j*self.k2*torch.fft.fft2(f)).real

    def laplacian(self, f):
        return torch.fft.ifft2(-self.k_squared*torch.fft.fft2(f)).real

    def gradient_squared(self, f):
        return self.dx1(f).square()+self.dy1(f).square()

    def rhs(self, state):
        a, b, v, w = state.unbind(dim=-3)
        rho = a.square()+b.square()
        return torch.stack([
            -.5*self.kappa1*self.laplacian(b)-self.xi*b*v,
            .5*self.kappa1*self.laplacian(a)+self.xi*a*v,
            w,
            self.kappa2*self.laplacian(v)-self.mu**2*v+self.xi*rho,
        ], dim=-3)

    def mass(self, state):
        return self.cell_area*state[..., :2, :, :].square().sum(dim=(-3, -2, -1))

    def energy(self, state):
        a, b, v, w = state.unbind(dim=-3)
        density = .5*(self.kappa1*(self.gradient_squared(a)+self.gradient_squared(b))
                       + self.kappa2*self.gradient_squared(v)+w.square()+self.mu**2*v.square())
        density = density-self.xi*(a.square()+b.square())*v
        return self.cell_area*density.sum(dim=(-2, -1))

    def invariant_increment(self, state, change):
        """精确多项式差值展开，避免 Newton 中两个总量相减的消减误差。"""
        a, b, v, w = state.unbind(dim=-3)
        da, db, dv, dw = change.unbind(dim=-3)
        rho = a.square()+b.square()
        drho = 2*a*da+da.square()+2*b*db+db.square()
        quadratic = torch.zeros_like(v)
        for derivative in (self.dx1, self.dy1):
            old, delta = derivative(state), derivative(change)
            cross = 2*old*delta+delta.square()
            quadratic = quadratic+self.kappa1*(cross[..., 0, :, :]+cross[..., 1, :, :]) \
                + self.kappa2*cross[..., 2, :, :]
        quadratic = quadratic+2*w*dw+dw.square()+self.mu**2*(2*v*dv+dv.square())
        energy_change = .5*quadratic-self.xi*(rho*dv+drho*v+drho*dv)
        return self.cell_area*torch.stack([drho.sum(dim=(-2, -1)),
                                           energy_change.sum(dim=(-2, -1))], dim=-1)


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
        result[..., :m1, :m2] = torch.einsum(
            'bixy,ioxy->boxy', zhat[..., :m1, :m2], weights[0, ..., :m1, :m2])
        result[..., -m1:, :m2] = torch.einsum(
            'bixy,ioxy->boxy', zhat[..., -m1:, :m2], weights[1, ..., :m1, :m2])
        return torch.fft.irfft2(result, s=z.shape[-2:])

class FourierCNN(nn.Module):
    def __init__(self, channels, width, modes):
        super().__init__()
        self.lift = nn.Conv2d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(2)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(2)])
        self.project = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.project.weight); nn.init.zeros_(self.project.bias)

    def forward(self, features):
        z = self.lift(features)
        for spectral, local in zip(self.spectral, self.local):
            z = torch.tanh(spectral(z)+local(z))
        # 无边界因子、无端点投影；保留有序周期网格及逆 FFT 的周期解释。
        return self.project(z)


SNAPSHOT_TIMES = (0., 2., 3., 5.)


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


def save_plots(output_directory, x, times, states, history, snapshots):
    """守恒图与演化图；指定时刻从训练后的第二网络取值。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = np.asarray(times)
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

    fig, ax = plt.subplots(figsize=(9, 5))
    # 对 x2 求平均，得到二维解沿 x1 的时间演化概览。
    amplitude = np.sqrt(np.sum(states[:, :2]**2, axis=1)).mean(axis=-1)
    mesh = ax.pcolormesh(x, times, amplitude, shading='auto', cmap='viridis')
    fig.colorbar(mesh, ax=ax, label='mean of |u| over x2')
    ax.set(xlabel='x1', ylabel='Physical time t', title='Evolution of mean |u|')
    fig.tight_layout()
    fig.savefig(output_directory / 'long_time_evolution.png', dpi=200)
    plt.close(fig)

    snapshot_states = []
    fig, axes = plt.subplots(2, len(SNAPSHOT_TIMES), figsize=(16, 7), sharex=True, sharey=True)
    available_times = []
    for column, requested_time in enumerate(SNAPSHOT_TIMES):
        if not times[0] <= requested_time <= times[-1]:
            for ax in axes[:, column]:
                ax.set_visible(False)
            continue
        snapshot = snapshots[requested_time]
        snapshot_states.append(snapshot)
        available_times.append(requested_time)
    # 每个物理量在所有快照间共用色标，便于比较演化。
    for row, label in enumerate(('|u|', 'v')):
        fields = [np.sqrt(s[0]**2+s[1]**2) if row == 0 else s[2]
                  for s in snapshot_states]
        if not fields:
            continue
        lower, upper = min(f.min() for f in fields), max(f.max() for f in fields)
        for column, requested_time in enumerate(SNAPSHOT_TIMES):
            if requested_time not in available_times:
                continue
            ax = axes[row, column]
            field = fields[available_times.index(requested_time)]
            mesh = ax.pcolormesh(x, x, field.T, shading='auto', cmap='viridis',
                                 vmin=lower, vmax=upper)
            ax.set(title=f'{label}, t = {requested_time:g}', xlabel='x1', ylabel='x2', aspect='equal')
            fig.colorbar(mesh, ax=ax, label=label)
    fig.suptitle('Time snapshots from the PDE-trained continuous FNO')
    fig.tight_layout()
    fig.savefig(output_directory / 'time_snapshots.png', dpi=200)
    plt.close(fig)
    np.savez_compressed(
        output_directory / 'time_snapshots.npz', x1=x, x2=x,
        t=np.asarray(available_times), states=np.asarray(snapshot_states),
        state_components=np.asarray(['real_u', 'imag_u', 'v', 'v_t']),
        interpolation='PDE-trained continuous 2D FNO',
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(times[1:], np.maximum([row['pde_rms'] for row in history], 1e-17), label='Validation RMS')
    ax.semilogy(times[1:], np.maximum([row['pde_max_time_rms'] for row in history], 1e-17), label='Maximum time-slice RMS')
    ax.set(xlabel='Physical time t', ylabel='PDE residual', title='Continuous-network PDE residual')
    ax.legend()
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(output_directory / 'pde_residual.png', dpi=200)
    plt.close(fig)


class RKStep(nn.Module):
    """当前一步的 RK 方程；每个数学量都保留自己的名称。

    y_n.shape = (4, N, N)：当前 Re(u)、Im(u)、v、v_t 状态。
    K[i].shape = (4, N, N)：网络预测的第 i 个阶段斜率。
    Y[i].shape = (4, N, N)：由所有 K_j 和 RK 系数重建的第 i 个阶段状态。
    i = 0, 1, 2 对应数学下标 i = 1, 2, 3。
    """

    def __init__(self, grid, y_n, h, width=16, modes=32):
        super().__init__()
        self.grid = grid
        self.h = h
        self.register_buffer('y_n', y_n.detach().clone())

        # 只在这里选择时间离散的 Butcher 表；FNO 和空间导数不变。
        a, b1, b2, c = gauss_legendre_tableau(y_n)
        self.stage_count = len(c)
        self.register_buffer('a', a)
        self.register_buffer('b1', b1)
        self.register_buffer('b2', b2)
        self.register_buffer('c', c)

        # 输入：四个状态分量与阶段 c_i；输出四个分量的斜率。
        # 模态数不超过当前网格可表示的实数 Fourier 频率数。
        self.network = FourierCNN(channels=5, width=width, modes=min(modes, grid.n // 2))

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
        """相同 K_i，两套权重构造 d1、d2；这里不乘 h。"""
        d1 = torch.zeros_like(self.y_n)
        d2 = torch.zeros_like(self.y_n)
        for i in range(self.stage_count):
            d1 = d1 + self.b1[i] * K[i]
            d2 = d2 + self.b2[i] * K[i]
        return d1, d2

    def rk_endpoint(self, K):
        """普通 RK 终点，使用第一套权重。"""
        d1, _ = self.rk_directions(K)
        return self.y_n + self.h * d1


def train_rk(step, adam_iterations, lbfgs_iterations, learning_rate):
    """用优化器求解三个隐式阶段方程，不使用参考解作标签。"""
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


class ContinuousFNO(nn.Module):
    """四分量 Hermite 连续解；端点导数取耦合 rhs，修正不改变端点。"""
    def __init__(self, grid, state_start, state_end, duration, width, modes, terms):
        super().__init__()
        self.grid = grid
        self.duration = float(duration)
        self.terms = terms
        self.width, self.modes = width, modes
        self.register_buffer('state_start', state_start.detach().clone())
        self.register_buffer('state_end', state_end.detach().clone())
        self.register_buffer('slope_start', grid.rhs(state_start).detach())
        self.register_buffer('slope_end', grid.rhs(state_end).detach())
        self.network = FourierCNN(channels=9, width=width, modes=min(modes, grid.n//2))

    def coefficients(self):
        context = torch.cat([self.state_start, self.state_end], dim=0)
        fields = []
        for k in range(self.terms):
            label = 0. if self.terms == 1 else -1. + 2.*k/(self.terms-1)
            features = torch.cat([context, torch.full_like(context[:1], label)], dim=0)[None]
            value = self.network(features)[0]
            fields.append(value)
        return torch.stack(fields, dim=0)

    def fields(self, s):
        """返回状态及其时间导数，形状均为 (时间点数,4,N,N)。"""
        s = s.reshape(-1, 1, 1, 1)
        H = self.duration
        u0, u1 = self.state_start, self.state_end
        f0, f1 = self.slope_start, self.slope_end
        u = (2*s**3-3*s**2+1)*u0 + (s**3-2*s**2+s)*H*f0 \
            + (-2*s**3+3*s**2)*u1 + (s**3-s**2)*H*f1
        ut = ((6*s**2-6*s)*u0 + (3*s**2-4*s+1)*H*f0 \
              + (-6*s**2+6*s)*u1 + (3*s**2-2*s)*H*f1) / H
        a = self.coefficients()
        z = 2*s-1
        C, Cs = torch.zeros_like(u), torch.zeros_like(u)
        for k in range(self.terms):
            C = C + z**k*a[k]
            if k:
                Cs = Cs + 2*k*z**(k-1)*a[k]
        envelope = s**2*(1-s)**2
        envelope_s = 2*s-6*s**2+4*s**3
        return u+H**2*envelope*C, ut+H*(envelope_s*C+envelope*Cs)

    def pde_residual(self, s):
        u, ut = self.fields(s)
        return ut - self.grid.rhs(u)


def train_continuous(curve, config):
    """只优化第二网络。端点、H 和两个 gamma 均固定，不用参考解作标签。"""
    s_train = torch.linspace(0., 1., config.pde_points, device=curve.state_start.device,
                             dtype=curve.state_start.dtype)
    s_validation = torch.linspace(0., 1., 2*config.pde_points-1, device=s_train.device,
                                  dtype=s_train.dtype)
    params = list(curve.network.parameters())
    best = None
    best_loss = math.inf

    def objective():
        # 两个复数 u 分量、v_t=w 与 w_t 的残差一起训练。
        return config.pde_weight * curve.pde_residual(s_train).square().mean()

    def remember(value):
        nonlocal best, best_loss
        score = float(value.detach())
        if not math.isfinite(score):
            raise NewtonFailure('nonfinite_continuous_pde_loss')
        if score < best_loss:
            best_loss = score
            best = {name: p.detach().clone() for name, p in curve.network.state_dict().items()}

    if config.pde_adam:
        optimizer = torch.optim.Adam(params, lr=config.pde_lr)
        for epoch in range(config.pde_adam):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            remember(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 100.)
            optimizer.step()
    if config.pde_lbfgs:
        optimizer = torch.optim.LBFGS(params, max_iter=config.pde_lbfgs,
                                     tolerance_grad=1e-10, tolerance_change=1e-14,
                                     line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            remember(loss)
            loss.backward()
            return loss

        optimizer.step(closure)
    remember(objective())
    curve.network.load_state_dict(best)
    with torch.no_grad():
        residual = curve.pde_residual(s_validation)
        time_rms = residual.square().mean(dim=(-3, -2, -1)).sqrt()
        info = dict(pde_train_rms=math.sqrt(best_loss/config.pde_weight),
                    pde_rms=float(residual.square().mean().sqrt()),
                    pde_max_time_rms=float(time_rms.max()),
                    pde_start_rms=float(time_rms[0]), pde_end_rms=float(time_rms[-1]),
                    pde_validation_points=len(s_validation))
    if not all(math.isfinite(value) for value in info.values()):
        raise NewtonFailure('nonfinite_continuous_pde_validation', info)
    # 验证点包含训练点之间的中点，并包含两端；按最差时间切片验收。
    if info['pde_max_time_rms'] > config.pde_tol:
        raise NewtonFailure('continuous_pde_tolerance_not_met', info)
    return info


def invariant_values(grid, state):
    """按质量 M、能量 E 顺序返回两个离散守恒量。"""
    return torch.stack([grid.mass(state), grid.energy(state)], dim=-1)


def invariant_gradients(grid, state):
    """M、E 对四个状态分量的梯度，使用周期离散分部积分。"""
    a, b, v, w = state.unbind(dim=-3)
    grad_mass = grid.cell_area*torch.stack([2*a, 2*b, torch.zeros_like(v),
                                           torch.zeros_like(w)], dim=-3)
    grad_energy = grid.cell_area*torch.stack([
        -grid.kappa1*grid.laplacian(a)-2*grid.xi*a*v,
        -grid.kappa1*grid.laplacian(b)-2*grid.xi*b*v,
        -grid.kappa2*grid.laplacian(v)+grid.mu**2*v-grid.xi*(a.square()+b.square()),
        w,
    ], dim=-3)
    return grad_mass, grad_energy


@torch.no_grad()
def solve_mrrk_parameters(step, d1, d2, targets, config):
    """联立求解质量、能量约束，未知量为 (gamma1,gamma2)。

    y_next = y_RK + gamma1*D1 + gamma2*D2，其中 Dk=h*dk。
    初值为 (0,0)；时间因子为 1+gamma1+gamma2。
    阻尼和参数界是本实现的求根保护，不是论文的存在性保证。
    """
    grid, y_n = step.grid, step.y_n
    D1, D2 = step.h * d1, step.h * d2
    tolerance = y_n.new_tensor([config.mass_tol, config.energy_tol])
    # 使用等价坐标 (s,gamma2)，s=gamma1+gamma2，减少相消。
    difference2 = D2 - D1
    directions = (D1, difference2)
    parameters = y_n.new_zeros(2)
    trace = []
    diagnostic = dict(trace=trace, parameter_bound=config.gamma_bound, h=step.h,
                      direction_difference_norm=float(torch.linalg.vector_norm(difference2)),
                      rk_increment_norm=float(torch.linalg.vector_norm(D1)))

    def fail(reason):
        diagnostic.update(status='failed', reason=reason)
        raise NewtonFailure(reason, diagnostic)

    # 对质量和耦合三次能量作精确差值展开，减少消减误差。
    offsets = invariant_values(grid, y_n)-targets

    def equation(parameters):
        total_gamma, gamma2 = parameters
        change = (1. + total_gamma) * D1 + gamma2 * difference2
        F = offsets+grid.invariant_increment(y_n, change)
        gradients = invariant_gradients(grid, y_n + change)
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
            # 质量、能量两个原始守恒函数均需复核通过。
            y_next = y_n + (1. + total_gamma) * D1 + gamma2 * difference2
            actual = invariant_values(grid, y_next) - targets
            if not bool(torch.isfinite(actual).all()) or not bool((actual.abs() <= 2*tolerance).all()):
                fail('invariants_recheck_failed')
            diagnostic.update(status='converged', iterations=iteration,
                              gamma1=gamma1, gamma2=gamma2,
                              time_factor=1.+total_gamma,
                              mass_error=float(actual[0]), energy_error=float(actual[1]))
            return gamma1, gamma2, diagnostic
        if iteration == config.newton_iters:
            fail('max_newton_iterations')

        # 质量、能量按各自容差缩放，回溯同时考虑两个误差。
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
        if rank == 2:
            try:
                scaled_delta = torch.linalg.solve(normalized_J, scaled_F)
            except RuntimeError:
                fail('jacobian_solve_failed')
        else:
            # 方向退化时只接受兼容的 SVD Newton 步，最终复核质量和能量。
            projected = U.T @ scaled_F
            inverse = torch.zeros_like(singular_values)
            inverse[active] = 1. / singular_values[active]
            scaled_delta = Vh.T @ (inverse * projected)
            unresolved = normalized_J @ scaled_delta - scaled_F
            if float(unresolved.abs().max()) > 1.:
                fail('incompatible_rank_deficient_mrrk_system')
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
def multiple_relaxation(step, targets, config):
    """训练后：预测 K -> 两个 RK 方向 -> 联立求两个 gamma -> 状态/时间更新。"""
    K = step.predict_slopes()
    d1, d2 = step.rk_directions(K)
    gamma1, gamma2, diagnostic = solve_mrrk_parameters(step, d1, d2, targets, config)
    # 与求根器采用相同的差方向写法，减少大系数相消误差。
    time_factor = diagnostic['time_factor']
    D1, D2 = step.h * d1, step.h * d2
    y_next = step.y_n + time_factor * D1 + gamma2 * (D2-D1)
    physical_time_step = time_factor * step.h
    actual = invariant_values(step.grid, y_next) - targets
    tolerance = y_next.new_tensor([config.mass_tol, config.energy_tol])
    if not bool(torch.isfinite(actual).all()) or not bool((actual.abs() <= 2*tolerance).all()):
        raise NewtonFailure('最终状态的质量/能量复核未通过', diagnostic)
    return y_next, physical_time_step, gamma1, gamma2, K, diagnostic


def fit_step(step, targets, config, step_number, t, output_directory):
    """加训仍无可接受根时，减半 h 后重新训练；失败尝试不推进状态或时间。"""
    total_refinements = 0
    for halving in range(config.max_step_halvings + 1):
        for refinement in range(config.max_refinements + 1):
            train_rk(step, config.adam, config.lbfgs, config.lr)
            with torch.no_grad():
                rk_rms = math.sqrt(float(step.rk_loss()))
            if not math.isfinite(rk_rms):
                raise RuntimeError('RK 残差不是有限数。')
            try:
                phase = 'rk'
                if rk_rms > config.rk_tol:
                    raise NewtonFailure('rk_residual_exceeds_tolerance', dict(rk_rms=rk_rms))
                phase = 'relaxation'
                result = multiple_relaxation(step, targets, config)
                y_next, duration, gamma1, gamma2, K, diagnostic = result
                # 第二网络的输入端点必须是松弛后的结果，时间尺度必须用 duration。
                phase = 'continuous_pde'
                curve = ContinuousFNO(step.grid, step.y_n, y_next, duration,
                                      config.width, config.modes, config.pde_terms).to(step.y_n.device)
                pde_info = train_continuous(curve, config)
                return result, rk_rms, total_refinements, halving, curve, pde_info
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
    parser.add_argument('--n', type=int, default=64, help='每个空间方向的网格点数')
    parser.add_argument('--length', type=float, default=40., help='正方形边长，默认 [-1,1)^2')
    parser.add_argument('--kappa1', type=float, default=-.4)
    parser.add_argument('--kappa2', type=float, default=.1)
    parser.add_argument('--xi', type=float, default=.2)
    parser.add_argument('--mu', type=float, default=.1)
    parser.add_argument('--t-end', type=float, default=5., help='最少推进到的物理时间')
    parser.add_argument('--mass-tol', type=float, default=1e-8)
    parser.add_argument('--dt', type=float, default=0.01, help='名义 RK 步长 h')
    parser.add_argument('--steps', type=int, default=500, help='最少接受步数；不足 t-end 时继续推进')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--modes', type=int, default=16, help='二维 FNO 每个方向保留的低频模态数')
    parser.add_argument('--adam', type=int, default=400)
    parser.add_argument('--lbfgs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--rk-tol', type=float, default=1e-1, help='阶段斜率残差 RMS')
    parser.add_argument('--energy-tol', type=float, default=1e-8)
    parser.add_argument('--newton-iters', type=int, default=40)
    parser.add_argument('--gamma-bound', type=float, default=.5, help='两个参数分别的近零搜索半径；失败则加训或减小 h')
    parser.add_argument('--pde-adam', type=int, default=200, help='第二 FNO 的 Adam 次数')
    parser.add_argument('--pde-lbfgs', type=int, default=100, help='第二 FNO 的 L-BFGS 次数')
    parser.add_argument('--pde-lr', type=float, default=.003)
    parser.add_argument('--pde-weight', type=float, default=1., help='第二网络 PDE 损失权重')
    parser.add_argument('--pde-terms', type=int, default=4, help='连续解修正的时间多项式项数')
    parser.add_argument('--pde-points', type=int, default=9, help='步内 PDE 训练时间点数，含两端')
    parser.add_argument('--pde-tol', type=float, default=1e-1, help='验证点中最差时间切片的 PDE RMS 容差')
    parser.add_argument('--max-refinements', type=int, default=3)
    parser.add_argument('--max-step-halvings', type=int, default=6, help='当前步失败后的最大减半次数')
    parser.add_argument('--min-dt', type=float, default=None, help='最小名义步长，默认 dt/64')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', default='skg2d_fno_mrrk_gauss3_pde_output')
    config = parser.parse_args(argv)

    if min(config.dt, config.length, config.mass_tol, config.steps, config.width, config.modes,
           config.rk_tol, config.energy_tol, config.newton_iters,
           config.gamma_bound, config.lr, config.threads, config.pde_lr,
           config.pde_weight, config.pde_terms, config.pde_tol) <= 0:
        parser.error('步长、区间长度、步数、宽度、模态数、容差、学习率和线程数必须为正。')
    if min(config.adam, config.lbfgs, config.pde_adam, config.pde_lbfgs,
           config.max_refinements, config.max_step_halvings) < 0:
        parser.error('优化次数和加训次数不能为负。')
    if config.t_end < 0:
        parser.error('--t-end 不能为负。')
    if not all(math.isfinite(value) for value in (config.kappa1, config.kappa2, config.xi, config.mu)):
        parser.error('方程系数必须为有限数。')
    if config.pde_points < 3:
        parser.error('--pde-points 至少为 3，覆盖起点、内部和终点。')
    if config.min_dt is None:
        config.min_dt = config.dt / 64.
    if not 0. < config.min_dt <= config.dt:
        parser.error('--min-dt 必须为正且不大于 --dt。')

    device = configure(config.threads, config.seed, config.device)
    grid = PeriodicGrid(config.n, length=config.length, device=device,
                        kappa1=config.kappa1, kappa2=config.kappa2, xi=config.xi, mu=config.mu)

    y = initial_state(grid).detach()
    mass_target = grid.mass(y).detach()
    energy_target = grid.energy(y).detach()
    targets = torch.stack([mass_target, energy_target])
    # 相对能量误差以初值的绝对值为分母。
    initial_energy = float(energy_target)
    energy_scale = max(abs(initial_energy), torch.finfo(y.dtype).eps)
    output_directory = Path(config.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    step_directory = output_directory / 'steps'
    step_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(step_directory / 'step_00000000.npz',
                        x1=grid.x.cpu().numpy(), x2=grid.x.cpu().numpy(), t=0., state=y.cpu().numpy(),
                        energy=float(energy_target), mass=float(mass_target),
                        complex_state=(y[0]+1j*y[1]).cpu().numpy(),
                        v=y[2].cpu().numpy(), v_t=y[3].cpu().numpy())

    t = 0.
    times = [t]
    states = [y.cpu().numpy()]
    snapshots = {0.: states[0].copy()}
    history = []
    previous_network = None
    next_h = config.dt
    step_number = 0
    # 将实际使用的表保存到输出目录，避免不同方法的结果混淆。
    a, b1, b2, c_nodes = gauss_legendre_tableau(y)
    (output_directory / 'butcher_tableau.json').write_text(
        json.dumps(dict(method='Gauss-Legendre collocation, 3 stages',
                        main_order=6, embedded_order=1,
                        a=a.cpu().tolist(), b1=b1.cpu().tolist(), b2=b2.cpu().tolist(),
                        c=c_nodes.cpu().tolist()), indent=2), encoding='utf-8',
    )

    # 自适应减小 h 后不能再用初始 dt 推算最大步数，直接按实际 t 判断。
    while step_number < config.steps or t < config.t_end:
        step_number += 1
        step = RKStep(grid, y, h=next_h, width=config.width, modes=config.modes).to(device)
        if previous_network is not None:
            step.network.load_state_dict(previous_network)

        result, rk_rms, refinements, halvings, curve, pde_info = fit_step(
            step, targets, config, step_number, t, output_directory,
        )
        y_next, time_step, gamma1, gamma2, K, diagnostic = result
        next_h = step.h

        mass_error = float(grid.mass(y_next) - mass_target)
        if not math.isfinite(mass_error) or abs(mass_error) > 2*config.mass_tol:
            raise NewtonFailure('mass_recheck_failed', dict(mass_error=mass_error))
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
            grad_M, grad_E = invariant_gradients(grid, y)
            rhs_value = grid.rhs(y)
            mass_rhs_rate = float((grad_M * rhs_value).sum())
            energy_rhs_rate = float((grad_E * rhs_value).sum())
            correction = step.h * ((diagnostic['time_factor']-1.)*d1
                                   + gamma2*(d2-d1))
            correction_norm = float(torch.linalg.vector_norm(correction))
            increment_norm = float(torch.linalg.vector_norm(step.h*d1))
        row = {
            'step': step_number,
            't': t,
            'h': step.h,
            'physical_time_step': time_step,
            'gamma1': gamma1,
            'gamma2': gamma2,
            'max_abs_gamma': max(abs(gamma1), abs(gamma2)),
            'time_factor': diagnostic['time_factor'],
            'rk_rms': rk_rms,
            'mass': float(grid.mass(y)),
            'mass_error': mass_error,
            'energy': initial_energy + energy_error,
            'energy_error': energy_error,
            'relative_energy_error': abs(energy_error) / energy_scale,
            'mass_rhs_rate': mass_rhs_rate,
            'energy_rhs_rate': energy_rhs_rate,
            'correction_norm': correction_norm,
            'relative_correction': correction_norm / max(increment_norm, torch.finfo(y.dtype).eps),
            'newton_iterations': diagnostic['iterations'],
            'refinements': refinements,
            'step_halvings': halvings,
            **pde_info,
        }
        history.append(row)
        times.append(t)
        states.append(y.cpu().numpy())
        print(
            f'step={step_number:02d}, t={t:.6f}, h={step.h:.4g}, '
            f'gamma1={gamma1:+.4e}, gamma2={gamma2:+.4e}, '
            f'RK={rk_rms:.2e}, dE={row["energy_error"]:+.4e}, '
            f'dM={mass_error:+.4e}, '
            f'PDE={pde_info["pde_max_time_rms"]:.2e}',
            flush=True,
        )

        with torch.no_grad():
            dense_s = torch.linspace(0., 1., 2*config.pde_points-1, device=y.device, dtype=y.dtype)
            dense_u, _ = curve.fields(dense_s)
            dense_states = dense_u.cpu().numpy()
            # 指定物理时刻从当前区间的连续 FNO 解取值，不再线性插值端点。
            for requested_time in SNAPSHOT_TIMES:
                if requested_time not in snapshots and t_start <= requested_time <= t:
                    s = y.new_tensor([(requested_time-t_start)/time_step])
                    snapshot_u, _ = curve.fields(s)
                    snapshots[requested_time] = snapshot_u[0].cpu().numpy()
        torch.save(dict(state_dict=curve.state_dict(), duration=time_step,
                        terms=config.pde_terms, width=config.width, modes=config.modes,
                        n=grid.n, length=grid.length,
                        coefficients=dict(kappa1=grid.kappa1, kappa2=grid.kappa2, xi=grid.xi, mu=grid.mu),
                        t_start=t_start, t_end=t, pde_info=pde_info),
                   step_directory / f'continuous_{step_number:08d}.pt')
        np.savez_compressed(
            step_directory / f'step_{step_number:08d}.npz',
            x1=grid.x.cpu().numpy(), x2=grid.x.cpu().numpy(), t=t, t_start=t-time_step, h=step.h,
            state_start=step.y_n.cpu().numpy(), state=y.cpu().numpy(),
            rk_state=(step.y_n+step.h*d1).cpu().numpy(),
            K=torch.stack(K).cpu().numpy(),
            Y=torch.stack(step.stage_states(K)).cpu().numpy(),
            d1=d1.cpu().numpy(), d2=d2.cpu().numpy(),
            b1=step.b1.cpu().numpy(), b2=step.b2.cpu().numpy(),
            a=step.a.cpu().numpy(), c=step.c.cpu().numpy(),
            gamma1=gamma1, gamma2=gamma2, time_factor=diagnostic['time_factor'],
            energy=float(grid.energy(y)), complex_state=(y[0]+1j*y[1]).cpu().numpy(),
            v=y[2].cpu().numpy(), v_t=y[3].cpu().numpy(),
            mass=float(grid.mass(y)), mass_error=mass_error,
            energy_error=energy_error,
            dense_s=dense_s.cpu().numpy(), dense_t=t_start+time_step*dense_s.cpu().numpy(),
            dense_states=dense_states,
            pde_rms=pde_info['pde_rms'], pde_max_time_rms=pde_info['pde_max_time_rms'],
        )
        (step_directory / f'newton_{step_number:08d}.json').write_text(
            json.dumps(diagnostic, indent=2), encoding='utf-8',
        )
        np.savez_compressed(
            output_directory / 'solution.npz',
            x1=grid.x.cpu().numpy(), x2=grid.x.cpu().numpy(), t=times, states=states,
            masses=[float(mass_target)] + [item['mass'] for item in history],
            energies=[initial_energy] + [item['energy'] for item in history],
            complex_states=np.asarray(states)[:, 0] + 1j*np.asarray(states)[:, 1],
            v=np.asarray(states)[:, 2], v_t=np.asarray(states)[:, 3],
            state_components=np.asarray(['real_u', 'imag_u', 'v', 'v_t']),
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
        'max_mass_error': max(abs(row['mass_error']) for row in history),
        'invariant_definitions': dict(mass='integral(|u|^2)',
            energy='integral((kappa1*|grad(u)|^2+kappa2*|grad(v)|^2+v_t^2+mu^2*v^2)/2-xi*|u|^2*v)'),
        'equations': ['i*u_t + kappa1/2*Delta(u) + xi*u*v = 0',
                      'v_tt - kappa2*Delta(v) + mu^2*v - xi*|u|^2 = 0'],
        'state_components': ['real_u', 'imag_u', 'v', 'v_t'],
        'spatial_axis_order': ['x1', 'x2'],
        'boundary': 'periodic; supplied Gaussian initial values sampled without alteration',
        'snapshot_times': list(SNAPSHOT_TIMES),
        'method': '2D FNO stages + Gauss MRRK + independent PDE-trained continuous 2D FNO',
        'main_rk_order': 6,
        'embedded_rk_order': 1,
        'relaxation_parameters': 2,
        'invariant_order': ['mass', 'energy'],
        'max_abs_gamma': max(row['max_abs_gamma'] for row in history),
        'paper_doi': '10.1007/s10915-023-02312-4',
        'max_mass_rhs_rate': max(abs(row['mass_rhs_rate']) for row in history),
        'max_energy_rhs_rate': max(abs(row['energy_rhs_rate']) for row in history),
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
    save_plots(output_directory, grid.x.cpu().numpy(), times, states, history, snapshots)
    return summary


if __name__ == '__main__':
    main()
