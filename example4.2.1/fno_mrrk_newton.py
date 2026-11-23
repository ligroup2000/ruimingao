# -*- coding: utf-8 -*-
"""周期非线性薛定谔方程：i*u_t + u_xx + 2*|u|^2*u = 0。
初值条件为：u(x,t)=sech(x+4t) * exp(-i(2x+3t)) (算例4.2.1)
动量：1/2*sum(|u|^2)
能量：1/2*sum(|du|^2-(|u|^2)^2)

区间为[-20,20], N=128, 将u=a+ib处理为 a,b分开计算
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


def exact_solution(x, t=0.):
    """给定真解的实部、虚部，形状为 (2,N)。"""
    amplitude = 1. / torch.cosh(x+4*t)
    phase = 2*x+3*t
    return torch.stack([amplitude*torch.cos(phase), -amplitude*torch.sin(phase)], dim=-2)


class PeriodicGrid:
    def __init__(self, n, length, device):
        if n < 32 or n % 2:
            raise ValueError('n 需为不小于32的偶数')
        self.n, self.length = n, length
        self.dx = length/n
        self.x = -length/2+self.dx*torch.arange(n, device=device)
        self.k = 2*math.pi*torch.fft.rfftfreq(n, d=self.dx, device=device)
        # 奇数阶导数的 Nyquist 模态置零，使 D_x 为反对称算子。
        self.k[-1] = 0.

    def dx1(self, f):
        return torch.fft.irfft(1j*self.k*torch.fft.rfft(f, dim=-1), n=self.n, dim=-1)

    def dx2(self, f):
        return torch.fft.irfft(-self.k.square()*torch.fft.rfft(f, dim=-1), n=self.n, dim=-1)

    # f项
    def rhs(self, state):
        # i*u_t+u_xx+2*|u|^2*u=0，即 u_t=i*(u_xx+2*|u|^2*u)。
        a, b = state.unbind(dim=-2)
        rho = a.square()+b.square()
        return torch.stack([-self.dx2(b)-2*rho*b, self.dx2(a)+2*rho*a], dim=-2)

    def mass(self, state):
        return .5*self.dx * state.square().sum(dim=(-2, -1))

    def energy(self, state):
        rho = state.square().sum(dim=-2)
        gradient_squared = self.dx1(state).square().sum(dim=-2)
        return .5*self.dx * (gradient_squared-rho.square()).sum(dim=-1)

    # 检测用函数，在代码中默认不使用
    def interpolate(self, values, points, derivative=0):
        """周期三角插值，支持边界值与导数检查。"""
        coeff = torch.fft.rfft(values, dim=-1)/self.n
        k = 2*math.pi*torch.fft.rfftfreq(self.n, d=self.dx, device=values.device)
        factor = torch.ones_like(k)*2
        factor[0] = factor[-1] = 1
        phase = torch.exp(1j*k[:, None]*(points[None, :]+self.length/2))
        if derivative:
            phase = phase*(1j*k[:, None])**derivative
        return torch.einsum('...k,kq,k->...q', coeff, phase, factor.to(coeff.dtype)).real


class NewtonFailure(RuntimeError):
    def __init__(self, message, diagnostic=None):
        super().__init__(message)
        self.diagnostic = dict(diagnostic or {})
        self.diagnostic.setdefault('reason',message)

# 神经网络的全局部分
class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, 2)/(width*math.sqrt(2)))

    def forward(self, z):
        zhat = torch.fft.rfft(z, dim=-1)
        count = min(self.modes, zhat.shape[-1])
        result = torch.zeros_like(zhat)
        weights = torch.view_as_complex(self.weight.contiguous())
        result[..., :count] = torch.einsum('bik,iok->bok', zhat[..., :count], weights[..., :count])
        return torch.fft.irfft(result, n=z.shape[-1], dim=-1)

class FourierCNN(nn.Module):
    def __init__(self, channels, width, modes):
        super().__init__()
        self.lift = nn.Conv1d(channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(2)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(2)])
        self.project = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.project.weight); nn.init.zeros_(self.project.bias)

    def forward(self, features):
        z = self.lift(features)
        for spectral, local in zip(self.spectral, self.local):
            z = torch.tanh(spectral(z)+local(z))
        # 无边界因子、无端点投影；保留有序周期网格及逆 FFT 的周期解释。
        return self.project(z)


SNAPSHOT_TIMES = (0., 1., 3., 5., 7., 9.)


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

def save_plots(output_directory, x, times, states, history, snapshots):
    """守恒图与演化图；指定时刻使用修正后的端点作 Hermite 插值。"""
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
    amplitude = np.sqrt(np.sum(states**2, axis=1))
    mesh = ax.pcolormesh(x, times, amplitude, shading='auto', cmap='viridis')
    fig.colorbar(mesh, ax=ax, label='|u(x, t)|')
    ax.set(xlabel='x', ylabel='Physical time t', title='Evolution of |u|')
    fig.tight_layout()
    fig.savefig(output_directory / 'long_time_evolution.png', dpi=200)
    plt.close(fig)

    snapshot_states = []
    references = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, requested_time in zip(axes.flat, SNAPSHOT_TIMES):
        if not times[0] <= requested_time <= times[-1]:
            ax.set_visible(False)
            continue
        snapshot = snapshots[requested_time]
        # 参考解只用于绘图，不参与训练或能量修正。
        with torch.no_grad():
            reference = exact_solution(torch.as_tensor(x), requested_time).cpu().numpy()
        snapshot_states.append(snapshot)
        references.append(reference)
        ax.plot(x, np.sqrt(np.sum(snapshot**2, axis=0)), label='FNO + MRRK', linewidth=1.5)
        ax.plot(x, np.sqrt(np.sum(reference**2, axis=0)), '--', label='Soliton reference', linewidth=1.2)
        ax.set(title=f't = {requested_time:g}', xlabel='x', ylabel='|u(x, t)|')
        ax.grid(True, alpha=.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Time snapshots from the cubic Hermite')
    fig.tight_layout()
    fig.savefig(output_directory / 'time_snapshots.png', dpi=200)
    plt.close(fig)
    np.savez_compressed(
        output_directory / 'time_snapshots.npz', x=x,
        t=np.asarray([time for time in SNAPSHOT_TIMES if times[0] <= time <= times[-1]]), states=np.asarray(snapshot_states),
        reference=np.asarray(references), interpolation='cubic Hermite',
    )

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

    y_n.shape = (2, N)：当前实部、虚部状态。
    K[i].shape = (2, N)：网络预测的第 i 个阶段斜率。
    Y[i].shape = (2, N)：由所有 K_j 和 RK 系数重建的第 i 个阶段状态。
    i = 0, 1, 2 对应数学下标 i = 1, 2, 3。
    """

    def __init__(self, grid, y_n, h, width=16, modes=32, method='lobatto'):
        super().__init__()
        self.grid = grid
        self.h = h
        self.register_buffer('y_n', y_n.detach().clone())

        # 只在这里选择时间离散的 Butcher 表；FNO 和空间导数不变。
        self.method = method
        tableau = {'gauss': gauss_legendre_tableau, 'lobatto': lobatto_iiia_tableau}[method]
        a, b1, b2, c = tableau(y_n)
        self.stage_count = len(c)
        self.register_buffer('a', a)
        self.register_buffer('b1', b1)
        self.register_buffer('b2', b2)
        self.register_buffer('c', c)

        # 输入：Re(u_n)、Im(u_n)、阶段 c_i；输出 K_i 的实部和虚部。
        # 模态数不超过当前网格可表示的实数 Fourier 频率数。
        self.network = FourierCNN(channels=3, width=width, modes=min(modes, grid.n // 2 + 1))

    def predict_slopes(self):
        """K_i = FNO_theta(Re(u_n), Im(u_n), c_i)，逐阶段计算。

        FNO 要求输入形状为 (batch, channels, N)。这里只有一个样本，
        所以用 unsqueeze(0) 加上 batch 维度，输出后再 squeeze(0) 去掉。
        """
        K = []
        a_n, b_n = self.y_n

        for i in range(self.stage_count):
            time_channel = self.c[i] * torch.ones_like(a_n)
            input_channels = torch.stack([a_n, b_n, time_channel], dim=0)
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
        """L_RK = (1/stage_count) * sum_i mean(|R_i|^2)。"""
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
    """用优化器求解隐式阶段方程，不使用参考解作标签。"""
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
# hybrid Picard solver
def polish_rk_stages(step, K, max_iterations, tolerance):
    """以 FNO 阶段为初值精修当前 RK 方程：频域线性隐式 + 非线性 Picard。

    精修后的斜率不再是纯网络输出；零次迭代禁用，不作端点投影。
    """
    grid, h, A = step.grid, step.h, step.a

    def residual_rms(slopes):
        residuals = torch.stack(step.stage_residuals(step.stage_states(slopes), slopes))
        return float(residuals.square().mean().sqrt())

    raw_rms = residual_rms(K)
    info = dict(fno_rk_rms=raw_rms, stage_polish_iterations=0, polished_rk_rms=raw_rms)
    if not math.isfinite(raw_rms):
        raise NewtonFailure('nonfinite_stage_polish_residual', info)
    if not max_iterations or raw_rms <= tolerance:
        return K, raw_rms, info

    y = step.y_n
    # 分别变换 a=Re(u)、b=Im(u)，FFT 系数也拆成实数存储。
    initial_hat = torch.view_as_real(torch.fft.rfft(y, dim=-1))
    identity = torch.eye(step.stage_count, dtype=y.dtype, device=y.device)
    identity = identity.expand(grid.k.numel(), -1, -1)
    B = h*grid.k.square()[:, None, None]*A
    # a_t=-b_xx-2*rho*b，b_t=a_xx+2*rho*a。
    # 每个频率用实数块矩阵 [I,-B; B,I] 联立求 a、b，
    # Fourier 系数的实部、虚部作为两个独立右端同时求解。
    matrix = torch.cat([torch.cat([identity, -B], dim=-1),
                        torch.cat([B, identity], dim=-1)], dim=-2)
    stages = torch.stack(step.stage_states(K))
    try:
        factors = torch.linalg.lu_factor(matrix)
        for iteration in range(1, max_iterations + 1):
            a, b = stages[:, 0], stages[:, 1]
            rho = a.square() + b.square()
            nonlinear = torch.stack([-2*rho*b, 2*rho*a], dim=1)
            nonlinear_hat = torch.view_as_real(torch.fft.rfft(nonlinear, dim=-1))
            rhs = initial_hat[None] + h*torch.einsum('ij,jcnr->icnr', A, nonlinear_hat)
            # (stage, component, frequency, real/imag) -> (frequency, 2*stage, 2)
            packed = rhs.permute(2, 1, 0, 3).reshape(grid.k.numel(), 2*step.stage_count, 2)
            solved = torch.linalg.lu_solve(*factors, packed)
            stage_hat = solved.reshape(grid.k.numel(), 2, step.stage_count, 2)
            stage_hat = stage_hat.permute(2, 1, 0, 3).contiguous()
            stages = torch.fft.irfft(torch.view_as_complex(stage_hat), n=grid.n, dim=-1)
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
        s = s.reshape(-1, 1, 1)
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
        time_mse = residual.square().mean(dim=(-2, -1))
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
    # 与 Newton 求根器使用同一精确多项式差值，减少消减误差。
    rho_n = y_n.square().sum(dim=-2, keepdim=True)
    drho = (2*y_n*change+change.square()).sum(dim=-2, keepdim=True)
    ux_n, dux = grid.dx1(y_n), grid.dx1(change)
    increments = .5*grid.dx*torch.stack([
        drho.sum(),
        (2*ux_n*dux+dux.square()).sum()-(2*rho_n*drho+drho.square()).sum()])
    F = invariant_values(grid, y_n)-targets+increments

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
        s, _, nx = self.polished_stages.shape
        context = torch.cat([self.polished_stages.flatten(0, 1), self.polished_endpoint], dim=0)
        context = context.unsqueeze(0).expand(s, -1, -1)
        labels = self.nodes.reshape(s, 1, 1).expand(-1, 1, nx)
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
                           config.width, min(config.modes, step.grid.n//2+1))
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
    """按质量 M、能量 E 顺序返回两个离散守恒量。"""
    return torch.stack([grid.mass(state), grid.energy(state)], dim=-1)


def invariant_gradients(grid, state):
    """M、E 对实部和虚部网格值的梯度，使用周期离散分部积分。"""
    rho = state.square().sum(dim=-2, keepdim=True)
    grad_mass = grid.dx * state
    grad_energy = grid.dx * (-grid.dx2(state)-2*rho*state)
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

    # 对质量和四次能量作差值展开，减少两个总量相减的消减误差。
    ux_n = grid.dx1(y_n)
    rho_n = y_n.square().sum(dim=-2, keepdim=True)
    M_offset = grid.mass(y_n) - targets[0]
    E_offset = grid.energy(y_n) - targets[1]

    def equation(parameters):
        total_gamma, gamma2 = parameters
        change = (1. + total_gamma) * D1 + gamma2 * difference2
        dux = grid.dx1(change)
        rho_change = (2*y_n*change+change.square()).sum(dim=-2, keepdim=True)
        F_M = M_offset + .5*grid.dx*rho_change.sum()
        F_E = E_offset + .5*grid.dx * (
            (2*ux_n*dux+dux.square()).sum()
            - (2*rho_n*rho_change+rho_change.square()).sum()
        )
        F = torch.stack([F_M, F_E])
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
def multiple_relaxation(step, targets, config, K=None):
    """训练后：预测 K -> 两个 RK 方向 -> 联立求两个 gamma -> 状态/时间更新。"""
    if K is None:
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
                phase = 'stage_polish'
                with torch.no_grad():
                    K, rk_rms, stage_info = polish_rk_stages(
                        step, step.predict_slopes(), config.stage_polish_iters,
                        config.stage_polish_tol,
                    )
                phase = 'rk'
                if rk_rms > config.rk_tol:
                    raise NewtonFailure('rk_residual_exceeds_tolerance', dict(rk_rms=rk_rms))
                phase = 'relaxation'
                result = multiple_relaxation(step, targets, config, K=K)
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
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--length', type=float, default=40., help='周期区间总长度，默认 [-20,20)')
    parser.add_argument('--t-end', type=float, default=10., help='最少推进到的物理时间')
    parser.add_argument('--mass-tol', type=float, default=1e-12)
    parser.add_argument('--dt', type=float, default=0.1, help='名义 RK 步长 h')
    parser.add_argument('--steps', type=int, default=100, help='最少接受步数；不足 t-end 时继续推进')
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--modes', type=int, default=64, help='FNO 保留的低频模态数，含零频')
    parser.add_argument('--adam', type=int, default=400)
    parser.add_argument('--lbfgs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--rk-tol', type=float, default=1e-1, help='阶段斜率残差 RMS')
    parser.add_argument('--stage-polish-iters', type=int, default=0,
                        help='FNO 后数值精修 RK 阶段的最大次数；0 禁用')
    parser.add_argument('--stage-polish-tol', type=float, default=1e-10,
                        help='数值精修后的阶段残差 RMS 容差')
    parser.add_argument('--energy-tol', type=float, default=1e-12)
    parser.add_argument('--newton-iters', type=int, default=20)
    parser.add_argument('--gamma-bound', type=float, default=.5, help='两个参数分别的近零搜索半径；失败则加训或减小 h')
    parser.add_argument('--corrector-adam', type=int, default=10, help='第二 FNO 的 Adam 次数；0 关闭阶段修正')
    parser.add_argument('--corrector-lr', type=float, default=1e-4)
    parser.add_argument('--corrector-points', type=int, default=9, help='第二网络 Hermite 内部训练点数')
    parser.add_argument('--corrector-interior-weight', type=float, default=1.)
    parser.add_argument('--pde-points', type=int, default=9, help='Hermite PDE 验证点数，含两端')
    parser.add_argument('--pde-tol', type=float, default=1e-1, help='验证点中最差时间切片的 PDE RMS 容差')
    parser.add_argument('--max-refinements', type=int, default=3)
    parser.add_argument('--max-step-halvings', type=int, default=6, help='当前步失败后的最大减半次数')
    parser.add_argument('--min-dt', type=float, default=None, help='最小名义步长，默认 dt/64')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', default='snapshot')
    parser.add_argument('--method', choices=['gauss', 'lobatto', 'both'], default='both',
                        help='Gauss: 3 stages, order 6; Lobatto IIIA: 4 stages, order 6; both: run separately')
    config = parser.parse_args(argv)

    if min(config.dt, config.length, config.mass_tol, config.steps, config.width, config.modes,
           config.rk_tol, config.energy_tol, config.newton_iters,
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

    methods = ('gauss', 'lobatto') if config.method == 'both' else (config.method,)
    summaries = {}
    for method in methods:
        method_config = argparse.Namespace(**vars(config))
        method_config.method = method
        if config.method == 'both':
            base = Path(config.output)
            method_config.output = str(base.with_name(base.name+'_'+method))
        print(f'Running {method}: output={method_config.output}', flush=True)
        summaries[method] = run_method(method_config)
    return summaries if config.method == 'both' else summaries[config.method]


def run_method(config):
    method_label = {'gauss': 'Gauss-Legendre', 'lobatto': 'Lobatto IIIA'}[config.method]
    tableau = {'gauss': gauss_legendre_tableau, 'lobatto': lobatto_iiia_tableau}[config.method]
    device = configure(config.threads, config.seed, config.device)
    grid = PeriodicGrid(config.n, length=config.length, device=device)

    y = exact_solution(grid.x, t=0.).detach()
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
                        x=grid.x.cpu().numpy(), t=0., state=y.cpu().numpy(),
                        energy=float(energy_target), mass=float(mass_target),
                        complex_state=(y[0]+1j*y[1]).cpu().numpy())

    t = 0.
    times = [t]
    states = [y.cpu().numpy()]
    snapshots = {0.: states[0].copy()}
    history = []
    previous_network = None
    next_h = config.dt
    step_number = 0
    # 将实际使用的表保存到输出目录，避免不同方法的结果混淆。
    a, b1, b2, c_nodes = tableau(y)
    (output_directory / 'butcher_tableau.json').write_text(
        json.dumps(dict(method=f'{method_label} collocation, {len(c_nodes)} stages',
                        main_order=6, embedded_order=1,
                        a=a.cpu().tolist(), b1=b1.cpu().tolist(), b2=b2.cpu().tolist(),
                        c=c_nodes.cpu().tolist()), indent=2), encoding='utf-8',
    )

    # 自适应减小 h 后不能再用初始 dt 推算最大步数，直接按实际 t 判断。
    while step_number < config.steps or t < config.t_end:
        step_number += 1
        step = RKStep(grid, y, h=next_h, width=config.width, modes=config.modes, method=config.method).to(device)
        if previous_network is not None:
            step.network.load_state_dict(previous_network)

        result, rk_rms, refinements, halvings, curve, pde_info, stage_outputs = fit_step(
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
        reference = exact_solution(grid.x, t)
        error_u = y - reference
        relative_l2_u = float(
            torch.linalg.vector_norm(error_u)
            / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(y.dtype).eps)
        )
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
            'fno_rk_rms': diagnostic['fno_rk_rms'],
            'polished_rk_rms': diagnostic['polished_rk_rms'],
            'stage_polish_iterations': diagnostic['stage_polish_iterations'],
            'mass': float(grid.mass(y)),
            'mass_error': mass_error,
            'energy': initial_energy + energy_error,
            'energy_error': energy_error,
            'relative_energy_error': abs(energy_error) / energy_scale,
            'mass_rhs_rate': mass_rhs_rate,
            'energy_rhs_rate': energy_rhs_rate,
            'correction_norm': correction_norm,
            'relative_correction': correction_norm / max(increment_norm, torch.finfo(y.dtype).eps),
            'relative_l2_u': relative_l2_u,
            'newton_iterations': diagnostic['iterations'],
            'refinements': refinements,
            'step_halvings': halvings,
            **pde_info,
            **{name: value for name, value in diagnostic.items() if name.startswith('corrector_')},
        }
        history.append(row)
        times.append(t)
        states.append(y.cpu().numpy())
        print(
            f'step={step_number:02d}, t={t:.6f}, h={step.h:.4g}, '
            f'gamma1={gamma1:+.4e}, gamma2={gamma2:+.4e}, '
            f'RK={rk_rms:.2e}, dE={row["energy_error"]:+.4e}, '
            f'dM={mass_error:+.4e}, '
            f'PDE={pde_info["pde_max_time_rms"]:.2e}, '
            f'relL2={relative_l2_u:.2e}',
            flush=True,
        )

        with torch.no_grad():
            dense_s = torch.linspace(0., 1., 2*config.pde_points-1, device=y.device, dtype=y.dtype)
            dense_u, _ = curve.fields(dense_s)
            dense_states = dense_u.cpu().numpy()
            # 指定物理时刻从当前区间的三次 Hermite 曲线取值。
            for requested_time in SNAPSHOT_TIMES:
                if requested_time not in snapshots and t_start <= requested_time <= t:
                    s = y.new_tensor([(requested_time-t_start)/time_step])
                    snapshot_u, _ = curve.fields(s)
                    snapshots[requested_time] = snapshot_u[0].cpu().numpy()
        torch.save(dict(state_dict=curve.state_dict(), duration=time_step,
                        interpolation='cubic Hermite', width=config.width, modes=config.modes,
                        t_start=t_start, t_end=t, pde_info=pde_info),
                   step_directory / f'continuous_{step_number:08d}.pt')
        np.savez_compressed(
            step_directory / f'step_{step_number:08d}.npz',
            x=grid.x.cpu().numpy(), t=t, t_start=t-time_step, h=step.h,
            state_start=step.y_n.cpu().numpy(), state=y.cpu().numpy(),
            rk_state=(step.y_n+step.h*d1).cpu().numpy(),
            K=torch.stack(K).cpu().numpy(),
            **{name: value.cpu().numpy() for name, value in stage_outputs.items()},
            d1=d1.cpu().numpy(), d2=d2.cpu().numpy(),
            b1=step.b1.cpu().numpy(), b2=step.b2.cpu().numpy(),
            a=step.a.cpu().numpy(), c=step.c.cpu().numpy(),
            gamma1=gamma1, gamma2=gamma2, time_factor=diagnostic['time_factor'],
            energy=float(grid.energy(y)), complex_state=(y[0]+1j*y[1]).cpu().numpy(),
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
            x=grid.x.cpu().numpy(), t=times, states=states,
            masses=[float(mass_target)] + [item['mass'] for item in history],
            energies=[initial_energy] + [item['energy'] for item in history],
            complex_states=np.asarray(states)[:, 0, :] + 1j*np.asarray(states)[:, 1, :],
            reference=reference.cpu().numpy(),
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
        'invariant_definitions': dict(mass='1/2*integral(|u|^2)',
                                      energy='1/2*integral(|u_x|^2-|u|^4)'),
        'equation': 'i*u_t + u_xx + 2*|u|^2*u = 0',
        'state_components': ['real', 'imaginary'],
        'reference': 'sech(x+4*t)*exp(-i*(2*x+3*t)); exact on the real line',
        'method': f'FNO stages + {method_label} MRRK + FNO stage correction + cubic Hermite dense output',
        'tableau': config.method,
        'stage_count': len(c_nodes),
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
        'relative_l2_u': history[-1]['relative_l2_u'],
    }
    (output_directory / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8',
    )
    save_plots(output_directory, grid.x.cpu().numpy(), times, states, history, snapshots)
    return summary


if __name__ == '__main__':
    main()
