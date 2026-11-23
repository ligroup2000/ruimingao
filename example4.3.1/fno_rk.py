# -*- coding: utf-8 -*-
"""二维SKG FNO + 三阶段 Gauss-Legendre RK 对照实验。

与 fno_mrrk_newton.py 使用相同的方程、FFT 空间离散、初值和阶段网络。
仅用 RK 更新 y_next = y_n + h*sum(b_i*K_i)，物理时间直接增加 h。
默认 dt=0.1，恰好推进至 50 s（500 步），最后一步必要时截短。
每步固定执行 Adam/L-BFGS 次数，无残差或守恒容差筛选、无松弛、
无失败加训、无步长减半、无第二个连续网络。残差和守恒误差只记录。
L-BFGS 每次只做一次迭代，关闭正容差与线搜索，以外层次数控制训练。
出现 NaN/Inf 则报错并保存失败信息，不将非有限结果作为预测继续推进。
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .fno_mrrk_newton import (
        FourierCNN, PeriodicGrid, configure, exact_solution, gauss_legendre_tableau,
    )
except ImportError:
    from fno_mrrk_newton import (
        FourierCNN, PeriodicGrid, configure, exact_solution, gauss_legendre_tableau,
    )


SNAPSHOT_TIMES = (0., 10., 20., 30., 40., 50.)
REFERENCE_NOTE = (
    'Three nearby periodic images of sech(x+4*t)*exp(-i*(2*x+3*t)); '
    'approximate reference only, not an exact periodic NLS solution.'
)


def periodic_reference(x, t, length):
    """叠加邻近周期镜像，同时平移振幅和相位，保留穿越边界后的相位。"""
    central_image = round(-4*t/length)
    return sum(exact_solution(x + j*length, t)
               for j in range(central_image-1, central_image+2))


class RKStep(nn.Module):
    """只含主 RK 表；FNO 预测 K_i，损失为 mean(|K_i-f(Y_i)|^2)。"""

    def __init__(self, grid, y_n, h, width, modes):
        super().__init__()
        self.grid, self.h = grid, h
        self.register_buffer('y_n', y_n.detach().clone())
        a, b, _, c = gauss_legendre_tableau(y_n)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('c', c)
        self.network = FourierCNN(3, width, min(modes, grid.n//2+1))

    def predict_slopes(self):
        return torch.stack([
            self.network(torch.cat([
                self.y_n, node*torch.ones_like(self.y_n[:1]),
            ], dim=0)[None])[0]
            for node in self.c
        ])

    def stage_states(self, slopes):
        return self.y_n[None] + self.h*torch.einsum('ij,jcn->icn', self.a, slopes)

    def rk_loss(self):
        slopes = self.predict_slopes()
        return (slopes-self.grid.rhs(self.stage_states(slopes))).square().mean()

    def endpoint(self, slopes):
        return self.y_n + self.h*torch.einsum('i,icn->cn', self.b, slopes)

    def dense_output(self, slopes, s):
        """y(s)=y_n+h*sum_j integral_0^s L_j(z) dz*K_j，s∈[0,1]。"""
        s = s.reshape(-1)
        weights = []
        for j in range(3):
            others = [i for i in range(3) if i != j]
            p, q = self.c[others[0]], self.c[others[1]]
            denominator = (self.c[j]-p)*(self.c[j]-q)
            weights.append((s**3/3-(p+q)*s**2/2+p*q*s)/denominator)
        weights = torch.stack(weights, dim=1)
        return self.y_n[None] + self.h*torch.einsum('qi,icn->qcn', weights, slopes)


def train_rk(step, config):
    """固定训练预算；不依据 RK、质量或能量误差提前停止或重试。"""
    if config.adam:
        optimizer = torch.optim.Adam(step.network.parameters(), lr=config.lr)
        for _ in range(config.adam):
            optimizer.zero_grad(set_to_none=True)
            loss = step.rk_loss()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Adam RK loss is NaN/Inf')
            loss.backward()
            optimizer.step()

    if config.lbfgs:
        optimizer = torch.optim.LBFGS(
            step.network.parameters(), lr=config.lbfgs_lr,
            max_iter=1, tolerance_grad=0., tolerance_change=0.,
            line_search_fn=None,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = step.rk_loss()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('L-BFGS RK loss is NaN/Inf')
            loss.backward()
            return loss

        # 保留同一个优化器的历史，但每次仅执行一次迭代，不做容差退出。
        for _ in range(config.lbfgs):
            optimizer.step(closure)


def save_plots(directory, x, times, states, history, snapshots, length):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for relative, filename in [
        (False, 'mass_energy_conservation.png'),
        (True, 'mass_energy_relative_error.png'),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, name, symbol in zip(axes, ['mass', 'energy'], ['M', 'E']):
            key = f'relative_{name}_error' if relative else f'{name}_error'
            ax.plot(times, [0.] + [row[key] for row in history])
            ylabel = (f'|{symbol}(t)-{symbol}(0)| / |{symbol}(0)|' if relative
                      else f'{symbol}(t)-{symbol}(0)')
            ax.set(xlabel='Time (s)', ylabel=ylabel, title=f'RK {name} conservation')
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.grid(True, alpha=.3)
        fig.tight_layout()
        fig.savefig(directory / filename, dpi=200)
        plt.close(fig)

    amplitude = np.linalg.norm(states, axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(x, times, amplitude, shading='auto', cmap='viridis')
    fig.colorbar(mesh, ax=ax, label='|u(x,t)|')
    ax.set(xlabel='x', ylabel='Time (s)', title='FNO + RK prediction')
    fig.tight_layout()
    fig.savefig(directory / 'long_time_evolution.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, time in zip(axes.flat, SNAPSHOT_TIMES):
        if time not in snapshots:
            ax.set_visible(False)
            continue
        reference = periodic_reference(torch.as_tensor(x), time, length).numpy()
        ax.plot(x, np.linalg.norm(snapshots[time], axis=0), label='FNO + RK')
        ax.plot(x, np.linalg.norm(reference, axis=0), '--', label='Approx. periodic soliton')
        ax.set(title=f't = {time:g} s', xlabel='x', ylabel='|u|')
        ax.grid(True, alpha=.3)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'time_snapshots.png', dpi=200)
    plt.close(fig)

    reference = periodic_reference(torch.as_tensor(x), times[-1], length).numpy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, value, ref, name in zip(
        axes, [states[-1, 0], states[-1, 1], amplitude[-1]],
        [reference[0], reference[1], np.linalg.norm(reference, axis=0)],
        ['Re(u)', 'Im(u)', '|u|'],
    ):
        ax.plot(x, value, label='FNO + RK')
        ax.plot(x, ref, '--', label='Approx. periodic soliton')
        ax.set(xlabel='x', ylabel=name, title=f'{name}, t = {times[-1]:g} s')
        ax.grid(True, alpha=.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / 'final_prediction.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(times[1:], np.maximum([row['rk_rms'] for row in history], 1e-17))
    ax.set(xlabel='Time (s)', ylabel='Stage residual RMS', title='RK residual (diagnostic only)')
    ax.grid(True, alpha=.3)
    fig.tight_layout()
    fig.savefig(directory / 'rk_residual.png', dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=256)
    parser.add_argument('--length', type=float, default=80.)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--t-end', type=float, default=50.)
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--modes', type=int, default=64)
    parser.add_argument('--adam', type=int, default=400)
    parser.add_argument('--lbfgs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--lbfgs-lr', type=float, default=1.)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', default='nls_fno_rk_T50_output')
    config = parser.parse_args(argv)
    positive = [config.dt, config.t_end, config.length, config.lr, config.lbfgs_lr,
                config.width, config.modes, config.threads]
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error('时间、步长、区间长度、学习率、宽度、模态数和线程数必须为有限正数。')
    if config.n < 32 or config.n % 2:
        parser.error('--n 必须是不小于 32 的偶数。')
    if min(config.adam, config.lbfgs) < 0 or config.adam+config.lbfgs == 0:
        parser.error('训练次数不能为负，且至少使用一种优化器。')

    device = configure(config.threads, config.seed, config.device)
    grid = PeriodicGrid(config.n, config.length, device)
    y = exact_solution(grid.x, 0.).detach()
    mass0, energy0 = float(grid.mass(y)), float(grid.energy(y))
    eps = torch.finfo(y.dtype).eps
    mass_scale, energy_scale = max(abs(mass0), eps), max(abs(energy0), eps)
    directory = Path(config.output)
    directory.mkdir(parents=True, exist_ok=True)
    step_directory = directory / 'steps'
    step_directory.mkdir(parents=True, exist_ok=True)
    x = grid.x.cpu().numpy()
    times, states, history = [0.], [y.cpu().numpy()], []
    snapshots = {0.: states[0].copy()}
    previous_network = None
    a, b, _, c = gauss_legendre_tableau(y)
    (directory / 'butcher_tableau.json').write_text(json.dumps(dict(
        method='Gauss-Legendre, 3 stages, no relaxation', formal_order=6,
        a=a.cpu().tolist(), b=b.cpu().tolist(), c=c.cpu().tolist(),
    ), indent=2), encoding='utf-8')
    (directory / 'config.json').write_text(json.dumps(vars(config), indent=2), encoding='utf-8')
    (directory / 'history.jsonl').write_text('', encoding='utf-8')
    np.savez_compressed(step_directory / 'step_00000000.npz',
                        x=x, t=0., state=states[0], mass=mass0, energy=energy0)

    # 由整数步号计算时刻，避免累加 0.1 导致终点多出一个极小步。
    total_steps = math.ceil(config.t_end/config.dt)
    for number in range(1, total_steps+1):
        t_start = times[-1]
        t = min(number*config.dt, config.t_end)
        h = t-t_start
        step = RKStep(grid, y, h, config.width, config.modes).to(device)
        if previous_network is not None:
            step.network.load_state_dict(previous_network)
        try:
            train_rk(step, config)
            with torch.no_grad():
                slopes = step.predict_slopes()
                stage_states = step.stage_states(slopes)
                residual = slopes-grid.rhs(stage_states)
                rk_rms = float(residual.square().mean().sqrt())
                y_next = step.endpoint(slopes).detach()
                mass, energy = float(grid.mass(y_next)), float(grid.energy(y_next))
                if (not bool(torch.isfinite(y_next).all()) or
                        not all(math.isfinite(v) for v in [rk_rms, mass, energy])):
                    raise FloatingPointError('RK prediction or diagnostics are NaN/Inf')
        except (FloatingPointError, RuntimeError) as error:
            (directory / 'failure.json').write_text(json.dumps(dict(
                step=number, t_start=t_start, h=h, reason=str(error),
                last_completed_time=times[-1],
            ), indent=2), encoding='utf-8')
            raise

        with torch.no_grad():
            reference = periodic_reference(grid.x, t, config.length)
            relative_l2 = float(torch.linalg.vector_norm(y_next-reference) /
                                torch.linalg.vector_norm(reference).clamp_min(eps))
            for requested in SNAPSHOT_TIMES:
                if requested not in snapshots and t_start <= requested <= t:
                    s = y.new_tensor([(requested-t_start)/h])
                    snapshots[requested] = step.dense_output(slopes, s)[0].cpu().numpy()
        row = dict(
            step=number, t=t, h=h, physical_time_step=h, rk_rms=rk_rms,
            mass=mass, energy=energy, mass_error=mass-mass0, energy_error=energy-energy0,
            relative_mass_error=abs(mass-mass0)/mass_scale,
            relative_energy_error=abs(energy-energy0)/energy_scale,
            relative_l2_to_approx_periodic_reference=relative_l2,
        )
        history.append(row)
        y = y_next
        times.append(t)
        states.append(y.cpu().numpy())
        previous_network = {name: value.detach().clone()
                            for name, value in step.network.state_dict().items()}
        np.savez_compressed(
            step_directory / f'step_{number:08d}.npz', x=x, t=t, t_start=t_start, h=h,
            state_start=step.y_n.cpu().numpy(), state=states[-1],
            complex_state=(y[0]+1j*y[1]).cpu().numpy(),
            K=slopes.cpu().numpy(), Y=stage_states.cpu().numpy(),
            a=step.a.cpu().numpy(), b=step.b.cpu().numpy(), c=step.c.cpu().numpy(),
            mass=mass, energy=energy, mass_error=mass-mass0, energy_error=energy-energy0,
            rk_rms=rk_rms,
        )
        with (directory / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f'step={number:04d}/{total_steps}, t={t:.6f}, h={h:.4g}, '
              f'RK={rk_rms:.3e}, dM={mass-mass0:+.4e}, dE={energy-energy0:+.4e}, '
              f'relM={row["relative_mass_error"]:.3e}, '
              f'relE={row["relative_energy_error"]:.3e}', flush=True)

    states = np.asarray(states)
    np.savez_compressed(
        directory / 'solution.npz', x=x, t=np.asarray(times), states=states,
        complex_states=states[:, 0]+1j*states[:, 1],
        masses=np.asarray([mass0]+[row['mass'] for row in history]),
        energies=np.asarray([energy0]+[row['energy'] for row in history]),
        final_reference=reference.cpu().numpy(), reference_note=REFERENCE_NOTE,
    )
    snapshot_times = sorted(snapshots)
    np.savez_compressed(
        directory / 'time_snapshots.npz', x=x, t=np.asarray(snapshot_times),
        states=np.asarray([snapshots[time] for time in snapshot_times]),
        reference=np.asarray([periodic_reference(torch.as_tensor(x), time, config.length).numpy()
                              for time in snapshot_times]),
        interpolation='Integrated RK Lagrange basis; no second network',
        reference_note=REFERENCE_NOTE,
    )
    (directory / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    summary = dict(
        config=vars(config), final_time=times[-1], steps=len(history),
        method='FNO stages + Gauss-Legendre RK; no relaxation', formal_rk_order=6,
        initial_mass=mass0, initial_energy=energy0,
        final_mass=history[-1]['mass'], final_energy=history[-1]['energy'],
        max_mass_error=max(abs(row['mass_error']) for row in history),
        max_energy_error=max(abs(row['energy_error']) for row in history),
        max_relative_mass_error=max(row['relative_mass_error'] for row in history),
        max_relative_energy_error=max(row['relative_energy_error'] for row in history),
        max_rk_rms=max(row['rk_rms'] for row in history),
        relative_l2_to_approx_periodic_reference=history[-1]['relative_l2_to_approx_periodic_reference'],
        relative_mass_denominator=mass_scale, relative_energy_denominator=energy_scale,
        equation='i*u_t + u_xx + 2*|u|^2*u = 0',
        invariant_definitions=dict(mass='dx/2*sum(|u|^2)',
                                   energy='dx/2*sum(|u_x|^2-|u|^4)'),
        state_components=['real', 'imaginary'], reference_note=REFERENCE_NOTE,
    )
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    torch.save(dict(network=previous_network, t=times[-1], state=y.cpu(), config=vars(config)),
               directory / 'final_model.pt')
    save_plots(directory, x, times, states, history, snapshots, config.length)
    return summary


if __name__ == '__main__':
    main()
