# -*- coding: utf-8 -*-
"""2D 周期非线性薛定谔方程：i*u_t + u_xx + 2*|u|^2*u = 0。
区间为正方形区域, |x|+|y|<=\sqrt{2}L, L=10
N=128, 将u=a+ib处理为 a,b分开计算
RK + 无picard迭代优化 + 无松弛
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import fno_mrrk_newton as common
else:
    import fno_mrrk_newton as common


def dense_fields(step, slopes, fractions):
    """Integrated stage polynomial; endpoints coincide with the main RK update."""
    nodes = step.c.cpu().numpy()
    weights, derivatives = [], []
    points = fractions.cpu().numpy()
    for j, node in enumerate(nodes):
        others = np.delete(nodes, j)
        cardinal = np.polynomial.Polynomial.fromroots(others)
        cardinal = cardinal/cardinal(node)
        integral = cardinal.integ()
        weights.append(integral(points)-integral(0.))
        derivatives.append(cardinal(points))
    stages = torch.stack(slopes)
    w = step.y_n.new_tensor(np.asarray(weights).T)
    dw = step.y_n.new_tensor(np.asarray(derivatives).T)
    states = step.y_n[None]+step.h*torch.einsum('si,icxy->scxy', w, stages)
    rates = torch.einsum('si,icxy->scxy', dw, stages)
    return states, rates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=40)
    parser.add_argument('--L', type=float, default=10.)
    parser.add_argument('--length', type=float, default=None)
    parser.add_argument('--nu', type=float, default=.1)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=500, help='Minimum step count; 0 uses t-end only')
    parser.add_argument('--t-end', type=float, default=5.)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--modes', type=int, default=4)
    parser.add_argument('--adam', type=int, default=400)
    parser.add_argument('--lbfgs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--rk-tol', type=float, default=1e-2)
    parser.add_argument('--stage-polish-iters', type=int, default=0,
                        help='Picard stage polish iterations; 0 disables polish')
    parser.add_argument('--stage-polish-tol', type=float, default=1e-10)
    parser.add_argument('--max-refinements', type=int, default=3,
                        help='Additional FNO training rounds; no numerical stage polish')
    parser.add_argument('--dense-points', type=int, default=17)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, default=None,
                        help='Default output depends on whether polish is enabled')
    cfg = parser.parse_args(argv)
    cfg.length = 2*cfg.L if cfg.length is None else cfg.length
    cfg.L = cfg.length/2
    positive = (cfg.length, cfg.dt, cfg.width, cfg.modes, cfg.lr, cfg.rk_tol, cfg.threads,
                cfg.stage_polish_tol)
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error('Lengths, dt, network sizes, lr, rk-tol and threads must be finite and positive')
    if cfg.n < 4 or cfg.n % 2 or cfg.dense_points < 3:
        parser.error('n must be even and >=4; dense-points must be >=3')
    if min(cfg.steps, cfg.adam, cfg.lbfgs, cfg.max_refinements, cfg.stage_polish_iters) < 0:
        parser.error('Step and iteration counts must be nonnegative')
    if not math.isfinite(cfg.t_end) or cfg.t_end < 0 or not math.isfinite(cfg.nu):
        parser.error('t-end must be finite and nonnegative; nu must be finite')

    if cfg.output is None:
        name = 'nls2d_fno_rk_polish_output' if cfg.stage_polish_iters else 'nls2d_fno_rk_plain_output'
        cfg.output = Path(__file__).parent / name
    summaries = {}
    for method in common.TABLEAU_METHODS:
        method_cfg = argparse.Namespace(**vars(cfg))
        method_cfg.output = cfg.output / method
        print(f'Running {method}: output={method_cfg.output}', flush=True)
        summaries[method] = run_method(method_cfg, method)
    return summaries


def run_method(cfg, method):
    method_label, stage_count, main_order = common.TABLEAU_METHODS[method]
    tableau_factory = (common.gauss_legendre_tableau if method == 'gauss_legendre'
                       else common.lobatto_iiia_tableau)
    device = common.configure(cfg.threads, cfg.seed, cfg.device)
    grid = common.PeriodicGrid(cfg.n, cfg.length, device, nu=cfg.nu)
    y = common.initial_state(grid).detach()
    targets = common.invariant_values(grid, y).detach()
    out = cfg.output
    steps_dir = out / 'steps'
    steps_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(cfg), 'output': str(out), 'tableau': method}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    a, b1, _, nodes = tableau_factory(y)
    (out / 'butcher_tableau.json').write_text(json.dumps(dict(
        method=f'{method_label}, {stage_count} stages, no relaxation', main_order=main_order,
        a=a.cpu().tolist(), b1=b1.cpu().tolist(), c=nodes.cpu().tolist(),
    ), indent=2), encoding='utf-8')
    states, times, history = [y.cpu().numpy()], [0.], []
    snapshots = {0.: states[0].copy()}
    np.savez_compressed(steps_dir / 'step_00000000.npz', **grid.coordinate_arrays(),
                        t=0., state=states[0], state_components=['a', 'b'])
    previous_network = None
    t = 0.
    number = 0
    while number < cfg.steps or t < cfg.t_end-1e-12*max(1., cfg.t_end):
        number += 1
        step = common.RKStep(grid, y, cfg.dt, cfg.width, cfg.modes,
                             tableau=method).to(device)
        if previous_network is not None:
            step.network.load_state_dict(previous_network)
        for refinement in range(cfg.max_refinements+1):
            common.train_rk(step, cfg.adam, cfg.lbfgs, cfg.lr)
            with torch.no_grad():
                slopes = step.predict_slopes()
                residuals = torch.stack(step.stage_residuals(step.stage_states(slopes), slopes))
                rk_rms = float(residuals.square().mean().sqrt())
            if not math.isfinite(rk_rms):
                raise RuntimeError(f'Nonfinite RK stage residual at step {number}')
            if rk_rms <= cfg.rk_tol:
                break
        with torch.no_grad():
            stage_info = dict(fno_rk_rms=rk_rms, stage_polish_iterations=0)
            if cfg.stage_polish_iters:
                # Polish only stage equations. The endpoint remains ordinary RK.
                slopes, rk_rms, stage_info = common.polish_rk_stages(
                    step, slopes, cfg.stage_polish_iters, cfg.stage_polish_tol)
            # Only the primary B is used; no relaxation directions are evaluated.
            increment = torch.einsum('i,icxy->cxy', step.b1, torch.stack(slopes))
            y_next = step.y_n+step.h*increment
            if not bool(torch.isfinite(y_next).all()):
                raise RuntimeError(f'Nonfinite RK endpoint at step {number}')
            fractions = torch.linspace(0., 1., cfg.dense_points, device=device, dtype=y.dtype)
            dense, rates = dense_fields(step, slopes, fractions)
            pde_residual = rates-grid.rhs(dense)
            time_rms = pde_residual.square().mean(dim=(-3, -2, -1)).sqrt()
            t_start, t = t, t+step.h
            y = y_next.detach()
            values = common.invariant_values(grid, y)
            errors = values-targets
            row = dict(step=number, t=t, h=step.h, rk_rms=rk_rms,
                       **stage_info,
                       rk_within_tol=rk_rms <= cfg.rk_tol, refinements=refinement,
                       mass=float(values[0]), energy=float(values[1]),
                       mass_error=float(errors[0]), energy_error=float(errors[1]),
                       relative_energy_error=float(errors[1].abs()/targets[1].abs().clamp_min(torch.finfo(y.dtype).eps)),
                       momentum=values[2:].cpu().tolist(), momentum_error=errors[2:].cpu().tolist(),
                       pde_rms=float(pde_residual.square().mean().sqrt()),
                       pde_max_time_rms=float(time_rms.max()))
            for requested in common.SNAPSHOT_TIMES:
                if requested not in snapshots and t_start-1e-12 <= requested <= t+1e-12:
                    fraction = min(1., max(0., (requested-t_start)/step.h))
                    sample, _ = dense_fields(step, slopes, y.new_tensor([fraction]))
                    snapshots[requested] = sample[0].cpu().numpy()
        previous_network = {name: value.detach().clone() for name, value in step.network.state_dict().items()}
        history.append(row)
        states.append(y.cpu().numpy())
        times.append(t)
        print(f'step={number:04d}, t={t:.6f}, RK={rk_rms:.2e}, '
              f'FNO_RK={row["fno_rk_rms"]:.2e}, polish={row["stage_polish_iterations"]}, '
              f'RK_ok={row["rk_within_tol"]}, dM={row["mass_error"]:+.4e}, '
              f'dE={row["energy_error"]:+.4e}, dPx={row["momentum_error"][0]:+.4e}, '
              f'dPy={row["momentum_error"][1]:+.4e}', flush=True)
        np.savez_compressed(
            steps_dir / f'step_{number:08d}.npz', **grid.coordinate_arrays(),
            t=t, t_start=t_start, h=step.h, state_start=step.y_n.cpu().numpy(),
            state=y.cpu().numpy(), K=torch.stack(slopes).cpu().numpy(),
            dense_t=t_start+step.h*fractions.cpu().numpy(), dense_states=dense.cpu().numpy(),
            rk_rms=rk_rms, **stage_info, state_components=['a', 'b'])
        (out / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if history:
        common.save_plots(out, grid.x.cpu().numpy(), times, states, history, snapshots, times)
    available = [time for time in common.SNAPSHOT_TIMES if time in snapshots]
    np.savez_compressed(out / 'time_snapshots.npz', **grid.coordinate_arrays(), t=available,
                        states=np.asarray([snapshots[time] for time in available]),
                        state_components=['a', 'b'], interpolation='integrated RK stage polynomial')
    summary = dict(config=config, method=f'FNO {method_label} without relaxation'
                   + (' + Picard stage polish' if cfg.stage_polish_iters else ''),
                   final_time=t, steps=number, stage_polish=bool(cfg.stage_polish_iters), relaxation_parameters=0,
                   tableau=method, stage_count=stage_count, main_rk_order=main_order,
                   stage_solve='approximate FNO; formal order requires accurate stages',
                   max_mass_error=max((abs(row['mass_error']) for row in history), default=0.),
                   max_energy_error=max((abs(row['energy_error']) for row in history), default=0.),
                   steps_exceeding_rk_tolerance=sum(not row['rk_within_tol'] for row in history))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    main()
