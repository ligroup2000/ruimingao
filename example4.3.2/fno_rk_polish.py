# -*- coding: utf-8 -*-
"""3维周期 Schrödinger--Klein--Gordon 耦合算例4.3.2。

i*u_t + kappa1/2*Delta(u) + xi*u*v = 0，
v_tt - kappa2*Delta(v) + mu^2*v - xi*|u|^2 = 0。
默认区域 [-20,20)^2，不重复右端点；state=(Re(u),Im(u),v,w)，w=v_t。

No stage polish, second network or relaxation.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from fno_mrrk_newton import (
    COEFFICIENTS, STATE_COMPONENTS, PeriodicGrid, RKStep,
    configure, initial_state, optimize,
    gauss_legendre_tableau, lobatto_iiia_tableau,
)


def unrelaxed_endpoint(step, K):
    """Ordinary RK endpoint and physical duration; no invariant solve."""
    d1, _ = step.rk_directions(K)
    change = step.h*d1
    return step.y_n+change, step.h, change


def plot_errors(output, history, reference=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for quantity in ('mass', 'energy'):
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, layout='constrained')
        for rows, label in ((history, 'FNO RK (no polish, no relaxation)'), (reference, 'MRRK')):
            if rows is None:
                continue
            times = [0.] + [r['t'] for r in rows]
            axes[0].plot(times, [0.] + [r[f'{quantity}_error'] for r in rows], label=label)
            axes[1].plot(times, [0.] + [r[f'relative_{quantity}_error'] for r in rows], label=label)
        axes[0].set_ylabel(f'{quantity}(t) - {quantity}(0)')
        axes[1].set(xlabel='Physical time t', ylabel=f'Relative {quantity} error')
        for ax in axes:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.grid(True, alpha=.3)
            ax.legend()
        fig.savefig(output/f'{quantity}_comparison.png', dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=20)
    parser.add_argument('--L', type=float, default=10.)
    parser.add_argument('--length', type=float, default=None)
    parser.add_argument('--dt', type=float, default=.1)
    parser.add_argument('--t-end', type=float, default=50.)
    parser.add_argument('--steps', type=int, default=0, help='Minimum steps, in addition to t-end')
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--modes', type=int, default=4)
    parser.add_argument('--adam', type=int, default=400)
    parser.add_argument('--lbfgs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.003)
    parser.add_argument('--rk-tol', type=float, default=1e-2)
    parser.add_argument('--max-refinements', type=int, default=3)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', default='kgs3d_guo43_fno_rk_polish_output')
    parser.add_argument('--method', choices=['both','gauss','lobatto'], default='both')
    parser.add_argument('--compare-to', type=Path, help='MRRK output directory containing history.json; with --method both, use the base path before _gauss/_lobatto')
    cfg = parser.parse_args(argv)
    length = cfg.length if cfg.length is not None else 2*cfg.L
    positive = (length, cfg.dt, cfg.width, cfg.modes, cfg.lr, cfg.rk_tol, cfg.threads)
    if any(not math.isfinite(v) or v <= 0 for v in positive):
        parser.error('Dimensions, dt, lengths and tolerances must be finite and positive')
    if min(cfg.adam, cfg.lbfgs, cfg.max_refinements, cfg.steps) < 0:
        parser.error('Require nonnegative iteration counts')
    if not math.isfinite(cfg.t_end) or cfg.t_end < 0:
        parser.error('t-end must be finite and nonnegative')
    methods = ('gauss','lobatto') if cfg.method == 'both' else (cfg.method,)
    summaries = {}
    for method in methods:
        run_cfg = argparse.Namespace(**vars(cfg))
        run_cfg.method = method
        if cfg.method == 'both':
            base = Path(cfg.output)
            run_cfg.output = str(base.with_name(base.name+'_'+method))
            if cfg.compare_to is not None:
                base = cfg.compare_to
                run_cfg.compare_to = base.with_name(base.name+'_'+method)
        summaries[method] = run_method(run_cfg)
    return summaries if cfg.method == 'both' else summaries[cfg.method]


def run_method(cfg):
    length = cfg.length if cfg.length is not None else 2*cfg.L
    method_name = {'gauss':'Gauss', 'lobatto':'Lobatto IIIA'}[cfg.method]
    reference = None
    if cfg.compare_to is not None:
        reference = json.loads((cfg.compare_to/'history.json').read_text(encoding='utf-8'))
    device = configure(cfg.threads, cfg.seed, cfg.device)
    grid = PeriodicGrid(cfg.n, length, device)
    y = initial_state(grid)
    targets = grid.invariants(y).detach()
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    tableau = {'gauss':gauss_legendre_tableau, 'lobatto':lobatto_iiia_tableau}[cfg.method]
    a,b1,b2,c = tableau(y)
    (output/'butcher_tableau.json').write_text(json.dumps(dict(
        method=method_name, stage_count=len(c), main_order=6, embedded_order=1,
        a=a.cpu().tolist(), b1=b1.cpu().tolist(), b2=b2.cpu().tolist(),
        c=c.cpu().tolist()),indent=2),encoding='utf-8')
    history, previous = [], None
    while len(history) < cfg.steps or len(history)*cfg.dt < cfg.t_end:
        step = RKStep(grid, y, cfg.dt, cfg.width, cfg.modes, method=cfg.method).to(device)
        if previous is not None:
            step.network.load_state_dict(previous)
        for refinement in range(cfg.max_refinements+1):
            try:
                optimize(step.network, step.rk_loss, cfg.adam, cfg.lbfgs, cfg.lr)
                with torch.no_grad():
                    K = step.predict_slopes()
                    fno_rms = float(step.residual(K).square().mean().sqrt())
                    rk_rms = fno_rms
                    if not math.isfinite(rk_rms) or rk_rms > cfg.rk_tol:
                        raise RuntimeError(f'RK network residual {rk_rms}')
                    y_next, _, _ = unrelaxed_endpoint(step, K)
                    if not bool(torch.isfinite(y_next).all()):
                        raise RuntimeError('Nonfinite RK endpoint')
                break
            except RuntimeError:
                if refinement == cfg.max_refinements:
                    raise
        y = y_next.detach()
        previous = {k:v.detach().clone() for k,v in step.network.state_dict().items()}
        values = grid.invariants(y)
        row = dict(step=len(history)+1, t=(len(history)+1)*cfg.dt, h=cfg.dt,
                   physical_time_step=cfg.dt, rk_rms=rk_rms, fno_rk_rms=fno_rms)
        for i, name in enumerate(('mass', 'energy')):
            error = float(values[i]-targets[i])
            row[name] = float(values[i])
            row[f'{name}_error'] = error
            row[f'relative_{name}_error'] = abs(error)/max(abs(float(targets[i])), torch.finfo(y.dtype).eps)
        history.append(row)
        (output/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(f'{method_name} step={row["step"]}, t={row["t"]:.6f}, RK={rk_rms:.3e}, '
              f'dE={row["energy_error"]:+.3e}, dM={row["mass_error"]:+.3e}', flush=True)
    summary = dict(config={**vars(cfg), 'compare_to':str(cfg.compare_to) if cfg.compare_to else None},
                   coefficients=COEFFICIENTS, state_components=STATE_COMPONENTS,
                   method=f'3D single-FNO {method_name} RK, no polish or relaxation',
                   stage_count=len(c), main_rk_order=6, embedded_order=1,
                   enforced_invariants=[], diagnostic_order=['mass', 'energy'],
                   steps=len(history), final_time=len(history)*cfg.dt,
                   length=length, boundary='periodic in x1,x2,x3')
    for i, name in enumerate(('mass', 'energy')):
        summary[f'initial_{name}'] = float(targets[i])
        summary[f'final_{name}'] = float(grid.invariants(y)[i])
        summary[f'max_{name}_error'] = max((abs(r[f'{name}_error']) for r in history), default=0.)
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot_errors(output, history, reference)
    return summary


if __name__ == '__main__':
    main()
