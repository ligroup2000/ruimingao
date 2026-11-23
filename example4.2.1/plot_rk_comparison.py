"""Plot saved MRRK/RK results only; never train or advance either solver."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(directory):
    with np.load(directory / 'solution.npz', allow_pickle=False) as data:
        result = {key: data[key].copy() for key in ['x', 't', 'states', 'masses', 'energies']}
    t, states = result['t'], result['states']
    if t.ndim != 1 or states.shape != (len(t), 2, len(result['x'])):
        raise ValueError(f'Invalid solution dimensions: {directory}')
    if not np.all(np.diff(t) > 0):
        raise ValueError(f'Time must strictly increase: {directory}')
    for value in result.values():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite data: {directory}')
    if any(result[key].shape != t.shape for key in ['masses', 'energies']):
        raise ValueError(f'Invalid invariant dimensions: {directory}')
    return result


def interpolate(data, times):
    """Interpolate saved real/imaginary endpoints, with no extrapolation."""
    times = np.asarray(times)
    if times.min() < data['t'][0] or times.max() > data['t'][-1]:
        raise ValueError('Requested times outside saved interval')
    right = np.searchsorted(data['t'], times, side='right').clip(1, len(data['t'])-1)
    left = right-1
    weight = ((times-data['t'][left])/(data['t'][right]-data['t'][left]))[:, None, None]
    return (1-weight)*data['states'][left]+weight*data['states'][right]


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=250, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def snapshot_values(directory, data, times):
    """Prefer saved snapshots, then native endpoints, then temporal interpolation."""
    values, sources = [], []
    snapshot_path = directory / 'time_snapshots.npz'
    with np.load(snapshot_path, allow_pickle=False) as saved:
        if not np.array_equal(saved['x'], data['x']):
            raise ValueError(f'Snapshot grid mismatch: {directory}')
        for t in times:
            matches = np.flatnonzero(np.isclose(saved['t'], t, rtol=0, atol=1e-12))
            native = np.flatnonzero(np.isclose(data['t'], t, rtol=0, atol=1e-12))
            if len(matches):
                values.append(saved['states'][matches[0]])
                sources.append('saved snapshot')
            elif len(native):
                values.append(data['states'][native[0]])
                sources.append('saved RK endpoint')
            else:
                values.append(interpolate(data, [t])[0])
                sources.append('linear time interpolation')
    return np.asarray(values), sources


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mrrk', type=Path, default=base / 'nls_fno_mrrk_gauss3_pde_output')
    parser.add_argument('--rk', type=Path, default=base / 'nls_fno_rk_T50_output')
    parser.add_argument('--output', type=Path, default=base / 'rk_mrrk_comparison')
    args = parser.parse_args()
    data = [load(args.mrrk), load(args.rk)]
    if not np.array_equal(data[0]['x'], data[1]['x']):
        raise ValueError('The spatial grids must agree')
    start = max(d['t'][0] for d in data)
    end = min(d['t'][-1] for d in data)
    if end <= start:
        raise ValueError('No common time interval')
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    labels = ['FNO + MRRK (relaxation)', 'FNO + RK (no relaxation)']
    colors = ['#0072B2', '#D55E00']
    styles = ['-', '--']
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    metrics = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for d, label, color, style in zip(data, labels, colors, styles):
        mask = (d['t'] >= start) & (d['t'] <= end)
        row = {'method': label, 'saved_final_time': float(d['t'][-1]),
               'last_native_time_in_comparison': float(d['t'][mask][-1])}
        for col, (key, symbol) in enumerate([('masses', 'M'), ('energies', 'E')]):
            error = d[key][mask]-d[key][0]
            relative = np.abs(error)/max(abs(d[key][0]), np.finfo(float).eps)
            axes[0, col].plot(d['t'][mask], error, style, color=color, label=label)
            # Zero error cannot appear on a logarithmic axis; leave it unplotted.
            axes[1, col].semilogy(d['t'][mask], np.where(relative > 0, relative, np.nan),
                                  style, color=color, label=label)
            axes[0, col].set(title=f'{"Mass" if col == 0 else "Energy"} drift',
                             ylabel=f'{symbol}(t) - {symbol}(0)')
            axes[0, col].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            axes[1, col].set(ylabel=f'|{symbol}(t)-{symbol}(0)| / |{symbol}(0)|')
            row[f'max_abs_{symbol}_drift'] = float(np.max(np.abs(error)))
            row[f'max_relative_{symbol}_drift'] = float(np.max(relative))
        metrics.append(row)
    for ax in axes.flat:
        ax.set(xlabel='Physical time (s)', xlim=(start, end))
        ax.grid(True, alpha=.25, which='both')
        ax.legend(fontsize=9)
    fig.suptitle('Conservation comparison (native saved time points)')
    save(fig, output, 'conservation_comparison')

    # Separate panels retain visibility of the much smaller MRRK drift.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout='constrained')
    for col, (d, label, color) in enumerate(zip(data, labels, colors)):
        mask = (d['t'] >= start) & (d['t'] <= end)
        for ax, key, symbol in zip(axes[:, col], ['masses', 'energies'], ['M', 'E']):
            ax.plot(d['t'][mask], d[key][mask]-d[key][0], color=color)
            ax.set(xlabel='Physical time (s)', ylabel=f'{symbol}(t)-{symbol}(0)',
                   title=label, xlim=(start, end))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
            ax.grid(True, alpha=.25)
    fig.suptitle('Conservation drift (independent vertical scales)')
    save(fig, output, 'conservation_individual_scales')

    times = np.linspace(start, end, 501)
    aligned = np.stack([interpolate(d, times) for d in data])
    amplitude = np.linalg.norm(aligned, axis=2)
    x = data[0]['x']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, layout='constrained')
    for ax, values, label in zip(axes[:2], amplitude, labels):
        mesh = ax.pcolormesh(x, times, values, shading='auto', cmap='viridis',
                             vmin=0, vmax=amplitude.max())
        ax.set(title=label, xlabel='x')
    fig.colorbar(mesh, ax=list(axes[:2]), label='|u(x,t)|', shrink=.9)
    difference = np.linalg.norm(aligned[0]-aligned[1], axis=1)
    mesh = axes[2].pcolormesh(x, times, difference, shading='auto', cmap='magma', vmin=0)
    axes[2].set(title='Complex solution difference', xlabel='x')
    fig.colorbar(mesh, ax=axes[2], label='|u_MRRK - u_RK|', shrink=.9)
    axes[0].set_ylabel('Physical time (s)')
    fig.suptitle('Evolution comparison; real/imaginary endpoints linearly interpolated in time')
    save(fig, output, 'evolution_comparison')

    snapshot_times = np.asarray([0., 1., 3., 5., 7., 9.])
    if snapshot_times[0] < start or snapshot_times[-1] > end:
        raise ValueError('Saved solutions must cover all requested snapshot times: 0,1,3,5,7,9')
    snapshot_results = [snapshot_values(directory, d, snapshot_times)
                        for directory, d in zip([args.mrrk, args.rk], data)]
    snapshot_states = np.stack([result[0] for result in snapshot_results])
    # Given exact solution on the real line; on this time interval its peak stays
    # inside the domain. It is not an exact finite-periodic-domain solution.
    phase = 2*x[None, :]+3*snapshot_times[:, None]
    envelope = 1/np.cosh(x[None, :]+4*snapshot_times[:, None])
    snapshot_reference = np.stack([envelope*np.cos(phase), -envelope*np.sin(phase)], axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True,
                             layout='constrained')
    for i, (ax, t) in enumerate(zip(axes.flat, snapshot_times)):
        ax.plot(x, np.linalg.norm(snapshot_reference[i], axis=0), color='black',
                linewidth=1.8, alpha=.8, label='Exact soliton (real line)', zorder=1)
        for values, label, color, style in zip(snapshot_states, labels, colors, styles):
            ax.plot(x, np.linalg.norm(values[i], axis=0), style, color=color, label=label,
                    marker='o' if style == '-' else 's', markevery=8,
                    markersize=3, markerfacecolor='none', linewidth=1.2)
        ax.set(title=f't = {t:g} s', xlabel='x', ylabel='|u(x,t)|')
        ax.grid(True, alpha=.25)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('Amplitude snapshots: MRRK, RK and exact soliton')
    fig.supxlabel('Exact reference is on the real line; not a strict periodic-domain exact solution.',
                  fontsize=9)
    save(fig, output, 'snapshot_comparison')

    with (output / 'conservation_metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    np.savez_compressed(output / 'aligned_comparison.npz', x=x, t=times,
                        states=aligned, labels=np.asarray(labels),
                        complex_difference=difference,
                        snapshot_times=snapshot_times, snapshot_states=snapshot_states,
                        snapshot_reference=snapshot_reference)
    notes = dict(
        sources=[str(args.mrrk.resolve()), str(args.rk.resolve())],
        comparison_interval=[float(start), float(end)],
        conservation='Original saved masses/energies relative to each initial value; no interpolation.',
        evolution='Linear time interpolation of real and imaginary saved endpoint states; no extrapolation.',
        difference='Magnitude of complex solution difference; not an error against an exact solution.',
        snapshot_times=snapshot_times.tolist(),
        snapshot_sources={label: result[1] for label, result in zip(labels, snapshot_results)},
        reference='Snapshots: sech(x+4*t)*exp(-i*(2*x+3*t)); exact on the real line, not strictly periodic. No reference used for 0-50 s evolution.',
        caveat='Runs also differ in optimizer stopping/line search, extra training and continuous network; not an isolated relaxation ablation.',
        metrics=metrics,
    )
    (output / 'comparison_notes.json').write_text(json.dumps(notes, indent=2), encoding='utf-8')
    print(json.dumps(notes, indent=2))
    print(f'Saved PNG/PDF figures to {output}')


if __name__ == '__main__':
    main()
