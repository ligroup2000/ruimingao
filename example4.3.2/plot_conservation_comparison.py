"""Compare ROOT/<group>/<method>/history.json without rerunning solvers."""
import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Defaults relative to this script, independent of the IDE working directory.
BASE_DIR = Path(__file__).resolve().parent
# Edit this path to use another dataset when running directly in PyCharm.
DEFAULT_ROOT = BASE_DIR


def load_history(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'history.json'
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'{path}: expected a nonempty history list')
    keys = ['t'] + [key for q in ('mass', 'energy')
                    for key in (q, f'{q}_error', f'relative_{q}_error')]
    for i, row in enumerate(rows):
        for key in keys:
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError(f'{path}: row {i}, missing/nonfinite {key}')
    if rows[0]['t'] < 0 or any(b['t'] <= a['t'] for a, b in zip(rows, rows[1:])):
        raise ValueError(f'{path}: physical time must be nonnegative and increasing')
    return rows


def plot_comparison(runs, output, title=''):
    """runs: (history, label, color, linestyle) entries sharing physical time units."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
    for quantity in ('mass', 'energy'):
        # fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True,
        #                          layout='constrained')
        if quantity == 'mass':
            pos = 0
        else:
            pos = 1
        for rows, label, color, style in runs:
            times = [r['t'] for r in rows]
            values = [r[quantity] for r in rows]
            errors = [r[f'{quantity}_error'] for r in rows]
            relative = [r[f'relative_{quantity}_error'] for r in rows]
            if times[0] > 0:
                # history.json starts after step 1; recover Q(0) = Q(t) - error.
                times = [0.] + times
                values = [values[0] - errors[0]] + values
                errors = [0.] + errors
                relative = [0.] + relative
            axes[pos].plot(times, relative, linestyle=style, label=label, color=color, linewidth=1.5)
            # for ax, data in zip(axes, (values, errors, relative)):
            #     ax.plot(times, data, linestyle=style, color=color,
            #             linewidth=1.5, label=label)
        # axes[0].set_ylabel(quantity.capitalize())
        # axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0),
        #                          useOffset=False)
        # axes[1].set_ylabel(f'{quantity}(t) - {quantity}(0)')
        # axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        # symlog retains exact zeros and resolves small conservation errors.
        axes[pos].set_yscale('symlog', linthresh=1e-16)
        axes[pos].set_ylabel(f'Relative {quantity} error')
        axes[pos].set_xlabel('t')
        axes[pos].grid(True, alpha=0.3)
        axes[pos].legend()
        # for ax in axes:
        #     ax.grid(True, alpha=.3)
        #     ax.legend()
    fig.suptitle(f'{title + ": " if title else ""} comparison conservation')
    for ext in ('png', 'pdf'):
        target = output / f'quantity_comparison.{ext}'
        fig.savefig(target, dpi=200)
        print(target.resolve())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT, type=Path,
                        help='Directory containing group/method/history.json')
    parser.add_argument('--groups', nargs='+', default=['mrrk', 'rk'],
                        help='Group subdirectory names; distinguished by line style')
    parser.add_argument('--methods', nargs='+', default=['gauss', 'lobatto'],
                        help='Method subdirectory names; distinguished by color')
    parser.add_argument('--output', type=Path,
                        help='Default: ROOT/conservation_comparison')
    parser.add_argument('--title', default='')
    args = parser.parse_args()
    root = args.root.resolve()
    paths = [(group, method, root / group / method / 'history.json')
             for group in args.groups for method in args.methods]
    missing = [str(path) for _, _, path in paths if not path.is_file()]
    if missing:
        parser.error('Missing history files (check --root/--groups/--methods):\n'
                     + '\n'.join(missing))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    styles = ['-', '--', '-.', ':']
    runs = []
    for group, method, path in paths:
        rows = load_history(path)
        if group == 'mrrk':
            group0 = 'relaxation'
        else:
            group0 = 'unrelaxation'
        label = f'{group0}-{method.capitalize()}'
        color = colors[args.methods.index(method) % len(colors)]
        style = styles[args.groups.index(group) % len(styles)]
        runs.append((rows, label, color, style))
        print(f'{label}: {len(rows)} steps, t={rows[0]["t"]:g}..{rows[-1]["t"]:g}')
    plot_comparison(runs, args.output or root / 'conservation_comparison', args.title)


if __name__ == '__main__':
    main()
