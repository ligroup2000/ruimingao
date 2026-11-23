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
    keys = ['t'] + [key for key in ('gamma1', 'gamma2')]
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
    for method, rows, color in runs:
        if method == 'gauss':
            pos = 0
        else:
            pos = 1
        times = [r['t'] for r in rows]
        gamma1 = [r['gamma1'] for r in rows]; gamma1 = [abs(r) for r in gamma1]
        gamma2 = [r['gamma2'] for r in rows]; gamma2 = [abs(r) for r in gamma2]
        axes[pos].plot(times, gamma1, linestyle='-', label='$\gamma_1$', color='r', linewidth=1.5)
        axes[pos].plot(times, gamma2, linestyle='--', label='$\gamma_2$', color='b', linewidth=1.5)
        axes[pos].set_yscale('log')
        axes[pos].set_ylabel(f'relaxation parameters')
        axes[pos].set_xlabel('t')
        axes[pos].grid(True, alpha=0.3)
        axes[pos].legend()
        axes[pos].set_title(f'{method.title()}')
        # for ax in axes:
        #     ax.grid(True, alpha=.3)
        #     ax.legend()
    fig.suptitle(f'{title + ": " if title else ""} relaxation parameters')
    for ext in ('png', 'pdf'):
        target = output / f'gamma.{ext}'
        fig.savefig(target, dpi=200)
        print(target.resolve())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT, type=Path,
                        help='Directory containing group/method/history.json')
    parser.add_argument('--groups', nargs='+', default=['mrrk_output'],
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
        color = colors[args.methods.index(method) % len(colors)]
        runs.append((method, rows,  color))
        print(f'{method}: {len(rows)} steps, t={rows[0]["t"]:g}..{rows[-1]["t"]:g}')
    plot_comparison(runs, args.output or root / 'conservation_comparison', args.title)


if __name__ == '__main__':
    main()
