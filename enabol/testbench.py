from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDEX_COLUMNS = ['epoch', 'sample', 'global_step', 'sample_index']


class TestbenchData:
    """Container for hls4ml testbench outputs and trainable CSIM traces."""

    __test__ = False

    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        metadata: dict[str, str] | None = None,
        metadata_by_trace: dict[str, dict[str, str]] | None = None,
        source_dir: str | Path | None = None,
    ):
        self.frame = frame
        self.metadata = metadata or {}
        self.metadata_by_trace = metadata_by_trace or {}
        self.source_dir = Path(source_dir) if source_dir is not None else None

    def __repr__(self) -> str:
        rows = [
            ('Directory', str(self.source_dir) if self.source_dir is not None else '<unknown>'),
            ('Rows', str(len(self.frame))),
            ('Index', ', '.join(self.frame.index.names)),
            ('Traces', ', '.join(sorted(self.metadata_by_trace)) or '<none>'),
            ('Metrics', ', '.join(self.metrics) or '<none>'),
        ]
        for key in ['Project', 'Backend', 'Controller', 'Optimizer', 'Loss', 'Epochs', 'BatchSize']:
            if key in self.metadata:
                rows.append((key, self.metadata[key]))

        key_width = max(len(key) for key, _ in rows)
        value_width = max(len(value) for _, value in rows)
        border = '+' + '-' * (key_width + 2) + '+' + '-' * (value_width + 2) + '+'
        lines = ['TestbenchData', border]
        lines.extend(f'| {key:<{key_width}} | {value:<{value_width}} |' for key, value in rows)
        lines.append(border)
        return '\n'.join(lines)

    def __getitem__(self, key: str) -> pd.Series:
        return self.frame[key]

    @property
    def metrics(self) -> list[str]:
        return list(self.frame.columns)

    @classmethod
    def from_dir(cls, path: str | Path) -> 'TestbenchData':
        """Load hls4ml trainable traces.

        ``path`` may be the hls4ml output directory, the ``tb_data`` directory,
        or the final ``tb_data/training`` directory.
        """

        trace_dir = _resolve_training_trace_dir(path)
        trace_files = sorted(trace_dir.glob('*.dat'))
        if not trace_files:
            raise FileNotFoundError(f'No trainable trace .dat files found in {trace_dir}.')

        frames = []
        metadata_by_trace = {}
        for trace_file in trace_files:
            comments, metadata = _read_metadata(trace_file)
            frame = pd.read_csv(trace_file, comment='#')
            missing = [col for col in INDEX_COLUMNS if col not in frame.columns]
            if missing:
                raise ValueError(f'Trace file {trace_file} is missing index columns: {missing}')
            frame = frame.set_index(INDEX_COLUMNS)
            frames.append(frame)
            trace_name = metadata.get('Trace', trace_file.stem)
            metadata['_comments'] = '\n'.join(comments)
            metadata_by_trace[trace_name] = metadata

        merged = pd.concat(frames, axis=1).sort_index()
        first_metadata = next(iter(metadata_by_trace.values()), {})

        return cls(frame=merged, metadata=dict(first_metadata), metadata_by_trace=metadata_by_trace, source_dir=trace_dir)

    @classmethod
    def from_trainable_dir(cls, path: str | Path) -> 'TestbenchData':
        return cls.from_dir(path)

    def plot_training(
        self,
        metrics: list[str] | tuple[str, ...] = ('loss', 'alpha'),
        *,
        window_size: int = 30,
        levels: int = 3,
        show_metadata: bool = True,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ):
        metrics = [metric for metric in metrics if metric in self.frame.columns]
        if not metrics:
            raise ValueError(f'None of the requested metrics are available. Available metrics: {list(self.frame.columns)}')

        metadata_lines = _metadata_lines(self.metadata) if show_metadata else []
        n_metric_axes = len(metrics)
        nrows = n_metric_axes + (1 if metadata_lines else 0)
        if figsize is None:
            figsize = (11, 3.8 * n_metric_axes + (1.1 if metadata_lines else 0.4))

        height_ratios = ([0.45] if metadata_lines else []) + [1.0] * n_metric_axes
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(nrows=nrows, ncols=1, height_ratios=height_ratios)

        row = 0
        if metadata_lines:
            meta_ax = fig.add_subplot(grid[row, 0])
            _draw_metadata_box(meta_ax, metadata_lines)
            row += 1

        axes = []
        for i, metric in enumerate(metrics):
            ax = fig.add_subplot(grid[row + i, 0], sharex=axes[0] if axes else None)
            axes.append(ax)
            _turn_grid_on(ax)
            _plot_metric(ax, self.frame, metric, window_size=window_size, levels=levels)
            if metric == 'loss':
                ax.set_yscale('log')
                ax.set_ylabel('Loss')
            elif metric == 'alpha':
                ax.set_ylabel(r'Alpha ($\alpha$)')
            else:
                ax.set_ylabel(metric)

        _add_epoch_axis(axes[0], self.frame)
        axes[-1].set_xlabel('Global Step')
        if title:
            fig.suptitle(title)

        if show:
            plt.show()
        return fig, axes


def _resolve_training_trace_dir(path: str | Path) -> Path:
    root = Path(path)
    candidates = [
        root,
        root / 'training',
        root / 'tb_data' / 'training',
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob('*.dat')):
            return candidate
    raise FileNotFoundError(f'Could not find trainable trace .dat files under {root}.')


def _read_metadata(path: Path) -> tuple[list[str], dict[str, str]]:
    comments = []
    metadata = {}
    with path.open() as handle:
        for line in handle:
            if not line.startswith('#'):
                break
            comment = line[1:].strip()
            comments.append(comment)
            if ':' in comment:
                key, value = comment.split(':', 1)
                metadata[key.strip()] = value.strip()
    return comments, metadata


def _metadata_lines(metadata: dict[str, str]) -> list[str]:
    ordered_keys = [
        'Trace',
        'Generated by',
        'ENABOL version',
        'hls4ml-trainable version',
        'User',
        'Host',
        'Date',
        'Project',
        'Backend',
        'Controller',
        'Optimizer',
        'LearningRate',
        'Loss',
        'BatchSize',
        'Epochs',
        'Shuffle',
        'ShuffleSeed',
        'LogEvery',
    ]
    return [f'{key}: {metadata[key]}' for key in ordered_keys if key in metadata]


def _draw_metadata_box(ax, metadata_lines: list[str]) -> None:
    ax.axis('off')
    text = textwrap.fill('   '.join(metadata_lines), width=130)
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha='center',
        va='center',
        fontsize=9,
        bbox={'boxstyle': 'round', 'facecolor': '#fff4da', 'edgecolor': '#8a7a60', 'alpha': 0.9},
    )


def _turn_grid_on(ax) -> None:
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.75, alpha=0.45)
    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle='--', linewidth=0.5, alpha=0.18)


def _normal_quantile(level: int) -> tuple[float, float]:
    upper = 0.5 * (1 + math.erf(level / math.sqrt(2)))
    lower = 1 - upper
    return lower, upper


def _rolling_statistics(series: pd.Series, window_size: int, levels: int) -> dict[str, Any]:
    stats: dict[str, Any] = {'mean': series.rolling(window=window_size, min_periods=1).mean()}
    for level in range(1, levels + 1):
        lower, upper = _normal_quantile(level)
        stats[level] = {
            'lower': series.rolling(window=window_size, min_periods=1).quantile(lower),
            'upper': series.rolling(window=window_size, min_periods=1).quantile(upper),
            'coverage': upper - lower,
        }
    return stats


def _plot_metric(ax, frame: pd.DataFrame, metric: str, *, window_size: int, levels: int) -> None:
    series = frame[metric].dropna()
    x = series.index.get_level_values('global_step').to_numpy()
    stats = _rolling_statistics(series, window_size=window_size, levels=levels)
    color = '#1f77b4' if metric == 'loss' else '#111111'
    shade_color = '#5b6dff' if metric == 'loss' else '#ffb347'
    label = 'Loss' if metric == 'loss' else ('Alpha' if metric == 'alpha' else metric)

    ax.plot(x, stats['mean'].to_numpy(), '-', label=rf'{label} ($\mu$)', color=color, linewidth=1.8)
    for level in range(levels, 0, -1):
        interval = stats[level]
        ax.fill_between(
            x,
            interval['lower'].to_numpy(),
            interval['upper'].to_numpy(),
            alpha=0.07 * (levels - level + 1),
            color=shade_color,
            label=rf'$\pm$ {level}$-\sigma$ ({interval["coverage"]:.2%})',
        )

    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title=f'Smoothed {label}')


def _add_epoch_axis(ax, frame: pd.DataFrame) -> None:
    epoch_values = frame.index.get_level_values('epoch').to_numpy()
    step_values = frame.index.get_level_values('global_step').to_numpy()
    ticks = []
    labels = []
    for epoch in np.unique(epoch_values):
        positions = step_values[epoch_values == epoch]
        if positions.size:
            ticks.append(int(positions.min()))
            labels.append(str(epoch))
    epoch_ax = ax.twiny()
    epoch_ax.set_xlim(ax.get_xlim())
    epoch_ax.set_xticks(ticks)
    epoch_ax.set_xticklabels(labels)
    epoch_ax.set_xlabel('Epoch')
