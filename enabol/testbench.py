from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDEX_COLUMNS = ['epoch', 'sample', 'global_step', 'sample_index']
PARAMETER_TRACE_NAMES = {'weights', 'biases'}
PARAMETER_STAT_COLUMNS = [
    'mean',
    'std',
    'min',
    'max',
    'median',
    'q05',
    'q25',
    'q75',
    'q95',
    'norm_l2',
    'norm_inf',
    'sparsity_fraction',
    'saturation_fraction',
    'near_rail_fraction',
    'underflow_fraction',
]


@dataclass
class TestbenchLayerData:
    """Per-layer parameter traces loaded from ``tb_data/training/<layer>``."""

    name: str
    traces: dict[str, pd.DataFrame]
    metadata_by_trace: dict[str, dict[str, str]]

    @property
    def weights(self) -> pd.DataFrame | None:
        return self.traces.get('weights')

    @property
    def biases(self) -> pd.DataFrame | None:
        return self.traces.get('biases')

    @property
    def stats(self) -> pd.DataFrame:
        frames = []
        for trace_name, frame in self.traces.items():
            stats = _parameter_stats_frame(frame)
            stats = stats.rename(columns={column: f'{trace_name}.{column}' for column in stats.columns})
            frames.append(stats)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).sort_index()


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
        layers: dict[str, TestbenchLayerData] | None = None,
    ):
        self.frame = frame
        self.metadata = metadata or {}
        self.metadata_by_trace = metadata_by_trace or {}
        self.source_dir = Path(source_dir) if source_dir is not None else None
        self.layers = layers or {}

    def __repr__(self) -> str:
        rows = [
            ('Directory', str(self.source_dir) if self.source_dir is not None else '<unknown>'),
            ('Rows', str(len(self.frame))),
            ('Index', ', '.join(self.frame.index.names)),
            ('Traces', ', '.join(sorted(self.metadata_by_trace)) or '<none>'),
            ('Metrics', ', '.join(self.metrics) or '<none>'),
            ('Layers', ', '.join(sorted(self.layers)) or '<none>'),
            ('Layer Stats', ', '.join(self.layer_metrics) or '<none>'),
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

    @property
    def stats_frame(self) -> pd.DataFrame:
        frames = []
        for layer_name, layer in self.layers.items():
            stats = layer.stats
            if not stats.empty:
                stats = stats.rename(columns={column: f'{layer_name}.{column}' for column in stats.columns})
                frames.append(stats)
        if not frames:
            return pd.DataFrame(index=self.frame.index)
        return pd.concat(frames, axis=1).sort_index()

    @property
    def scalar_frame(self) -> pd.DataFrame:
        stats = self.stats_frame
        if stats.empty:
            return self.frame
        return pd.concat([self.frame, stats], axis=1).sort_index()

    @property
    def layer_metrics(self) -> list[str]:
        return list(self.stats_frame.columns)

    @property
    def scalar_metrics(self) -> list[str]:
        return list(self.scalar_frame.columns)

    @classmethod
    def from_dir(cls, path: str | Path, *, load_weights: bool | list[str] | tuple[str, ...] | set[str] = True) -> 'TestbenchData':
        """Load hls4ml trainable traces.

        ``path`` may be the hls4ml output directory, the ``tb_data`` directory,
        or the final ``tb_data/training`` directory.
        """

        trace_dir = _resolve_training_trace_dir(path)
        trace_files = sorted(trace_dir.rglob('*.dat'))
        if not trace_files:
            raise FileNotFoundError(f'No trainable trace .dat files found in {trace_dir}.')

        frames = []
        layer_traces: dict[str, dict[str, pd.DataFrame]] = {}
        layer_metadata: dict[str, dict[str, dict[str, str]]] = {}
        metadata_by_trace = {}
        for trace_file in trace_files:
            comments, metadata = _read_metadata(trace_file)
            frame = pd.read_csv(trace_file, comment='#')
            missing = [col for col in INDEX_COLUMNS if col not in frame.columns]
            if missing:
                raise ValueError(f'Trace file {trace_file} is missing index columns: {missing}')
            frame = frame.set_index(INDEX_COLUMNS)
            trace_name = metadata.get('Trace', _trace_name_from_path(trace_dir, trace_file))
            metadata['_comments'] = '\n'.join(comments)
            metadata['_path'] = str(trace_file.relative_to(trace_dir))

            parameter_info = _parameter_trace_info(trace_dir, trace_file)
            if parameter_info is not None:
                layer_name, trace_kind = parameter_info
                if _should_load_layer(layer_name, load_weights):
                    layer_traces.setdefault(layer_name, {})[trace_kind] = frame
                    layer_metadata.setdefault(layer_name, {})[trace_kind] = metadata
                    metadata_by_trace[trace_name] = metadata
                continue

            frames.append(frame)
            metadata_by_trace[trace_name] = metadata

        if frames:
            merged = pd.concat(frames, axis=1).sort_index()
        else:
            merged = pd.DataFrame()
            merged.index.names = INDEX_COLUMNS

        first_metadata = next(iter(metadata_by_trace.values()), {})
        layers = {
            layer_name: TestbenchLayerData(
                name=layer_name,
                traces=traces,
                metadata_by_trace=layer_metadata.get(layer_name, {}),
            )
            for layer_name, traces in layer_traces.items()
        }

        return cls(
            frame=merged,
            metadata=dict(first_metadata),
            metadata_by_trace=metadata_by_trace,
            source_dir=trace_dir,
            layers=layers,
        )

    @classmethod
    def from_trainable_dir(
        cls,
        path: str | Path,
        *,
        load_weights: bool | list[str] | tuple[str, ...] | set[str] = True,
    ) -> 'TestbenchData':
        return cls.from_dir(path, load_weights=load_weights)

    def plot_training(
        self,
        metrics: list[str] | tuple[str, ...] | None = None,
        *,
        window_size: int = 30,
        levels: int = 3,
        show_metadata: bool = True,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ):
        plot_frame = self.scalar_frame
        if metrics is None:
            metrics = list(plot_frame.columns)
        metrics = [metric for metric in metrics if metric in plot_frame.columns]
        if not metrics:
            raise ValueError(f'None of the requested metrics are available. Available metrics: {list(plot_frame.columns)}')

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
            _plot_metric(ax, plot_frame, metric, window_size=window_size, levels=levels)
            if metric == 'loss':
                ax.set_yscale('log')
                ax.set_ylabel('Loss')
            elif metric == 'alpha':
                ax.set_ylabel(r'Alpha ($\alpha$)')
            else:
                ax.set_ylabel(metric)

        _add_epoch_axis(axes[0], plot_frame)
        axes[-1].set_xlabel('Global Step')
        if title:
            fig.suptitle(title)

        if show:
            plt.show()
        return fig, axes


def _resolve_training_trace_dir(path: str | Path) -> Path:
    root = Path(path)
    candidates = [
        root / 'tb_data' / 'training',
        root / 'training',
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob('*.dat')):
            return candidate
    raise FileNotFoundError(f'Could not find trainable trace .dat files under {root}.')


def _trace_name_from_path(trace_dir: Path, trace_file: Path) -> str:
    return trace_file.relative_to(trace_dir).with_suffix('').as_posix()


def _parameter_trace_info(trace_dir: Path, trace_file: Path) -> tuple[str, str] | None:
    relative = trace_file.relative_to(trace_dir)
    if len(relative.parts) != 2:
        return None

    layer_name = relative.parts[0]
    trace_kind = relative.with_suffix('').parts[1]
    if trace_kind not in PARAMETER_TRACE_NAMES:
        return None
    return layer_name, trace_kind


def _should_load_layer(layer_name: str, load_weights: bool | list[str] | tuple[str, ...] | set[str]) -> bool:
    if isinstance(load_weights, bool):
        return load_weights
    return layer_name in set(load_weights)


def _parameter_stats_frame(
    frame: pd.DataFrame,
    *,
    zero_tol: float = 0.0,
    saturation_abs: float | None = None,
    near_rail_fraction: float = 0.95,
    underflow_abs: float | None = None,
) -> pd.DataFrame:
    rows = []
    for index, row in frame.iterrows():
        values = row.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            _parameter_stats_row(
                values,
                zero_tol=zero_tol,
                saturation_abs=saturation_abs,
                near_rail_fraction=near_rail_fraction,
                underflow_abs=underflow_abs,
            )
        )

    stats = pd.DataFrame(rows, index=frame.index, columns=PARAMETER_STAT_COLUMNS)
    return stats


def _parameter_stats_row(
    values: np.ndarray,
    *,
    zero_tol: float,
    saturation_abs: float | None,
    near_rail_fraction: float,
    underflow_abs: float | None,
) -> dict[str, float]:
    if values.size == 0:
        return {column: np.nan for column in PARAMETER_STAT_COLUMNS}

    abs_values = np.abs(values)
    stats = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values)),
        'q05': float(np.quantile(values, 0.05)),
        'q25': float(np.quantile(values, 0.25)),
        'q75': float(np.quantile(values, 0.75)),
        'q95': float(np.quantile(values, 0.95)),
        'norm_l2': float(np.linalg.norm(values, ord=2)),
        'norm_inf': float(np.max(abs_values)),
        'sparsity_fraction': float(np.mean(abs_values <= zero_tol)),
        'saturation_fraction': np.nan,
        'near_rail_fraction': np.nan,
        'underflow_fraction': np.nan,
    }

    if saturation_abs is not None:
        stats['saturation_fraction'] = float(np.mean(abs_values >= saturation_abs))
        stats['near_rail_fraction'] = float(np.mean(abs_values >= near_rail_fraction * saturation_abs))

    if underflow_abs is not None:
        stats['underflow_fraction'] = float(np.mean((abs_values > 0) & (abs_values < underflow_abs)))

    return stats


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
