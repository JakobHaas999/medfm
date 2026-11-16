import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import numpy as np
import pandas as pd


# All plots use this order. Baseline methods come first, then tree ensembles,
# then foundation models. QDA is the baseline quadratic classifier.
BASELINE_METHODS = ['L-SLR', 'SLR', 'QDA']
TREE_METHODS = ['RandomForest', 'XGBoost', 'CatBoost']
TABPFN_METHODS = ['TabPFN_v25', 'TabPFN_v26', 'TabPFN_v3', 'TabPFN_v35']
FOUNDATION_METHODS = TABPFN_METHODS + ['EXAONE', 'TabICL-V1', 'TabICL-V2']
# Kept for callers that still use the old two-group name.
CLASSICAL_METHODS = BASELINE_METHODS + TREE_METHODS
METHOD_ORDER = BASELINE_METHODS + TREE_METHODS + FOUNDATION_METHODS

# Metrics where a larger value is better. Every other metric is lower-is-better.
HIGHER_IS_BETTER = {'AUC'}

# Group decorations: title, member methods, background, border.
GROUPS = [
    ('Baseline methods', BASELINE_METHODS, '#faf7f3', '#9F4F0B'),
    ('Tree ensembles', TREE_METHODS, '#f3faf2', '#3F6B3A'),
    ('Foundation models', FOUNDATION_METHODS, '#f3f7fb', '#1B5E9A'),
]

COLOR_SCHEME = {
    'L-SLR':        '#D27C20',
    'SLR':          '#9F4F0B',
    'QDA':          '#D95F59',
    'RandomForest': '#6F9E4B',
    'XGBoost':      '#4F7F35',
    'CatBoost':     '#2F5E22',
    'TabPFN_v25':   '#5F95BF',
    'TabPFN_v26':   '#397FBA',
    'TabPFN_v3':    '#1B5E9A',
    'TabPFN_v35':   '#08386B',
    'EXAONE':       '#168B8A',
    'TabICL-V1':    '#9B8AC4',
    'TabICL-V2':    '#7564A8',
}

# Line styles distinguish related methods when their curves overlap.
LINE_STYLES = {
    'L-SLR':        '-',
    'SLR':          '--',
    'RandomForest': '-',
    'XGBoost':      '--',
    'CatBoost':     ':',
    'QDA':          '-',
    'TabPFN_v25':   '-',
    'TabPFN_v26':   '--',
    'TabPFN_v3':    '-.',
    'TabPFN_v35':   ':',
    'EXAONE':       '-',
    'TabICL-V1':    '-',
    'TabICL-V2':    '--',
}

# Only display labels change; saved result column names stay the same.
DISPLAY_NAMES = {
    'TabICL-V1': 'TabICL v1', 'TabICL-V2': 'TabICL v2',
    'TabPFN_v25': 'TabPFN 2.5', 'TabPFN_v26': 'TabPFN 2.6',
    'TabPFN_v3': 'TabPFN 3', 'TabPFN_v35': 'TabPFN 3.5',
}

# Each group legend is wrapped so it is at most this many rows tall. Keeping
# the legends short stops the three boxes from running past the figure edges.
LEGEND_MAX_ROWS = 3


def _setting_filename(save_path, setting, multiple):
    """Insert the setting name before the suffix when several settings are drawn."""
    if not save_path or not multiple:
        return save_path
    path = Path(save_path)
    return f"{path.stem}_{str(setting).replace(' ', '_')}{path.suffix}"


def _save_figure(fig, dir, save_path):
    """Save one figure into the caller's results directory.

    Reports are saved next to the results they describe, so ``dir`` is required
    whenever ``save_path`` is given. Nothing is written when ``save_path`` is
    None, which lets the caller display the figure only.
    """
    if save_path is None:
        return
    if dir is None:
        raise ValueError(f"dir is required to save {save_path}")
    Path(dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(dir) / save_path, dpi=300, bbox_inches='tight',
                facecolor=fig.get_facecolor())


def _grouped_axis(ax, methods, *, style='shaded'):
    """Separate the method groups with a gap and optional decoration."""
    positions = np.arange(1, len(methods) + 1, dtype=float)
    group_index = {}
    for index, (_, members, _, _) in enumerate(GROUPS):
        for method in members:
            group_index[method] = index
    for index in range(1, len(methods)):
        previous = group_index.get(methods[index - 1])
        current = group_index.get(methods[index])
        if previous is not None and current is not None and previous != current:
            positions[index:] += 0.8

    outer_top = 0.96
    height_points = max(ax.get_window_extent().height * 72 / ax.figure.dpi, 1)
    outer_bottom = max(0.025, 10 / height_points)
    for title, members, background, border in GROUPS:
        group_positions = [x for x, method in zip(positions, methods) if method in members]
        if not group_positions:
            continue
        left, right = min(group_positions) - 0.45, max(group_positions) + 0.45
        if style == 'frames':
            ax.add_patch(Rectangle(
                (left, outer_bottom), right - left, outer_top - outer_bottom,
                transform=ax.get_xaxis_transform(), fill=False,
                edgecolor=border, linewidth=1.1, clip_on=True, zorder=2,
            ))
            ax.text((left + right) / 2, outer_bottom, title,
                    transform=ax.get_xaxis_transform(), ha='center', va='center',
                    fontsize=9, fontweight='bold', color='#222222', clip_on=True,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5), zorder=5)
        else:
            ax.axvspan(left, right, color=background, zorder=0)
            ax.annotate(title, xy=((left + right) / 2, 1), xycoords=ax.get_xaxis_transform(),
                        xytext=(0, 8), textcoords='offset points', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', annotation_clip=False)
    return positions


def _pad_group_frames(ax):
    """Keep plotted data inside group frames, including whiskers and outliers."""
    frames = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    if not frames:
        return
    bottom = max(frame.get_y() for frame in frames) + 0.015
    top = min(frame.get_y() + frame.get_height() for frame in frames) - 0.015
    data_low, data_high = ax.dataLim.intervaly
    low, high = ax.get_ylim()
    if not np.all(np.isfinite([data_low, data_high])):
        return
    # Expand the current limits enough to fit data between the frame borders.
    span = max(high - low, (data_high - data_low) / (top - bottom),
               (data_high - low) / top, (high - data_low) / (1 - bottom))
    low = min(low, data_low - bottom * span)
    ax.set_ylim(low, low + span)


def _grouped_legend(fig, handles):
    """Center reference and method groups and return the top of the legend area.

    Each group is wrapped into at most ``LEGEND_MAX_ROWS`` rows. A shorter
    legend is narrower, so the three boxes stay inside the saved figure.
    """
    sections = [('Reference curves', ['Bayes ROC', 'Random guess (AUC = 0.5)'])]
    sections += [(title, members) for title, members, _, _ in GROUPS]
    legends = []
    bottom = 0.01
    for title, methods in sections:
        available = [method for method in methods if method in handles]
        if not available:
            continue
        ncol = max(1, -(-len(available) // LEGEND_MAX_ROWS))
        legend = fig.legend(
            handles=[handles[method] for method in available],
            labels=[DISPLAY_NAMES.get(method, method) for method in available],
            title=title, loc='lower center', bbox_to_anchor=(0.5, bottom),
            ncol=min(ncol, len(available)), fontsize=9, title_fontsize=10,
            frameon=True, fancybox=False, edgecolor='#999999', handlelength=3,
            facecolor='white', framealpha=1, borderpad=0.7,
        )
        legend.get_title().set_fontweight('bold')
        legends.append(legend)

    if legends:
        # Measure the actual widths so missing methods still leave a centered legend.
        fig.draw_without_rendering()
        widths = [legend.get_window_extent().transformed(fig.transFigure.inverted()).width
                  for legend in legends]
        gap = 0.025
        left = (1 - sum(widths) - gap * (len(legends) - 1)) / 2
        for legend, width in zip(legends, widths):
            legend.set_bbox_to_anchor((left + width / 2, bottom))
            left += width + gap

    fig.draw_without_rendering()
    return max((legend.get_window_extent().transformed(fig.transFigure.inverted()).y1
                for legend in legends), default=0)


def _title(setting, labels):
    """Use the supplied setting label, or the setting ID when no label is provided."""
    return labels.get(str(setting), str(setting)) if labels else str(setting)


# ─────────────────────────────────────────────────────────────────────
# ROC curves
# ─────────────────────────────────────────────────────────────────────
def plot_roc_curves_grid(errors_dict, settings_to_show, labels=None,
                         dir=None, figsize=(12, 10), save_path=None, *,
                         bayes_auc=None, show_bayes_roc=True):
    """Draw the average ROC curve for each model, one saved figure per setting.

    Interpolate each repetition onto a common false-positive-rate grid before averaging.
    Supply distribution-based references as bayes_auc to add a dashed Bayes ROC.
    show_bayes_roc=False hides it. Monte Carlo curves estimate the population
    optimum; individual finite-test-sample curves can cross this reference.

    Each setting gets its own figure so the curves and the grouped legend stay
    readable. With several settings, ``save_path`` gets the setting name added
    before the suffix (``roc.png`` -> ``roc_setting1.png``).
    """
    base_fpr = np.linspace(0, 1, 100)
    multiple = len(settings_to_show) > 1

    for setting in settings_to_show:
        df = errors_dict[setting]
        fig, ax = plt.subplots(figsize=figsize)
        legend_handles = {}

        for method in METHOD_ORDER:
            fpr_col = f'{method}_FPR'
            tpr_col = f'{method}_TPR'
            if fpr_col not in df.columns or tpr_col not in df.columns:
                continue

            tprs_interp = []
            for fpr, tpr in zip(df[fpr_col].tolist(), df[tpr_col].tolist()):
                if len(fpr) == 0 or len(tpr) == 0:
                    continue
                tprs_interp.append(np.interp(base_fpr, fpr, tpr))
            if not tprs_interp:
                continue

            mean_tpr = np.mean(tprs_interp, axis=0)
            mean_auc = np.mean(df[f'{method}_AUC'])

            line, = ax.plot(base_fpr, mean_tpr, linewidth=2,
                            color=COLOR_SCHEME[method], linestyle=LINE_STYLES[method],
                            label=f"{method} (AUC={mean_auc:.3f})")
            legend_handles.setdefault(method, line)

        if show_bayes_roc and bayes_auc and setting in bayes_auc:
            reference = bayes_auc[setting]
            if all(key in reference for key in ('fpr', 'tpr')):
                line, = ax.plot(reference['fpr'], reference['tpr'], color='black',
                                linestyle='--', linewidth=2, label='Bayes ROC', zorder=4)
                legend_handles.setdefault('Bayes ROC', line)

        line, = ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1,
                        label='Random guess (AUC = 0.5)')
        legend_handles.setdefault('Random guess (AUC = 0.5)', line)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12)
        ax.grid(alpha=0.3)

        legend_top = _grouped_legend(fig, legend_handles)
        fig.suptitle('Average ROC Curves', fontsize=14, y=0.99)
        fig.tight_layout(rect=[0, legend_top + 0.02, 1, 0.97])

        _save_figure(fig, dir, _setting_filename(save_path, setting, multiple))
        if plt.get_backend().lower() != 'agg':
            plt.show()


# ─────────────────────────────────────────────────────────────────────
# Boxplots (ClassError / AUC / Brier / any scalar metric)
# ─────────────────────────────────────────────────────────────────────
def plot_boxplots_grid(errors_dict, settings_to_show, metric='ClassError', labels=None,
                       dir=None, figsize=(12, 10), save_path=None, *, group_frames=False,
                       bayes_auc=None, show_bayes_auc=True):
    """Compare a scalar measure across repetitions, one saved figure per setting.

    Shade baseline, tree, and foundation groups with titles above the plot.
    Use group_frames=True to display group frames inside the plot instead.
    Pass a setting-to-reference dictionary as bayes_auc for AUC plots.
    The reference line is on by default; show_bayes_auc=False hides it.

    With several settings, ``save_path`` gets the setting name added before the
    suffix (``auc_box.png`` -> ``auc_box_setting1.png``).
    """
    multiple = len(settings_to_show) > 1

    for setting in settings_to_show:
        df = errors_dict[setting]
        fig, ax = plt.subplots(figsize=figsize)

        data_to_plot, colors, shown_methods = [], [], []
        for model in METHOD_ORDER:
            col_name = f'{model}_{metric}'
            if col_name not in df.columns:
                continue
            data_to_plot.append(df[col_name].dropna())
            colors.append(COLOR_SCHEME[model])
            shown_methods.append(model)

        if not shown_methods:
            plt.close(fig)
            continue

        positions = _grouped_axis(ax, shown_methods, style='frames' if group_frames else 'shaded')
        xtick_labels = [DISPLAY_NAMES.get(model, model) for model in shown_methods]
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, showmeans=False,
                        medianprops=dict(linewidth=1.5, color='black'))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # A population reference, not another fitted model or another box.
        if metric.upper() == 'AUC' and show_bayes_auc and bayes_auc and setting in bayes_auc:
            reference = bayes_auc[setting]
            value = reference['auc']
            prefix = 'Bayes AUC ≈' if reference.get('mc_se', 0) > 0 else 'Bayes AUC ='
            ax.axhline(value, color='black', linestyle='--', linewidth=1.3,
                       label=f'{prefix} {value:.3f}', zorder=4)
            # Place the label just above the line, aligned to the right edge.
            ax.annotate(f'{prefix} {value:.3f}', xy=(0.99, value),
                        xycoords=ax.get_yaxis_transform(), xytext=(0, 9),
                        textcoords='offset points', ha='right', va='bottom',
                        fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                        zorder=5)

        ax.set_xticks(positions)
        ax.set_xticklabels(xtick_labels, rotation=25, ha='right', fontsize=11)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12, pad=14 if group_frames else 30)
        ax.grid(axis='y', alpha=0.3)

        if metric.upper() in ('MSE', 'CLASSERROR'):
            ax.set_ylim(0.0, 0.6)
        if group_frames:
            _pad_group_frames(ax)

        fig.tight_layout()
        _save_figure(fig, dir, _setting_filename(save_path, setting, multiple))
        if plt.get_backend().lower() != 'agg':
            plt.show()


# ─────────────────────────────────────────────────────────────────────
# Runtime barplot (stacked by phase)
# ─────────────────────────────────────────────────────────────────────
# Each bar is split into the runner's timing phases. "Other" is the part of
# the total wall-clock time not covered by the four measured phases (for
# example the ROC/Brier computation after prediction).
TIME_PHASES = [
    ('SetupTime', 'Setup', '#c9d9ea'),
    ('TuneTime', 'Tune', '#8fb3d9'),
    ('FitTime', 'Fit', '#4f86c6'),
    ('PredictTime', 'Predict', '#1f4e79'),
    ('Other', 'Other', '#b8b8b8'),
]
_TIME_KEYS = ['SetupTime', 'TuneTime', 'FitTime', 'PredictTime']


def plot_runtime_barplot_simple(errors_dict, settings_to_show, labels=None,
                                dir=None, figsize=(12, 7), save_path=None):
    """Stack each model's mean runtime into setup, tune, fit, predict, and other.

    Timings are pooled across the selected settings. One bar per method, with
    the measured phases stacked bottom to top and the total wall-clock time
    printed above each bar. This is one summary plot, not a grid.
    """
    methods = METHOD_ORDER
    pooled = {m: {'Time': [], **{key: [] for key in _TIME_KEYS}} for m in methods}
    for setting in settings_to_show:
        df = errors_dict[setting]
        for m in methods:
            for key in pooled[m]:
                col = f'{m}_{key}'
                if col in df.columns:
                    values = np.asarray(df[col], dtype=float)
                    pooled[m][key].extend(values[np.isfinite(values)].tolist())

    methods = [m for m in methods if pooled[m]['Time']]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=figsize)
    x = _grouped_axis(ax, methods)
    bottoms = np.zeros(len(methods))

    for key, label, color in TIME_PHASES:
        heights = np.array([
            max(np.mean(pooled[m]['Time']) - sum(np.mean(pooled[m][k]) for k in _TIME_KEYS), 0.0)
            if key == 'Other' else np.mean(pooled[m][key])
            for m in methods
        ])
        ax.bar(x, heights, bottom=bottoms, color=color, edgecolor='black',
               linewidth=0.6, alpha=0.95, label=label)
        bottoms += heights

    for i, total in enumerate(bottoms):
        ax.text(x[i], total, f'{total:.2f}s', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods],
                       rotation=25, ha='right', fontsize=11)
    ax.set_ylabel('Average runtime (seconds)', fontsize=13)
    ax.set_title('Mean Runtime per Method by Phase', fontsize=14, pad=30)
    ax.margins(y=0.18)
    ax.grid(axis='y', alpha=0.3, color='#b0b0b0')
    ax.legend(title='Phase', loc='upper right', fontsize=10, title_fontsize=10,
              frameon=True, framealpha=1)
    fig.tight_layout()

    _save_figure(fig, dir, save_path)
    if plt.get_backend().lower() != 'agg':
        plt.show()


# ─────────────────────────────────────────────────────────────────────
# Forest plot of the best-vs-best (foundation minus tree) effect
# ─────────────────────────────────────────────────────────────────────
def plot_effect_forest(effect_frame, metrics=('AUC', 'Brier'), labels=None,
                       dir=None, figsize=None, save_path=None):
    """Forest plot of the paired foundation-minus-tree effect, one figure per metric.

    ``effect_frame`` is the raw table from ``build_effect_table`` with a
    ``(Setting, Metric)`` index and columns ``Mean effect``, ``SD effect``,
    ``Tree method`` and ``Foundation method``. Each setting is one row: the
    point is the mean paired effect and the bar is plus/minus one SD across
    paired runs. A dashed line at 0 marks "no difference", so points to the
    right mean the foundation model scored higher. The left axis names the
    setting; the right axis names the subtracted pair, foundation minus tree, so
    the label direction matches the sign of the effect.

    With several metrics, ``save_path`` gets the metric name added before the
    suffix (``forest.png`` -> ``forest_AUC.png``).
    """
    if effect_frame is None or getattr(effect_frame, 'empty', True):
        return
    if isinstance(metrics, str):
        metrics = [metrics]
    base = Path(save_path) if save_path else None

    for metric in metrics:
        try:
            frame = effect_frame.xs(metric, level='Metric')
        except KeyError:
            continue
        if frame.empty:
            continue

        settings = list(frame.index)
        effects = frame['Mean effect'].to_numpy(dtype=float)
        sds = frame['SD effect'].to_numpy(dtype=float)
        y = np.arange(len(settings))

        fig, ax = plt.subplots(figsize=figsize or (10, max(3.0, 0.7 * len(settings) + 1.6)))
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, zorder=1)
        ax.errorbar(effects, y, xerr=sds, fmt='none', ecolor='#555555',
                    elinewidth=1.4, capsize=4, zorder=2)
        ax.scatter(effects, y, c=['#1B5E9A' if v >= 0 else '#9E292F' for v in effects],
                   s=90, edgecolors='black', linewidth=1.0, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels([_title(setting, labels) for setting in settings], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel(f'Mean effect on {metric} (foundation - tree)', fontsize=12)
        ax.set_title(f'Best tree vs best foundation — {metric}', fontsize=13, pad=12)
        ax.grid(axis='x', alpha=0.3)
        ax.margins(x=0.25, y=0.15)

        # The right axis names the subtracted pair, foundation minus tree, so
        # the sign of each point matches the printed difference.
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks(y)
        right.set_yticklabels(
            [f"{frame.loc[setting, 'Foundation method']} - {frame.loc[setting, 'Tree method']}"
             for setting in settings],
            fontsize=9, color='#333333')
        right.tick_params(axis='y', length=0)
        for spine in right.spines.values():
            spine.set_visible(False)
        fig.tight_layout()

        name = base
        if base is not None:
            name = f"{base.stem}_{metric}{base.suffix}"
        _save_figure(fig, dir, name)
        if plt.get_backend().lower() != 'agg':
            plt.show()


# ─────────────────────────────────────────────────────────────────────
# Sensitivity-minus-main gain heatmap
# ─────────────────────────────────────────────────────────────────────
def plot_gain_heatmap(gain_frame, metrics=('AUC', 'Brier'), labels=None,
                      dir=None, figsize=None, save_path=None,
                      sensitivity_note='Sample-size sensitivity (sensitivity − main)'):
    """Row-scaled heatmap of sensitivity-minus-main gain, one figure per metric.

    ``gain_frame`` is the raw table from ``build_gain_table`` with a
    ``(Setting, Metric)`` index and one column per method. Rows are settings and
    columns are methods; each cell is the sensitivity mean minus the main mean.
    The sensitivity run changes the training sample size, so the title names that
    comparison through ``sensitivity_note``.

    Colours are scaled within each row rather than across the whole table: the
    method with the smallest change (closest to zero) is blue and the method with
    the largest change is red. "Largest" means the largest improvement, which
    follows the metric direction: higher AUC is better, lower Brier and ClassError
    are better. So every row uses its full colour range and the biggest mover in
    a setting stands out even when settings differ in scale. Only the colours are
    relative to the row; every cell prints the actual signed sensitivity-minus-main
    gain, so the same value can take a different colour in a different setting.

    With several metrics, ``save_path`` gets the metric name added before the
    suffix (``gain.png`` -> ``gain_AUC.png``).
    """
    if gain_frame is None or getattr(gain_frame, 'empty', True):
        return
    if isinstance(metrics, str):
        metrics = [metrics]
    base = Path(save_path) if save_path else None

    for metric in metrics:
        try:
            frame = gain_frame.xs(metric, level='Metric')
        except KeyError:
            continue
        if frame.empty:
            continue

        methods = [m for m in METHOD_ORDER if m in frame.columns]
        values = frame[methods].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)

        # Higher is better for AUC; lower is better for Brier and ClassError.
        # A change in the wrong direction counts as no improvement.
        sign = 1.0 if metric.upper() in HIGHER_IS_BETTER else -1.0
        improvement = np.where(np.isfinite(values), np.clip(sign * values, 0, None), np.nan)

        # Scale each row by its own largest improvement so every row spans the
        # full blue-to-red range.
        row_max = np.nanmax(improvement, axis=1, keepdims=True)
        row_max = np.where(np.isfinite(row_max) & (row_max > 0), row_max, 1.0)
        scaled = improvement / row_max

        fig, ax = plt.subplots(
            figsize=figsize or (0.75 * len(methods) + 3.5, 0.55 * len(frame) + 2.2))
        im = ax.imshow(scaled, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')

        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in methods],
                           rotation=45, ha='right', fontsize=10)
        ax.set_yticks(np.arange(len(frame)))
        ax.set_yticklabels([_title(s, labels) for s in frame.index], fontsize=10)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not np.isfinite(value):
                    continue
                strong = scaled[i, j] < 0.35 or scaled[i, j] > 0.65
                ax.text(j, i, f'{value:+.3f}', ha='center', va='center', fontsize=8,
                        color='white' if strong else 'black')

        ax.set_title(f'{sensitivity_note} — {metric}', fontsize=13, pad=12)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Relative improvement per setting (labels are actual gains)',
                       fontsize=9)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['least', 'largest'])
        fig.tight_layout()

        name = base
        if base is not None:
            name = f"{base.stem}_{metric}{base.suffix}"
        _save_figure(fig, dir, name)
        if plt.get_backend().lower() != 'agg':
            plt.show()
