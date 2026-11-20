import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from pathlib import Path
import numpy as np
import pandas as pd
import re


# All plots use this order. Baseline methods come first, then tree ensemble
# methods, then foundation models. QDA and QUADRO are baseline quadratic
# classifiers, not tree models.
BASELINE_METHODS = ['L-SLR', 'SLR', 'QDA', 'Quadro']
TREE_METHODS = ['RandomForest', 'XGBoost', 'CatBoost']
TABPFN_METHODS = ['TabPFN_v25', 'TabPFN_v26', 'TabPFN_v3', 'TabPFN_v35']
FOUNDATION_METHODS = TABPFN_METHODS + ['EXAONE', 'TabICL-V1', 'TabICL-V2']
# Kept for callers that still use the old two-group name.
CLASSICAL_METHODS = BASELINE_METHODS + TREE_METHODS
METHOD_ORDER = BASELINE_METHODS + TREE_METHODS + FOUNDATION_METHODS

# One canonical spelling for the three method families. Every plot title,
# legend title, table group and heatmap header uses these full names.
GROUP_LABELS = {
    'baseline': 'Baseline methods',
    'tree': 'Tree ensemble methods',
    'foundation': 'Foundation models',
}

# Metrics where a larger value is better. Every other metric is lower-is-better.
HIGHER_IS_BETTER = {'AUC'}

# Group decorations: title, member methods, background, border.
GROUPS = [
    (GROUP_LABELS['baseline'], BASELINE_METHODS, '#faf7f3', '#9F4F0B'),
    (GROUP_LABELS['tree'], TREE_METHODS, '#f3faf2', '#3F6B3A'),
    (GROUP_LABELS['foundation'], FOUNDATION_METHODS, '#f3f7fb', '#1B5E9A'),
]

COLOR_SCHEME = {
    'L-SLR':        '#D27C20',
    'SLR':          '#9F4F0B',
    'QDA':          '#D95F59',
    'Quadro':       '#9E292F',
    'RandomForest': '#6F9E4B',
    'XGBoost':      '#4F7F35',
    'CatBoost':     '#2F5E22',
    'TabPFN_v25':   '#5F95BF',
    'TabPFN_v26':   '#397FBA',
    'TabPFN_v3':    '#1B5E9A',
    'TabPFN_v35':   '#08386B',
    # EXAONE keeps the former TabICL purple range; TabICL uses the former
    # EXAONE teal range. The swap is global so every plot agrees, and it keeps
    # the two foundation-family half violins in the forest plot distinct.
    'EXAONE':       '#7564A8',
    'TabICL-V1':    '#168B8A',
    'TabICL-V2':    '#0E6C6B',
}

# Line styles distinguish related methods when their curves overlap.
LINE_STYLES = {
    'L-SLR':        '-',
    'SLR':          '--',
    'RandomForest': '-',
    'XGBoost':      '--',
    'CatBoost':     ':',
    'QDA':          '-',
    'Quadro':       '--',
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
    'Quadro': 'QUADRO',
    'TabICL-V1': 'TabICL v1', 'TabICL-V2': 'TabICL v2',
    'TabPFN_v25': 'TabPFN 2.5', 'TabPFN_v26': 'TabPFN 2.6',
    'TabPFN_v3': 'TabPFN 3', 'TabPFN_v35': 'TabPFN 3.5',
}

# Each group legend is wrapped so it is at most this many rows tall. Keeping
# the legends short stops the three boxes from running past the figure edges.
LEGEND_MAX_ROWS = 3

# Fallback colours for comparison methods that are not in the main colour
# scheme. Known methods always reuse their exact colour from every other plot.
COMPARISON_COLORS = ['#1B5E9A', '#D95F59']


def _comparison_color(method, index=None):
    """Colour one comparison by its method, with a stable fallback.

    Known methods reuse their colour from the violin/box plots, ROC curves and
    runtime bars. An unknown method uses a fallback index derived from the
    method order, never the rank it happens to hold in one setting.
    """
    if method in COLOR_SCHEME:
        return COLOR_SCHEME[method]
    if index is None:
        index = (FOUNDATION_METHODS.index(method)
                 if method in FOUNDATION_METHODS else 0)
    return COMPARISON_COLORS[index % len(COMPARISON_COLORS)]


def _forest_color(method, index=None):
    """Use the same per-method colour as the violin/box plots.

    The forest plot used to collapse all TabPFN methods to one blue and all
    TabICL methods to one teal. That worked for the two-model headline view but
    makes the all-foundations view ambiguous, so each method now keeps its own
    colour everywhere in the report.
    """
    return _comparison_color(method, index)


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


def _bayes_auc_reference(ax, setting, metric, bayes_auc, show):
    """Draw the population AUC reference; its label lives in the legend.

    The reference is a population value, not a fitted model. Keeping its label
    out of the plotting area leaves the distributions uncluttered.
    """
    if str(metric).upper() != 'AUC' or not show or not bayes_auc:
        return
    if setting not in bayes_auc:
        return

    reference = bayes_auc[setting]
    value = reference['auc']
    prefix = 'Bayes AUC ≈' if reference.get('mc_se', 0) > 0 else 'Bayes AUC ='
    ax.axhline(value, color='black', linestyle='--', linewidth=1.3,
               label=f'{prefix} {value:.3f}', zorder=4)


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
        if save_path is not None:
            plt.close(fig)


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
        _bayes_auc_reference(ax, setting, metric, bayes_auc, show_bayes_auc)
        reference_lines = [line for line in ax.lines
                           if line.get_label().startswith('Bayes AUC')]
        if reference_lines:
            ax.legend(handles=reference_lines, loc='lower right', fontsize=9,
                      frameon=True, framealpha=1)

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
        if save_path is not None:
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Violin plots (AUC only)
# ─────────────────────────────────────────────────────────────────────
def plot_violin_grid(errors_dict, settings_to_show, metric='AUC', labels=None,
                     dir=None, figsize=(12, 10), save_path=None, *, group_frames=False,
                     bayes_auc=None, show_bayes_auc=True):
    """Violin-and-box comparison for any scalar metric, one figure per setting.

    Each method gets a full vertical violin in its group colour with a compact
    black boxplot inside it, following the box+violin idea without adding raw
    scatter points or a separate mean line. The line inside the inner box is the
    median. Works for AUC, Brier and ClassError; the Bayes AUC reference is
    drawn only for AUC.

    With several settings, ``save_path`` gets the setting name added before the
    suffix (``auc_violin.png`` -> ``auc_violin_setting1.png``).
    """
    multiple = len(settings_to_show) > 1

    for setting in settings_to_show:
        df = errors_dict[setting]
        fig, ax = plt.subplots(figsize=figsize)

        data_to_plot, colors, shown_methods = [], [], []
        for model in METHOD_ORDER:
            column = f'{model}_{metric}'
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors='coerce')
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            data_to_plot.append(values.to_numpy())
            colors.append(COLOR_SCHEME[model])
            shown_methods.append(model)

        if not shown_methods:
            plt.close(fig)
            continue

        positions = _grouped_axis(ax, shown_methods,
                                  style='frames' if group_frames else 'shaded')
        parts = ax.violinplot(
            data_to_plot, positions=positions, widths=0.6, points=100,
            showmeans=False, showmedians=False, showextrema=False,
        )
        for body, color in zip(parts['bodies'], colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.7)
            body.set_zorder(2)

        # A narrow, unfilled boxplot inside each violin adds the quartiles and
        # median without hiding the density or adding a scatter layer.
        box = ax.boxplot(
            data_to_plot, positions=positions, widths=0.22, patch_artist=True,
            showmeans=False, showfliers=False, manage_ticks=False,
            boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.2),
            whiskerprops=dict(color='black', linewidth=1.0),
            capprops=dict(color='black', linewidth=1.0),
            medianprops=dict(color='black', linewidth=1.2),
        )
        for patch in box['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor('black')
            patch.set_zorder(3)
        for key in ('whiskers', 'caps', 'medians'):
            for item in box[key]:
                item.set_zorder(4)

        _bayes_auc_reference(ax, setting, metric, bayes_auc, show_bayes_auc)

        ax.set_xticks(positions)
        ax.set_xticklabels([DISPLAY_NAMES.get(model, model) for model in shown_methods],
                           rotation=25, ha='right', fontsize=11)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(_title(setting, labels), fontsize=12, pad=14 if group_frames else 30)
        ax.grid(axis='y', alpha=0.3)

        handles = [line for line in ax.lines
                   if line.get_label().startswith('Bayes AUC')]
        if handles:
            ax.legend(handles=handles, loc='lower right', fontsize=11,
                      bbox_to_anchor=(0.98, 0.03), borderaxespad=0.0,
                      frameon=True, framealpha=1)

        if group_frames:
            _pad_group_frames(ax)

        fig.tight_layout()
        _save_figure(fig, dir, _setting_filename(save_path, setting, multiple))
        if plt.get_backend().lower() != 'agg':
            plt.show()
        if save_path is not None:
            plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Runtime barplot (stacked by phase)
# ─────────────────────────────────────────────────────────────────────
# Each bar is split into the runner's timing phases. "Other" is the part of
# the total wall-clock time not covered by the four measured phases (for
# example the ROC/Brier computation after prediction). Use a colour-blind safe
# palette with a distinct hue per phase; shades of one blue are hard to tell
# apart once the bars are stacked.
TIME_PHASES = [
    ('SetupTime', 'Setup', '#56B4E9'),
    ('TuneTime', 'Tune', '#E69F00'),
    ('FitTime', 'Fit', '#009E73'),
    ('PredictTime', 'Predict', '#CC79A7'),
    ('Other', 'Other', '#999999'),
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
        ax.text(x[i], total, f'{total:.0f}s', ha='center', va='bottom',
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
    if save_path is not None:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Forest plot of the best-vs-best (foundation minus tree) effect
# ─────────────────────────────────────────────────────────────────────
# Short names for the forest y-axis. The full setting descriptions stay in the
# metadata and other plot titles; only these compact labels go on the axis.
# Each is at most three words so the nine sensitivity rows stay readable.
SHORT_SETTING_LABELS = {
    'simulation_setting1': 'Low-dim Gaussian',
    'simulation_setting2': 'Quadratic signal',
    'simulation_setting3': 'Linear + quadratic',
    'simulation_setting4': 'Class imbalance',
    'simulation_setting5': 'Heavy-tailed t',
    'simulation_setting6': 'Exp covariance decay',
    'simulation_setting7+': 'Heavy-tail mix (p=40)',
    'simulation_setting7': 'Sparse Gaussian (p=40)',
    'simulation_setting8': 'Sparse Gaussian (p=100)',
}


def _short_forest_label(setting, labels=None):
    """Compact y-axis name for a simulation setting in the forest plot."""
    key = str(setting)
    if key.endswith('_sensitivity'):
        key = key[:-len('_sensitivity')]
    short_id = re.fullmatch(r'S(\d+\+?)', key)
    if short_id:
        key = f'simulation_setting{short_id.group(1)}'
    if key in SHORT_SETTING_LABELS:
        return SHORT_SETTING_LABELS[key]

    # Real-data and unknown settings keep their supplied label. Drop only a
    # trailing sensitivity marker, which now belongs in the plot title.
    label = _title(setting, labels)
    return re.sub(r'\s*\(sensitivity\)\s*$', '', label, flags=re.IGNORECASE)


def _half_violin_side(method, index=0):
    """Put TabPFN above the setting row and TabICL below it.

    The forest y-axis is inverted, so ``low`` (smaller y) appears visually above
    the setting label and ``high`` appears below it. Unknown methods alternate
    to keep two half violins from overlapping.
    """
    if method in TABPFN_METHODS:
        return 'low'
    if str(method).startswith('TabICL'):
        return 'high'
    return 'low' if index % 2 == 0 else 'high'


def plot_effect_forest(effect_frame, metrics=('AUC', 'Brier'), labels=None,
                       dir=None, figsize=None, save_path=None, effect_samples=None,
                       n_foundation=2):
    """Compare top foundation models with the best tree, one plot per metric.

    ``n_foundation`` controls how many top-performing foundation models (ranked
    per setting/metric) are compared against the single best tree: ``2`` (default)
    shows the best two, ``3`` the top three, and ``0`` shows every available
    foundation model. To build effect data for a custom count, pass
    ``n_foundation`` to ``build_effect_table`` / ``build_effect_samples`` and
    provide the results here. For convenience, ``effect_frame`` may also be an
    ``errors_dict`` mapping setting IDs to result DataFrames; the table and
    paired samples are then built automatically for ``n_foundation``.

    When paired samples are available, each comparison is drawn as an upward
    half violin of foundation-minus-tree differences with a black line at the
    mean. If samples are unavailable, a mean/SD errorbar view is used. The left
    axis names each setting; the right axis has one labelled row per selected
    foundation comparison and names the best tree ensemble for that setting.
    If different settings select different top models, only that setting's
    selected models get rows, so no empty rows are added. Each method uses the
    same colour as in the violin/box plots. The legend only identifies the mean
    paired effect. A dashed line at 0 marks no difference. Sensitivity reports
    add ``(Sensitivity)`` to the title and use short setting names on the left
    axis so the description fits in two or three words.

    With several metrics, ``save_path`` gets the metric name added before the
    suffix (``forest.png`` -> ``forest_AUC.png``).
    """
    # Convenience: build effect table/samples directly from results.
    if isinstance(effect_frame, dict) and not hasattr(effect_frame, 'xs'):
        errors_dict = effect_frame
        try:
            from .table_functions import build_effect_table as _build_effect_table
            from .table_functions import build_effect_samples as _build_effect_samples
        except ImportError:
            from medfm.table_functions import build_effect_table as _build_effect_table
            from medfm.table_functions import build_effect_samples as _build_effect_samples
        if isinstance(metrics, str):
            metrics = [metrics]
        effect_obj = _build_effect_table(errors_dict, metrics=metrics, filename=None,
                                         n_foundation=n_foundation)
        if effect_obj is None:
            return
        effect_samples = _build_effect_samples(errors_dict, metrics=metrics,
                                               n_foundation=n_foundation)
        effect_frame = getattr(effect_obj, 'data', effect_obj)

    if effect_frame is None or getattr(effect_frame, 'empty', True):
        return
    # Accept a Styler from build_effect_table as well as a raw DataFrame.
    effect_frame = getattr(effect_frame, 'data', effect_frame)
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

        work = frame.copy()
        if 'Foundation rank' not in work.columns:
            work['Foundation rank'] = work.groupby(level=0).cumcount() + 1
        settings = list(dict.fromkeys(work.index))

        records = []
        for setting, row in work.iterrows():
            effect = pd.to_numeric(row.get('Mean effect'), errors='coerce')
            if not np.isfinite(effect):
                continue
            sd = pd.to_numeric(row.get('SD effect'), errors='coerce')
            rank = pd.to_numeric(row['Foundation rank'], errors='coerce')
            records.append({
                'setting': setting,
                'method': str(row['Foundation method']),
                'tree': str(row.get('Tree method', '')),
                'rank': int(rank) if np.isfinite(rank) else 0,
                'effect': float(effect),
                'sd': float(sd) if np.isfinite(sd) else 0.0,
            })
        if not records:
            continue

        # Each setting can have a different best tree and different top
        # foundation models. Keep only the methods actually selected in that
        # setting, but order them like the violin/box plots.
        def _ordered_methods(names):
            ordered = [m for m in FOUNDATION_METHODS if m in names]
            ordered.extend(sorted(set(names).difference(ordered)))
            return ordered

        methods_by_setting = {}
        for record in records:
            methods_by_setting.setdefault(record['setting'], set()).add(record['method'])
        methods_by_setting = {
            setting: _ordered_methods(names)
            for setting, names in methods_by_setting.items()
        }
        settings = [setting for setting in settings if setting in methods_by_setting]
        present = {m for names in methods_by_setting.values() for m in names}
        methods = _ordered_methods(present)
        color_index = {
            method: (FOUNDATION_METHODS.index(method)
                     if method in FOUNDATION_METHODS else len(FOUNDATION_METHODS) + index)
            for index, method in enumerate(methods)
        }
        selected_pairs = list(dict.fromkeys(
            (record['setting'], record['method']) for record in records
        ))

        # Collect paired differences; a partial set falls back to mean/SD view.
        sample_values = {}
        candidate = effect_samples
        if (candidate is not None and not getattr(candidate, 'empty', True)
                and {'Setting', 'Foundation method', 'Effect'}.issubset(candidate.columns)):
            if 'Metric' in candidate.columns:
                candidate = candidate[candidate['Metric'] == metric]
            for setting, method in selected_pairs:
                values = candidate.loc[
                    (candidate['Setting'] == setting)
                    & (candidate['Foundation method'] == method), 'Effect'
                ]
                values = pd.to_numeric(values, errors='coerce')
                values = values.replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) >= 2:
                    sample_values[(setting, method)] = values.to_numpy()
        use_violins = len(sample_values) == len(selected_pairs)

        # One row per selected (setting, foundation model) pair. The left axis
        # groups rows by setting; the right axis names each pair and its tree.
        best_tree = {}
        for record in records:
            if record['tree']:
                best_tree.setdefault(record['setting'], record['tree'])

        row_positions, row_labels, left_ticks, left_labels = [], [], [], []
        row_index = {}
        position = 0
        is_sensitivity = (
            any('sensitivity' in str(s).lower() for s in settings)
            or any('sensitivity' in str(_title(s, labels)).lower() for s in settings)
        )
        for setting in settings:
            start = position
            tree_name = best_tree.get(setting, 'best tree')
            tree_display = DISPLAY_NAMES.get(tree_name, tree_name)
            for method in methods_by_setting[setting]:
                row_index[(setting, method)] = position
                row_positions.append(position)
                row_labels.append(f'{DISPLAY_NAMES.get(method, method)} vs {tree_display}')
                position += 1
            left_ticks.append((start + position - 1) / 2)
            left_labels.append(_short_forest_label(setting, labels))

        fig_h = figsize[1] if figsize else max(3.0, 0.48 * len(row_positions) + 1.8)
        fig_w = figsize[0] if figsize else 11
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, zorder=1)

        if use_violins:
            for setting, method in selected_pairs:
                color = _forest_color(method, color_index[method])
                parts = ax.violinplot(
                    [sample_values[(setting, method)]],
                    positions=[row_index[(setting, method)]],
                    orientation='horizontal', side='low', widths=0.75, points=100,
                    showmeans=True, showmedians=False, showextrema=False,
                )
                body = parts['bodies'][0]
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.75)
                body.set_zorder(2)
                parts['cmeans'].set_color('black')
                parts['cmeans'].set_linewidth(2.0)
                parts['cmeans'].set_zorder(4)
        else:
            for record in records:
                color = _forest_color(record['method'], color_index[record['method']])
                ax.errorbar(
                    [record['effect']],
                    [row_index[(record['setting'], record['method'])]],
                    xerr=[record['sd']], fmt='o', color=color, ecolor=color,
                    elinewidth=1.4, capsize=4, markersize=7, markeredgecolor='black',
                    markeredgewidth=0.8, zorder=3,
                )

        # The legend only identifies the black mean marker; method names are on
        # the right axis next to their own row.
        legend_handles = [Line2D(
            [0], [0], color='black', linewidth=2.0 if use_violins else 1.4,
            marker='o' if not use_violins else None, markersize=7 if not use_violins else 6,
            label='Mean paired effect',
        )]

        ax.set_yticks(left_ticks)
        ax.set_yticklabels(left_labels, fontsize=11)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(row_positions)
        ax2.set_yticklabels(row_labels, fontsize=10)
        ax2.set_ylabel('Foundation model (vs best tree)', fontsize=11)
        ax2.margins(y=0.08)
        ax.set_xlabel(f'Paired effect on {metric} (foundation - tree)', fontsize=12)
        title = f'Paired {metric} effect for Foundation models versus best tree ensemble'
        if is_sensitivity:
            title += ' (Sensitivity)'
        ax.set_title(title, fontsize=13, pad=12)
        ax.grid(axis='x', alpha=0.3)
        ax.margins(x=0.25, y=0.08)
        ax.invert_yaxis()
        ax2.set_ylim(ax.get_ylim())
        legend_loc = {'AUC': 'upper right', 'BRIER': 'upper left'}.get(metric.upper(), 'best')
        ax2.legend(handles=legend_handles, loc=legend_loc, fontsize=9,
                   frameon=True, framealpha=1)

        fig.tight_layout()

        name = base
        if base is not None:
            name = f"{base.stem}_{metric}{base.suffix}"
        _save_figure(fig, dir, name)
        if plt.get_backend().lower() != 'agg':
            plt.show()
        if save_path is not None:
            plt.close(fig)


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
    Dark vertical borders split the baseline, tree and foundation columns, and a
    labelled strip above the heatmap names each of those sections.

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

        # Report tables print display names (e.g. QUADRO for Quadro), so accept
        # either spelling when reading a saved gain table.
        columns = {m: m if m in frame.columns else DISPLAY_NAMES.get(m, m)
                   for m in METHOD_ORDER}
        methods = [m for m, column in columns.items() if column in frame.columns]
        if not methods:
            continue
        values = (frame[[columns[m] for m in methods]]
                  .apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float))

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
            figsize=figsize or (0.75 * len(methods) + 3.5, 0.55 * len(frame) + 3.2))
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

        # Mark the baseline/tree-ensemble/foundation columns with dark borders
        # and a thin strip above the heatmap, even when a method is missing.
        # The full family names are kept here as elsewhere in the report.
        sections = []
        for title, members, background, border in GROUPS:
            indices = [index for index, method in enumerate(methods) if method in members]
            if indices:
                sections.append((title, min(indices), max(indices),
                                 background, border))
        for _, left, _, _, _ in sections[1:]:
            ax.axvline(left - 0.5, color='#222222', linewidth=1.5, zorder=5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label('Relative improvement per setting (labels are actual gains)',
                       fontsize=9)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['least', 'largest'])
        fig.tight_layout(rect=[0, 0, 1, 0.86])

        # Place the header after tight_layout so it stays aligned with the
        # columns in the final, possibly resized, heatmap axes.
        x0, y0, width, height = ax.get_position().bounds
        # A thin category strip sitting directly on top of the first setting row.
        header = fig.add_axes([x0, y0 + height, width, 0.16 * height / 3])
        header.set_xlim(ax.get_xlim())
        header.set_ylim(0, 1)
        header.axis('off')
        for label, left, right, background, border in sections:
            header.add_patch(Rectangle(
                (left - 0.5, 0), right - left + 1, 1,
                facecolor=background, edgecolor=border, linewidth=1.0,
            ))
            header.text((left + right) / 2, 0.5, label, ha='center', va='center',
                        fontsize=8, fontweight='bold', color='#222222')
        header.set_title(f'{sensitivity_note} — {metric}', fontsize=13, pad=6)

        name = base
        if base is not None:
            name = f"{base.stem}_{metric}{base.suffix}"
        _save_figure(fig, dir, name)
        if plt.get_backend().lower() != 'agg':
            plt.show()
        if save_path is not None:
            plt.close(fig)
