import os
import numpy as np
import pandas as pd

from .plot_functions import (
    BASELINE_METHODS,
    DISPLAY_NAMES,
    FOUNDATION_METHODS,
    GROUP_LABELS,
    TREE_METHODS,
)
from .utils import monte_carlo_se

METHODS = BASELINE_METHODS + TREE_METHODS + FOUNDATION_METHODS

# Reporting groups use the same full family names as the plots. Baseline
# methods are summarized but excluded from best-vs-best; the head-to-head
# comparison is tree ensemble methods versus foundation models.
GROUPS = [
    (GROUP_LABELS['baseline'], BASELINE_METHODS),
    (GROUP_LABELS['tree'], TREE_METHODS),
    (GROUP_LABELS['foundation'], FOUNDATION_METHODS),
]

# Scalar metrics only. FPR/TPR/Probas are list-valued — used by plots.
# ClassError is the observed misclassification rate, not squared error.
METRICS = ['AUC', 'ClassError', 'Brier', 'Time', 'SetupTime', 'TuneTime',
           'FitTime', 'PredictTime', 'CalSlope', 'CalIntercept']

HIGHER_IS_BETTER = {'AUC'}

# Result columns keep their stored keys, but report tables should spell method
# names the same way the plots do. Only case-only aliases are normalised here
# (currently Quadro -> QUADRO); other display names keep the stored key so
# column lookups by downstream code remain stable.
_TABLE_NAME_FIXES = {
    method: display for method, display in DISPLAY_NAMES.items()
    if method.lower() == display.lower()
}


def _printable(value):
    """Use the plot spelling for a method key, leaving all other values alone."""
    return _TABLE_NAME_FIXES.get(value, value) if isinstance(value, str) else value


def _presentation(table):
    """Copy of a report table whose method names use their plot spelling."""
    printable = table.rename(columns=_printable)
    for column in printable.columns:
        if printable[column].dtype == object:
            printable[column] = printable[column].map(_printable)
    return printable


def _numeric(values):
    """Observed numeric values only: drop non-numeric and non-finite entries."""
    return pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()


def _mcse(values):
    """Monte Carlo SE of a mean: sample SD divided by sqrt(number of runs)."""
    return monte_carlo_se(values)


def _has_observed_values(df, method, measure):
    """True when a result frame has at least one usable value for this pair."""
    column = f'{method}_{measure}'
    if column not in df.columns:
        return False
    return not _numeric(df[column]).empty


def _available_methods(frames, methods, measures):
    """Keep only methods observed in the supplied results.

    A method such as EXAONE is only present in the pilot, so the main and
    sensitivity reports would otherwise print an all-missing column.
    """
    frames = list(frames)
    return [
        method for method in methods
        if any(_has_observed_values(df, method, measure)
               for df in frames for measure in measures)
    ]


def _save_table(table, filename, tables_dir):
    """Write a report table into the caller's results directory.

    Reports are saved next to the results they describe, so ``tables_dir`` is
    required whenever ``filename`` is given. Nothing is written when
    ``filename`` is None, which lets the caller display the table only.
    """
    if filename is None:
        return
    if tables_dir is None:
        raise ValueError(f"tables_dir is required to save {filename}")
    os.makedirs(tables_dir, exist_ok=True)
    table.to_csv(os.path.join(tables_dir, filename))
    print(f"✓ {filename}")


def build_measures_table(errors_dict, metrics=None, filename='table.csv',
                         methods=None, tables_dir=None):
    """rows=(Setting, Metric), cols=method, values='mean ± SD (MCSE …)'.

    The SD describes run-to-run spread; the bracketed MCSE is the SE of the
    reported mean (SD / sqrt(number of Monte Carlo runs)).

    errors_dict : {setting_id: pd.DataFrame}
    Set filename=None to display the styled table without saving a CSV.
    """
    if not errors_dict:
        print(f"⚠️ no data to build {filename} — skipping")
        return None
    
    metrics = metrics or METRICS
    methods = _available_methods(errors_dict.values(), methods or METHODS, metrics)

    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            row = {'Setting': sid, 'Metric': metric}
            for m in methods:
                col = f'{m}_{metric}'
                if col in df.columns:
                    v = _numeric(df[col])
                    row[m] = (f"{v.mean():.3f} ± {v.std():.3f} (MCSE {_mcse(v):.4f})"
                              if not v.empty else '—')
                else:
                    row[m] = '—'
            rows.append(row)

    table = pd.DataFrame(rows).set_index(['Setting', 'Metric'])
    printable = _presentation(table)
    _save_table(printable, filename, tables_dir)
    return printable.style.apply(_style_best, axis=1)


# Only the three classification measures get a sample-size change table.
# Runtime and calibration shifts are already visible in measures.csv.
GAIN_METRICS = ['AUC', 'Brier', 'ClassError']


def build_gain_table(base_dict, sens_dict, metrics=GAIN_METRICS, methods=None,
                     filename='sample_size_changes.csv', tables_dir=None):
    """Sensitivity mean minus baseline mean for matching settings and measures.

    Only AUC, Brier and ClassError are reported by default: these are the
    classification results the sample-size change is meant to describe. The
    table holds changes in means only, never changes in Monte Carlo SE.
    """
    measures = list(metrics)
    methods = methods or METHODS
    if not base_dict or not sens_dict:
        return None

    common = [s for s in base_dict if s in sens_dict]
    if not common:
        return None

    # A gain needs the method on both sides of the comparison. Keeping a
    # method that only ran in one variant produced empty EXAONE columns.
    available = set(_available_methods(base_dict.values(), methods, measures))
    available &= set(_available_methods(sens_dict.values(), methods, measures))
    methods = [method for method in methods if method in available]
    if not methods:
        return None

    rows = []
    for sid in common:
        b, s = base_dict[sid], sens_dict[sid]
        for measure in measures:
            row = {'Setting': sid, 'Metric': measure}
            for m in methods:
                col = f'{m}_{measure}'
                bv = _numeric(b[col]) if col in b.columns else pd.Series(dtype=float)
                sv = _numeric(s[col]) if col in s.columns else pd.Series(dtype=float)
                row[m] = sv.mean() - bv.mean()
            rows.append(row)

    printable = _presentation(pd.DataFrame(rows).set_index(['Setting', 'Metric']))
    _save_table(printable, filename, tables_dir)
    return printable.style.format('{:+.3f}', na_rep='—')


def _rank_methods(df, methods, metric):
    """Rank observed means; calibration uses mean absolute distance from its ideal."""
    scores = {}
    for method in methods:
        column = f'{method}_{metric}'
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if metric == 'CalSlope':
            values = (values - 1.0).abs()
        elif metric == 'CalIntercept':
            values = values.abs()
        if values.notna().any():
            scores[method] = values.mean()
    return sorted(scores, key=scores.get, reverse=metric in HIGHER_IS_BETTER)


def build_best_methods_table(errors_dict, metrics=None, filename='best_methods.csv', tables_dir=None):
    """Best and second-best methods in each reporting group, per setting/dataset and measure.

    Groups are baseline, tree, and foundation. Rank once using all repetitions;
    show the raw measure as mean, sample SD, and Monte Carlo SE of the mean.
    Ties use the existing method order. This is a descriptive result summary.
    """
    metrics = metrics or METRICS
    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            for group, methods in GROUPS:
                ranked = _rank_methods(df, methods, metric)
                if not ranked:
                    continue
                row = {'Setting': sid, 'Metric': metric, 'Group': group}
                for rank, label in [(0, 'Best'), (1, 'Second')]:
                    if rank < len(ranked):
                        method = ranked[rank]
                        values = _numeric(df[f'{method}_{metric}'])
                        row[f'{label} method'] = method
                        row[f'{label} mean'] = values.mean()
                        row[f'{label} SD'] = values.std()
                        row[f'{label} MCSE'] = _mcse(values)
                        row[f'{label} runs'] = values.count()
                    else:
                        row[f'{label} method'] = '—'
                        row[f'{label} mean'] = np.nan
                        row[f'{label} SD'] = np.nan
                        row[f'{label} MCSE'] = np.nan
                        row[f'{label} runs'] = 0
                rows.append(row)
    if not rows:
        print(f"⚠️ no methods to summarize for {filename} — skipping")
        return None
    table = pd.DataFrame(rows).set_index(['Setting', 'Metric', 'Group'])
    printable = _presentation(table)
    _save_table(printable, filename, tables_dir)
    return printable.style.format(
        {column: '{:.4f}' if column.endswith('MCSE') else '{:.3f}'
         for column in printable if column.endswith(('mean', 'SD', 'MCSE'))},
        na_rep='—',
    )


def _effect_pairs(df, metric, all_models=False):
    """Select (rank, foundation method, tree method) pairs for one measure.

    The two headline comparisons use one shared best tree method, so the only
    difference between them is the foundation method. ``all_models`` adds the
    remaining foundation models and the remaining tree models for diagnostics.
    """
    tree = _rank_methods(df, TREE_METHODS, metric)
    foundation = _rank_methods(df, FOUNDATION_METHODS, metric)
    if not tree or not foundation:
        return []

    pairs = [(rank, method, tree[0])
             for rank, method in enumerate(foundation[:2], start=1)]
    if all_models:
        pairs += [(rank, method, tree[0])
                  for rank, method in enumerate(foundation[2:], start=3)]
        pairs += [(1, foundation[0], method) for method in tree[1:]]
    return pairs


def build_effect_table(errors_dict, metrics=None, filename='best_vs_best.csv', tables_dir=None, all_models=False):
    """Summarize within-run differences, foundation minus tree, as mean, SD, and MCSE.

    Baseline methods are excluded from this head-to-head comparison. Select the
    best tree method and the best two foundation methods once per measure using
    all repetitions. Each setting/measure then has one row per foundation: the
    best foundation against the best tree (rank 1) and the second-best foundation
    against the same best tree (rank 2). ``Foundation rank`` identifies the row
    so both effects can be drawn as separate series. A positive mean effect means
    the foundation model scored higher; for Brier and ClassError lower is better,
    so a negative effect favors the foundation model. With all_models=True,
    compare every model with the best method in the other comparison group.
    Calibration effects use raw slopes/intercepts, not their distances.
    """
    metrics = metrics or METRICS
    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            for foundation_rank, foundation_method, tree_method in _effect_pairs(
                    df, metric, all_models=all_models):
                foundation_values = _numeric(df[f'{foundation_method}_{metric}'])
                tree_values = _numeric(df[f'{tree_method}_{metric}'])
                # Keep the original repetition indices: a missing value in one
                # method must drop the same run from both sides of the pair.
                paired = pd.DataFrame({'Foundation': foundation_values, 'Tree': tree_values}).dropna()
                differences = paired['Foundation'] - paired['Tree']
                rows.append({
                    'Setting': sid, 'Metric': metric,
                    'Foundation rank': foundation_rank,
                    'Tree method': tree_method, 'Foundation method': foundation_method,
                    'Tree mean': paired['Tree'].mean(),
                    'Foundation mean': paired['Foundation'].mean(),
                    'Mean effect': differences.mean(), 'SD effect': differences.std(),
                    'MCSE effect': _mcse(differences),
                    'Paired runs': len(differences),
                })
    if not rows:
        print(f"⚠️ no group comparisons for {filename} — skipping")
        return None
    table = pd.DataFrame(rows).set_index(['Setting', 'Metric'])
    printable = _presentation(table)
    _save_table(printable, filename, tables_dir)
    return printable.style.format({
        'Tree mean': '{:.3f}', 'Foundation mean': '{:.3f}',
        'Mean effect': '{:+.3f}', 'SD effect': '{:.3f}', 'MCSE effect': '{:.4f}',
    }, na_rep='—')


def build_effect_samples(errors_dict, metrics=None, filename=None, tables_dir=None):
    """Per-run foundation-minus-tree differences for the headline comparisons.

    ``build_effect_table`` collapses the paired differences to a mean and SD.
    The half-violin forest needs the differences themselves, so this keeps one
    row per repetition with the setting, metric, foundation method, tree method
    and signed effect. Method names stay as stored result keys so plots can look
    up their colours. Set ``filename`` to also save the long table as a CSV.
    """
    metrics = metrics or METRICS
    rows = []
    for sid, df in errors_dict.items():
        for metric in metrics:
            for foundation_rank, foundation_method, tree_method in _effect_pairs(df, metric):
                foundation_values = pd.to_numeric(df[f'{foundation_method}_{metric}'], errors='coerce')
                tree_values = pd.to_numeric(df[f'{tree_method}_{metric}'], errors='coerce')
                paired = pd.DataFrame({'Foundation': foundation_values, 'Tree': tree_values})
                paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
                for run, effect in (paired['Foundation'] - paired['Tree']).items():
                    rows.append({
                        'Setting': sid, 'Metric': metric,
                        'Foundation rank': foundation_rank,
                        'Tree method': tree_method, 'Foundation method': foundation_method,
                        'Run': run, 'Effect': float(effect),
                    })
    if not rows:
        print(f"⚠️ no paired samples for {filename} — skipping")
        return None
    table = pd.DataFrame(rows)
    _save_table(table, filename, tables_dir)
    return table


def _style_best(row):
    metric = row.name[1]
    reverse = metric in HIGHER_IS_BETTER
    vals = {}
    for col in row.index:
        try:
            value = float(row[col].split(' ± ')[0])
            if metric == 'CalSlope':
                value = abs(value - 1.0)
            elif metric == 'CalIntercept':
                value = abs(value)
            vals[col] = value
        except Exception:
            pass
    if not vals:
        return [''] * len(row)
    ordered = sorted(vals.items(), key=lambda kv: kv[1], reverse=reverse)
    best = ordered[0][0]
    second = ordered[1][0] if len(ordered) > 1 else None
    return [
        'text-decoration: underline' if c == best
        else 'text-decoration: underline double' if c == second
        else ''
        for c in row.index
    ]
