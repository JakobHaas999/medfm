"""Run the pilot, simulations, sensitivity analysis, resampling, or report saved results."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def training_rows(result_path):
    """Total training rows recorded in a result file's meta, or None if absent."""
    meta_path = Path(result_path).with_suffix('.meta.json')
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    n = meta.get('n') or {}
    total = n.get('n1_tr', 0) + n.get('n2_tr', 0)
    return total or None


def prevalence_sample_sizes(y, total_train=1000, total_test=1000):
    """Split fixed train/test totals between the classes at the observed prevalence.

    The total number of rows stays the same across datasets, so only the class
    balance changes. Class counts always add up to the requested totals.
    """
    prevalence = float(np.mean(np.asarray(y)))
    n1_tr = int(round(total_train * (1.0 - prevalence)))
    n1_te = int(round(total_test * (1.0 - prevalence)))
    return {
        'n1_tr': n1_tr, 'n2_tr': total_train - n1_tr,
        'n1_te': n1_te, 'n2_te': total_test - n1_te,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=['pilot', 'simulation', 'sensitivity', 'resampling', 'report'])
    parser.add_argument('--repetitions', type=int, help='Number of Monte Carlo repetitions')
    parser.add_argument('--seed', type=int, help='Random seed; default 2025 (sensitivity: 12025)')
    parser.add_argument('--output', type=Path, help='New directory for this analysis')
    parser.add_argument('--results', type=Path, help='Saved result directory to report')
    parser.add_argument('--baseline', type=Path, help='Baseline results for the sensitivity sample-size change table')
    parser.add_argument('--no-bayes-auc', action='store_true', help='Hide the Bayes AUC line in reports')
    parser.add_argument('--no-bayes-roc', action='store_true', help='Hide the Bayes ROC curve in reports')
    parser.add_argument('--jobs', type=int, default=10, help='Worker processes for sensitivity/resampling (default: 10)')
    parser.add_argument('--settings', nargs='+', metavar='ID',
                        help='Setting IDs (pilot default: 2; simulation/sensitivity default: all)')
    parser.add_argument('--datasets', nargs='+', choices=['echo', 'glucose', 'blood_gas'],
                        help='Resampling datasets (default: all)')
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.settings is not None and args.analysis not in ('pilot', 'simulation', 'sensitivity'):
        parser.error('--settings is only available for pilot, simulation and sensitivity')
    if args.datasets is not None and args.analysis != 'resampling':
        parser.error('--datasets is only available for resampling')
    if args.baseline is not None and args.analysis != 'report':
        parser.error('--baseline is only available for report')
    if args.analysis == 'pilot' and args.repetitions is None:
        args.repetitions = 20
    if args.analysis == 'report':
        if args.results is None:
            parser.error('report requires --results')
    elif args.repetitions is None or args.repetitions < 1:
        parser.error('choose a positive --repetitions count')

    # Use a plotting backend that saves figures without opening windows.
    import matplotlib
    matplotlib.use('Agg')
    from medfm import (
        SENSITIVITY,
        SETTINGS,
        calculate_bayes_auc,
        config,
        evaluate_on_real_data,
        load_blood_gas_data,
        load_echo_data,
        load_glucose_data,
        save_outputs,
        simulation,
    )
    from medfm.simulation_core import evaluate_on_sim_data

    # Validate selections before creating an output folder or running models.
    if args.analysis in ('pilot', 'simulation', 'sensitivity'):
        settings = SENSITIVITY if args.analysis == 'sensitivity' else SETTINGS
        if args.analysis == 'pilot' and args.settings is None:
            settings = {2: settings[2]}
        if args.settings is not None:
            setting_ids = {str(setting_id): setting_id for setting_id in settings}
            unknown = [value for value in args.settings if value not in setting_ids]
            if unknown:
                parser.error(f"unknown --settings ID(s): {', '.join(unknown)}; "
                             f"choose from: {', '.join(setting_ids)}")
            settings = {setting_ids[value]: settings[setting_ids[value]] for value in args.settings}
    elif args.analysis == 'resampling':
        available_datasets = {
            'echo': ('case_study_1_echo_notes', load_echo_data),
            'glucose': ('case_study_2_blood_glucose_management_1', load_glucose_data),
            'blood_gas': ('case_study_3_blood_gas_oximetry', load_blood_gas_data),
        }
        selected = dict.fromkeys(args.datasets if args.datasets is not None else available_datasets)
        datasets = [(name, *available_datasets[name]) for name in selected]

    # Report existing results without loading data or fitting models.
    if args.analysis == 'report':
        from medfm.plot_functions import (
            plot_effect_forest,
            plot_gain_heatmap,
            plot_roc_curves_grid,
            plot_runtime_barplot_simple,
            plot_violin_grid,
        )
        from medfm.table_functions import (
            METRICS, build_measures_table, build_gain_table,
            build_best_methods_table, build_effect_table, build_effect_samples,
        )
        from medfm.utils import discover_sim, load_fixed, read_bayes_auc, read_labels
        files = {path.stem: path for path in sorted(args.results.glob('*.csv'))}
        if not files:
            parser.error('no result CSV files found in --results directory')
        # Sensitivity runs are compared against the matching main simulation.
        # Use --baseline when given; otherwise fall back to the sibling 'main'
        # directory so the gain table and gain plot are produced automatically.
        baseline = args.baseline
        if baseline is None and discover_sim(args.results, variant='sensitivity'):
            candidate = args.results.parent / 'main'
            if candidate.is_dir() and discover_sim(candidate):
                baseline = candidate
                print(f'Comparing sensitivity results against baseline {baseline}')
        if baseline is not None:
            baseline_files = discover_sim(baseline)
            sensitivity_files = discover_sim(args.results, variant='sensitivity')
            if not baseline_files or not sensitivity_files:
                parser.error('--baseline needs baseline simulation CSVs and --results needs sensitivity CSVs')
            if not set(baseline_files).intersection(sensitivity_files):
                parser.error('baseline and sensitivity directories have no matching settings')
        results = load_fixed(files)
        labels = read_labels({name: str(path) for name, path in files.items()})
        bayes = read_bayes_auc(files) if not (args.no_bayes_auc and args.no_bayes_roc) else {}
        names = list(results)
        # Real-data reports show every available foundation model against the
        # best tree ensemble. Simulation reports keep the two-model headline
        # comparison used in the paper.
        is_real_data = any(str(name).startswith('realdata_') for name in files)
        n_foundation = 0 if is_real_data else 2
        if is_real_data:
            print('Real-data report: showing all foundation models versus best tree ensemble')
        table_dir = args.results / 'tables'
        plot_dir = args.results / 'plots'
        table_dir.mkdir(exist_ok=True)
        plot_dir.mkdir(exist_ok=True)
        build_measures_table(results, filename='measures.csv', tables_dir=table_dir)
        build_best_methods_table(results, filename='best_methods.csv', tables_dir=table_dir)
        effect = build_effect_table(results, filename='best_vs_best.csv', tables_dir=table_dir,
                                    n_foundation=n_foundation)
        effect_samples = build_effect_samples(results, metrics=METRICS,
                                             n_foundation=n_foundation)
        gain = None
        if baseline is not None:
            gain = build_gain_table(
                load_fixed(baseline_files), load_fixed(sensitivity_files),
                filename='sample_size_changes.csv', tables_dir=table_dir,
            )
        plots = str(plot_dir) + '/'
        plot_violin_grid(results, names, labels=labels, metric='AUC', dir=plots, save_path='auc_violin.png', bayes_auc=bayes, show_bayes_auc=not args.no_bayes_auc)
        plot_violin_grid(results, names, labels=labels, metric='Brier', dir=plots, save_path='brier_violin.png')
        plot_roc_curves_grid(results, names, labels=labels, dir=plots, save_path='roc.png', bayes_auc=bayes, show_bayes_roc=not args.no_bayes_roc)
        plot_runtime_barplot_simple(results, names, labels=labels, dir=plots, save_path='runtime.png')
        if effect is not None:
            plot_effect_forest(effect.data, effect_samples=effect_samples,
                               labels=labels, dir=plots, save_path='forest.png',
                               n_foundation=n_foundation)
        if gain is not None:
            base_rows = training_rows(next(iter(baseline_files.values())))
            sens_rows = training_rows(next(iter(sensitivity_files.values())))
            if base_rows and sens_rows:
                gain_note = f'Sample-size sensitivity ({base_rows} → {sens_rows} training rows)'
            else:
                gain_note = 'Sample-size sensitivity'
            plot_gain_heatmap(gain.data, dir=plots, save_path='gain.png',
                              sensitivity_note=gain_note)
        print(f'Tables and plots saved in {args.results}')
        return

    # Make a new result folder so a previous analysis is not overwritten.
    parent = config.REAL_RESULTS_DIR if args.analysis == 'resampling' else config.SIM_RESULTS_DIR
    output = args.output or parent / f'{args.analysis}-{time.time_ns()}'
    if output.exists():
        parser.error('output directory already exists; choose another --output name')
    output.mkdir(parents=True)
    seed = args.seed if args.seed is not None else (12025 if args.analysis == 'sensitivity' else 2025)

    # The pilot and main simulation are sequential; sensitivity uses parallel repetitions.
    if args.analysis in ('pilot', 'simulation', 'sensitivity'):
        runner = evaluate_on_sim_data if args.analysis == 'sensitivity' else simulation
        for setting_id, cfg in settings.items():
            suffix = '_sensitivity' if args.analysis == 'sensitivity' else ''
            name = (f'setting{setting_id}_pilot' if args.analysis == 'pilot'
                    else f'simulation_setting{setting_id}{suffix}')
            worker_options = {'n_jobs': args.jobs} if args.analysis == 'sensitivity' else {}
            result = runner(
                n=cfg['n'], mu=cfg['mu'], sigma1=cfg['sigma1'], sigma2=cfg['sigma2'],
                dist=cfg['dist'], df=cfg.get('df'), iter=args.repetitions, g_seed=seed,
                include_exaone=args.analysis == 'pilot', **worker_options,
                progress_label=f'Setting {setting_id} {args.analysis} iterations',
                warnings_file=output / 'warnings' / f'{name}.warnings.json',
            )
            # simulation returns (results, warnings); the parallel runner returns results.
            if args.analysis in ('pilot', 'simulation'):
                errors, _warns = result
            else:
                errors = result
            save_outputs(errors, name, output_dir=output,
                         meta={**cfg, 'setting_id': setting_id, 'iter': args.repetitions, 'g_seed': seed,
                               'include_exaone': args.analysis == 'pilot',
                               'execution': 'parallel' if args.analysis == 'sensitivity' else 'sequential',
                               'workers': args.jobs if args.analysis == 'sensitivity' else 1,
                               'threads_per_worker': 1,
                               'bayes_auc': calculate_bayes_auc(cfg)})
    else:
        # Keep the total sample sizes, but match each dataset's observed positive rate.
        for dataset_key, dataset, loader in datasets:
            X, y, groups = loader(config.REAL_DATA_PATH)
            prevalence = float(np.mean(np.asarray(y)))
            sample_sizes = prevalence_sample_sizes(y)
            name = f'realdata_{dataset}'
            errors = evaluate_on_real_data(
                X, y, sample_sizes, groups=groups, iter=args.repetitions, g_seed=seed,
                n_jobs=args.jobs, include_exaone=False,
                progress_label=f"{dataset_key.replace('_', ' ').capitalize()} resampling iterations",
                warnings_file=output / 'warnings' / f'{name}.warnings.json',
            )
            save_outputs(errors, name, output_dir=output, meta={
                'dataset': dataset, 'n': sample_sizes, 'iter': args.repetitions,
                'prevalence': prevalence,
                'g_seed': seed, 'n_features': int(X.shape[1]), 'n_samples': int(X.shape[0]),
                'include_exaone': False, 'execution': 'parallel',
                'workers': args.jobs, 'threads_per_worker': 1,
            })
    print(f'Results saved in {output}')


if __name__ == '__main__':
    main()
