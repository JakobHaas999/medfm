import time
from tqdm import tqdm
import numpy as np

from joblib import Parallel, delayed, parallel_config
from pathlib import Path
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from .utils import capture_output, median_impute
from .execution import initialize_worker, run_logged_iteration
from . import config


# import models from model_definitions.py
from .model_definitions import (
    run_linear_sparse_logistic_regression,
    run_sparse_logistic_regression,
    run_random_forest,
    run_xgboost,
    run_catboost,
    run_qda,
    run_tabpfn_v25,
    run_tabpfn_v26,
    run_tabpfn_v3,
    run_tabpfn_v35,
    run_exaone,
    run_tabicl_v1,
    run_tabicl,
)




def _run_and_pack(fn, y_test, *args, **kwargs):
    """Run one model, measure its runtime, and calculate calibration from its test probabilities."""
    
    # time all
    start = time.perf_counter()
    res = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start

    (_, auc, class_error, fpr, tpr, proba, brier,
     setup_time, tune_time, fit_time, predict_time) = res
    p = np.clip(proba, 0.001, 0.999)
    cal = LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000).fit(logit(p).reshape(-1, 1), y_test)

    return (elapsed, class_error, auc, fpr, tpr, proba, brier,
            cal.coef_[0][0], cal.intercept_[0],
            setup_time, tune_time, fit_time, predict_time)
    
 
 # resampling Functions for realdata


def _group_split(rows, y_np, groups_np, seed, test_size=0.2, max_tries=10):
    """Split row indices into train/test with disjoint patient groups.

    Both sides keep both outcome classes so the models can be fitted and scored.
    The same groups are never present on both sides.
    """
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        raise ValueError("cannot split an empty set of rows")

    n_groups = np.unique(groups_np[rows]).size
    if n_groups < 2:
        raise ValueError(f"need at least two groups to split, found {n_groups}")

    n_splits = min(max(2, int(round(1.0 / test_size))), n_groups)
    for attempt in range(max_tries):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed) + attempt,
        )
        train_pos, test_pos = next(
            splitter.split(np.zeros(rows.size), y_np[rows], groups_np[rows])
        )
        train_rows, test_rows = rows[train_pos], rows[test_pos]
        if np.unique(y_np[train_rows]).size == 2 and np.unique(y_np[test_rows]).size == 2:
            return train_rows, test_rows

    raise ValueError(
        "could not build a grouped split with both classes on both sides "
        f"after {max_tries} attempts (rows={rows.size}, groups={n_groups})"
    )


def _take_patient_rows(rows, y_np, groups_np, cls, size, side, rng, exclude=None):
    """Pick one row from each of `size` distinct patients of class `cls`.

    Sampling patients instead of rows gives every patient the same weight, so a
    patient with many records does not dominate a repetition. `exclude` holds
    patients already used for the other class in the same split.
    """
    rows = np.asarray(rows, dtype=int)
    pool = rows[y_np[rows] == cls]
    groups = groups_np[pool]

    if exclude is not None and exclude.size:
        keep = ~np.isin(groups, exclude)
        pool = pool[keep]
        groups = groups[keep]

    if pool.size == 0:
        raise ValueError(
            f"not enough class-{cls} rows in the {side} patient pool: have 0, need {size}"
        )

    unique_groups = np.unique(groups)
    if unique_groups.size < size:
        raise ValueError(
            f"not enough class-{cls} patients in the {side} pool: "
            f"have {unique_groups.size}, need {size}"
        )

    chosen = rng.choice(unique_groups, size=size, replace=False)

    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    sorted_rows = pool[order]
    starts = np.searchsorted(sorted_groups, unique_groups, side="left")
    ends = np.searchsorted(sorted_groups, unique_groups, side="right")
    chosen_idx = np.searchsorted(unique_groups, chosen)
    counts = ends[chosen_idx] - starts[chosen_idx]
    offsets = np.floor(rng.random(size) * counts).astype(int)
    selected = sorted_rows[starts[chosen_idx] + offsets]
    return selected, chosen


# part1 - fucntion for single iteration calcualtion
def _single_real_iteration(i, X_np, y_np, groups_np,
                           n1_tr, n2_tr, n1_te, n2_te, g_seed, include_exaone=False):

    """Sample training and test rows from disjoint patient groups and evaluate every model once.

    Patients are split before rows are sampled, so the same patient can never appear
    in both the training and the test set. Each selected patient contributes one row
    and is used for only one class, so every patient has equal weight. The inner
    training/validation split keeps the same patient-disjoint property.
    """
    rng = np.random.default_rng(seed=i+g_seed)

    all_rows = np.arange(len(y_np))
    train_pool, test_pool = _group_split(
        all_rows, y_np, groups_np, seed=i+g_seed, test_size=0.2,
    )

    def take_pair(pool, size0, size1, side):
        """Sample one row per patient, using each patient for only one class."""
        n_groups0 = np.unique(groups_np[pool[y_np[pool] == 0]]).size
        n_groups1 = np.unique(groups_np[pool[y_np[pool] == 1]]).size
        if n_groups0 <= n_groups1:
            rows0, used0 = _take_patient_rows(
                pool, y_np, groups_np, 0, size0, side, rng,
            )
            rows1, _ = _take_patient_rows(
                pool, y_np, groups_np, 1, size1, side, rng, exclude=used0,
            )
        else:
            rows1, used1 = _take_patient_rows(
                pool, y_np, groups_np, 1, size1, side, rng,
            )
            rows0, _ = _take_patient_rows(
                pool, y_np, groups_np, 0, size0, side, rng, exclude=used1,
            )
        return rows0, rows1

    train0, train1 = take_pair(train_pool, n1_tr, n2_tr, 'training')
    test0, test1 = take_pair(test_pool, n1_te, n2_te, 'test')

    train_idx = rng.permutation(np.concatenate([train0, train1]))
    test_idx = rng.permutation(np.concatenate([test0, test1]))

    X_train = X_np[train_idx]
    y_train = y_np[train_idx]
    X_test = X_np[test_idx]
    y_test = y_np[test_idx]

    # Tune on patient-disjoint slices of the sampled training rows.
    sub_idx, val_idx = _group_split(
        train_idx, y_np, groups_np, seed=i+g_seed, test_size=0.2,
    )
    X_train_sub = X_np[sub_idx]
    y_train_sub = y_np[sub_idx]
    X_val = X_np[val_idx]
    y_val = y_np[val_idx]

    # Median-impute once per iteration so every model, including the foundation
    # models, sees the same training-derived fill values. Tuning uses medians
    # from the tuning slice; the final fit uses medians from the full training
    # sample, so validation/test values never inform the fills.
    X_train_sub, X_val = median_impute(X_train_sub, X_val)
    X_train, X_test = median_impute(X_train, X_test)

    # run models and store
    
    results = {
        'y_test': y_test,
        'L-SLR': _run_and_pack(
            run_linear_sparse_logistic_regression, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'SLR': _run_and_pack(
            run_sparse_logistic_regression, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'RandomForest': _run_and_pack(
            run_random_forest, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'XGBoost': _run_and_pack(
            run_xgboost, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'CatBoost': _run_and_pack(
            run_catboost, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'QDA': _run_and_pack(
            run_qda, y_test,
            X_train, X_test, y_train, y_test,
            X_train_sub, y_train_sub, X_val, y_val,
            random_state=i+g_seed),
        'TabPFN_v25': _run_and_pack(
            run_tabpfn_v25, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        'TabPFN_v26': _run_and_pack(
            run_tabpfn_v26, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        'TabPFN_v3': _run_and_pack(
            run_tabpfn_v3, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        'TabPFN_v35': _run_and_pack(
            run_tabpfn_v35, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        'TabICL-V1': _run_and_pack(
            run_tabicl_v1, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        'TabICL-V2': _run_and_pack(
            run_tabicl, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed),
        
    }
    if include_exaone:
        results['EXAONE'] = _run_and_pack(
            run_exaone, y_test,
            X_train, X_test, y_train, y_test, random_state=i+g_seed,
        )
    return results





#  part2 - fucntion for all iterations
def evaluate_on_real_data(X, y, n, groups=None, iter=100, g_seed = 2025, *, include_exaone=False, warnings_file=None, n_jobs=10, progress_label=None):
    #np.random.seed(g_seed)

    """Repeat patient-grouped sampling and model evaluation on an existing dataset.

    Check the available class sizes first, run repetitions in parallel, and combine their results.
    Patients in `groups` (or rows when groups is None) stay on one side of the outer and
    inner splits. progress_label optionally identifies the dataset in the progress bar.
    """
    initialize_worker()
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y, dtype=int)
    if groups is None:
        groups_np = np.arange(len(y_np))
    else:
        groups_np = np.asarray(groups)
        if groups_np.shape[0] != len(y_np):
            raise ValueError("groups must have one entry per row of X and y")

    # Count distinct patients (groups), not rows: each sampled patient must be
    # available for exactly one class across the training and test sets.
    idx0 = np.where(y_np == 0)[0]
    idx1 = np.where(y_np == 1)[0]
    n_groups0 = np.unique(groups_np[idx0]).size
    n_groups1 = np.unique(groups_np[idx1]).size

    n1_tr, n2_tr = n['n1_tr'], n['n2_tr']
    n1_te, n2_te = n['n1_te'], n['n2_te']

    if n_groups0 < n1_tr + n1_te or n_groups1 < n2_tr + n2_te:
        raise ValueError(
            f"Not enough patients per class.\n"
            f"Class 0: have {n_groups0} patients, need {n1_tr + n1_te}\n"
            f"Class 1: have {n_groups1} patients, need {n2_tr + n2_te}"
        )

    errors = {
        'L-SLR_ClassError': [], 'L-SLR_AUC': [], 'L-SLR_Time': [], 'L-SLR_FPR': [], 'L-SLR_TPR': [],
        'L-SLR_SetupTime': [], 'L-SLR_TuneTime': [], 'L-SLR_FitTime': [], 'L-SLR_PredictTime': [],
        'L-SLR_Brier': [], 'L-SLR_CalSlope': [], 'L-SLR_CalIntercept': [], 'L-SLR_Probas': [],
        'SLR_ClassError': [], 'SLR_AUC': [], 'SLR_Time': [], 'SLR_FPR': [], 'SLR_TPR': [],
        'SLR_SetupTime': [], 'SLR_TuneTime': [], 'SLR_FitTime': [], 'SLR_PredictTime': [],
        'SLR_Brier': [], 'SLR_CalSlope': [], 'SLR_CalIntercept': [], 'SLR_Probas': [],
        'RandomForest_ClassError': [], 'RandomForest_AUC': [], 'RandomForest_Time': [], 'RandomForest_FPR': [], 'RandomForest_TPR': [],
        'RandomForest_SetupTime': [], 'RandomForest_TuneTime': [], 'RandomForest_FitTime': [], 'RandomForest_PredictTime': [],
        'RandomForest_Brier': [], 'RandomForest_CalSlope': [], 'RandomForest_CalIntercept': [], 'RandomForest_Probas': [],
        'XGBoost_ClassError': [], 'XGBoost_AUC': [], 'XGBoost_Time': [], 'XGBoost_FPR': [], 'XGBoost_TPR': [],
        'XGBoost_SetupTime': [], 'XGBoost_TuneTime': [], 'XGBoost_FitTime': [], 'XGBoost_PredictTime': [],
        'XGBoost_Brier': [], 'XGBoost_CalSlope': [], 'XGBoost_CalIntercept': [], 'XGBoost_Probas': [],
        'CatBoost_ClassError': [], 'CatBoost_AUC': [], 'CatBoost_Time': [], 'CatBoost_FPR': [], 'CatBoost_TPR': [],
        'CatBoost_SetupTime': [], 'CatBoost_TuneTime': [], 'CatBoost_FitTime': [], 'CatBoost_PredictTime': [],
        'CatBoost_Brier': [], 'CatBoost_CalSlope': [], 'CatBoost_CalIntercept': [], 'CatBoost_Probas': [],
        'QDA_ClassError': [], 'QDA_AUC': [], 'QDA_Time': [], 'QDA_FPR': [], 'QDA_TPR': [],
        'QDA_SetupTime': [], 'QDA_TuneTime': [], 'QDA_FitTime': [], 'QDA_PredictTime': [],
        'QDA_Brier': [], 'QDA_CalSlope': [], 'QDA_CalIntercept': [],
        'QDA_Probas': [],
        'TabPFN_v25_ClassError': [], 'TabPFN_v25_AUC': [], 'TabPFN_v25_Time': [], 'TabPFN_v25_FPR': [], 'TabPFN_v25_TPR': [],
        'TabPFN_v25_SetupTime': [], 'TabPFN_v25_TuneTime': [], 'TabPFN_v25_FitTime': [], 'TabPFN_v25_PredictTime': [],
        'TabPFN_v25_Brier': [], 'TabPFN_v25_CalSlope': [], 'TabPFN_v25_CalIntercept': [], 'TabPFN_v25_Probas': [],
        'TabPFN_v26_ClassError': [], 'TabPFN_v26_AUC': [], 'TabPFN_v26_Time': [], 'TabPFN_v26_FPR': [], 'TabPFN_v26_TPR': [],
        'TabPFN_v26_SetupTime': [], 'TabPFN_v26_TuneTime': [], 'TabPFN_v26_FitTime': [], 'TabPFN_v26_PredictTime': [],
        'TabPFN_v26_Brier': [], 'TabPFN_v26_CalSlope': [], 'TabPFN_v26_CalIntercept': [], 'TabPFN_v26_Probas': [],
        'TabPFN_v3_ClassError': [], 'TabPFN_v3_AUC': [], 'TabPFN_v3_Time': [], 'TabPFN_v3_FPR': [], 'TabPFN_v3_TPR': [],
        'TabPFN_v3_SetupTime': [], 'TabPFN_v3_TuneTime': [], 'TabPFN_v3_FitTime': [], 'TabPFN_v3_PredictTime': [],
        'TabPFN_v3_Brier': [], 'TabPFN_v3_CalSlope': [], 'TabPFN_v3_CalIntercept': [], 'TabPFN_v3_Probas': [],
        'TabPFN_v35_ClassError': [], 'TabPFN_v35_AUC': [], 'TabPFN_v35_Time': [], 'TabPFN_v35_FPR': [], 'TabPFN_v35_TPR': [],
        'TabPFN_v35_SetupTime': [], 'TabPFN_v35_TuneTime': [], 'TabPFN_v35_FitTime': [], 'TabPFN_v35_PredictTime': [],
        'TabPFN_v35_Brier': [], 'TabPFN_v35_CalSlope': [], 'TabPFN_v35_CalIntercept': [], 'TabPFN_v35_Probas': [],
        'EXAONE_ClassError': [], 'EXAONE_AUC': [], 'EXAONE_Time': [], 'EXAONE_FPR': [], 'EXAONE_TPR': [],
        'EXAONE_SetupTime': [], 'EXAONE_TuneTime': [], 'EXAONE_FitTime': [], 'EXAONE_PredictTime': [],
        'EXAONE_Brier': [], 'EXAONE_CalSlope': [], 'EXAONE_CalIntercept': [], 'EXAONE_Probas': [],
        'TabICL-V1_ClassError': [], 'TabICL-V1_AUC': [], 'TabICL-V1_Time': [], 'TabICL-V1_FPR': [], 'TabICL-V1_TPR': [],
        'TabICL-V1_SetupTime': [], 'TabICL-V1_TuneTime': [], 'TabICL-V1_FitTime': [], 'TabICL-V1_PredictTime': [],
        'TabICL-V1_Brier': [], 'TabICL-V1_CalSlope': [], 'TabICL-V1_CalIntercept': [], 'TabICL-V1_Probas': [],
        'TabICL-V2_ClassError': [], 'TabICL-V2_AUC': [], 'TabICL-V2_Time': [], 'TabICL-V2_FPR': [], 'TabICL-V2_TPR': [],
        'TabICL-V2_SetupTime': [], 'TabICL-V2_TuneTime': [], 'TabICL-V2_FitTime': [], 'TabICL-V2_PredictTime': [],
        'TabICL-V2_Brier': [], 'TabICL-V2_CalSlope': [], 'TabICL-V2_CalIntercept': [], 'TabICL-V2_Probas': [],
        'y_test_all': []
    }

    if not include_exaone:
        errors = {key: values for key, values in errors.items() if not key.startswith("EXAONE_")}
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if warnings_file is None:
        warnings_file = Path(config.REAL_RESULTS_DIR) / "warnings" / f"{time.time_ns()}.warnings.json"
    worker_dir = Path(warnings_file).with_suffix(".workers")
    with capture_output(warnings_file, config.REAL_RESULTS_DIR) as (captured_warnings, progress), parallel_config(
        backend="loky", inner_max_num_threads=1, initializer=initialize_worker,
    ):
        results = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(
            delayed(run_logged_iteration)(
                _single_real_iteration, worker_dir / f"iteration_{i + 1}.warnings.json",
                i, X_np, y_np, groups_np,
                n1_tr, n2_tr, n1_te, n2_te, g_seed, include_exaone,
            )
            for i in range(iter)
        )

        for res, records, log_path in tqdm(results, total=iter, desc=progress_label or "Real-data resampling", file=progress, dynamic_ncols=True):
            captured_warnings.extend(records)
            print(Path(log_path).read_text(), end="")
            errors['y_test_all'].append(res['y_test'])
            for key, prefix in [
                ('L-SLR', 'L-SLR_'),
                ('SLR', 'SLR_'),
                ('RandomForest', 'RandomForest_'),
                ('XGBoost', 'XGBoost_'),
                ('CatBoost', 'CatBoost_'),
                ('QDA', 'QDA_'),
                ('TabPFN_v25', 'TabPFN_v25_'),
                ('TabPFN_v26', 'TabPFN_v26_'),
                ('TabPFN_v3', 'TabPFN_v3_'),
                ('TabPFN_v35', 'TabPFN_v35_'),
                ('EXAONE', 'EXAONE_'),
                ('TabICL-V1', 'TabICL-V1_'),
                ('TabICL-V2', 'TabICL-V2_')
            ]:
                if key == "EXAONE" and not include_exaone:
                    continue
                (t, class_error, auc, fpr, tpr, proba, brier, slope, intercept,
                 setup_time, tune_time, fit_time, predict_time) = res[key]
                errors[prefix + 'Time'].append(t)
                errors[prefix + 'SetupTime'].append(setup_time)
                errors[prefix + 'TuneTime'].append(tune_time)
                errors[prefix + 'FitTime'].append(fit_time)
                errors[prefix + 'PredictTime'].append(predict_time)
                errors[prefix + 'ClassError'].append(class_error)
                errors[prefix + 'AUC'].append(auc)
                errors[prefix + 'FPR'].append(fpr)
                errors[prefix + 'TPR'].append(tpr)
                errors[prefix + 'Brier'].append(brier)
                errors[prefix + 'CalSlope'].append(slope)
                errors[prefix + 'CalIntercept'].append(intercept)
                errors[prefix + 'Probas'].append(proba)
    return errors



