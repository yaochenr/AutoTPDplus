from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tack_benchmark_common import file_hash, metric_values


def guard_run_manifest(args):
    excluded = {'output_dir', 'fold_ids'}
    payload = {'arguments': {k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items() if k not in excluded},
               'input_sha256': {str(p): file_hash(p) for p in
                                (args.autotpdplus, args.protacdb, args.test)},
               'code_sha256': {str(p): file_hash(p) for p in [
                   Path(__file__), Path(__file__).with_name('tack_official_benchmark.py'),
                   Path(__file__).with_name('tack_benchmark_common.py'),
                   *sorted((args.tack_repo / 'tackai').rglob('*.py')),
                   *sorted((args.tack_repo / 'configs').rglob('*.yaml'))]}}
    path = args.output_dir / 'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f'Run configuration/data/code changed: use a fresh output directory ({path})')
    else:
        if (args.output_dir / 'official_runs').exists():
            raise ValueError('Existing unmanifested models: use a fresh output directory')
        path.write_text(json.dumps(payload, indent=2) + '\n')


def repeated_splits(frame, task, samples, fraction, seed):
    seen = set()
    labels = frame.Activity.to_numpy() if task == 'activity' else None
    for sample in range(samples):
        split_seed = seed + sample
        for attempt in range(10000):
            train_idx, val_idx = train_test_split(
                np.arange(len(frame)), test_size=fraction,
                random_state=split_seed, stratify=labels)
            signature = tuple(sorted(val_idx.tolist()))
            if signature not in seen:
                break
            split_seed += samples
        else:
            raise ValueError('Not enough distinct validation subsets for requested samples')
        seen.add(signature)
        yield {'fold': sample, 'train_idx': train_idx.tolist(),
               'test_idx': val_idx.tolist(), 'split_seed': split_seed,
               'model_seed': seed + sample}


def compound_groups(frame):
    from tack_official_benchmark import canonical_smiles
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    observed = {}
    for i, (_, row) in enumerate(frame.iterrows()):
        smiles = canonical_smiles(row.SMILES)
        if smiles is None:
            raise ValueError('Compound grouping requires valid SMILES for every row')
        identities = [('smiles', smiles)]
        key = row.get('Connectivity_Key')
        if pd.notna(key) and str(key).strip().lower() not in {'', 'nan', '<na>', 'none'}:
            identities.append(('connectivity', str(key).strip()))
        for identity in identities:
            if identity in observed:
                parent[find(i)] = find(observed[identity])
            else:
                observed[identity] = i
    roots = [find(i) for i in range(len(frame))]
    names = {root: f'compound_{n:04d}' for n, root in enumerate(dict.fromkeys(roots))}
    return np.asarray([names[root] for root in roots])


def compound_splits(frame, samples, fraction, seed):
    groups = compound_groups(frame)
    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < 2:
        raise ValueError('Compound holdout requires at least two compound groups')
    target = fraction * len(frame)
    seen = set()
    for sample in range(samples):
        split_seed = seed + sample
        for _ in range(10000):
            order = np.random.default_rng(split_seed).permutation(len(unique))
            cumulative = np.cumsum(counts[order])[:-1]
            n_groups = int(np.argmin(np.abs(cumulative - target))) + 1
            val_mask = np.isin(groups, unique[order[:n_groups]])
            val_idx = np.flatnonzero(val_mask)
            signature = tuple(val_idx.tolist())
            if signature not in seen:
                break
            split_seed += samples
        else:
            raise ValueError('Not enough distinct compound validation subsets')
        seen.add(signature)
        train_idx = np.flatnonzero(~val_mask)
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise AssertionError('Compound overlap in generated split')
        yield dict(fold=sample, train_idx=train_idx.tolist(), test_idx=val_idx.tolist(),
                   split_seed=split_seed, model_seed=seed + sample)


def compound_split_audit(frame, splits):
    groups = compound_groups(frame)
    rows = []
    for split in splits:
        ti, vi = split['train_idx'], split['test_idx']
        overlap = len(set(groups[ti]) & set(groups[vi]))
        if overlap or set(ti) & set(vi) or set(ti) | set(vi) != set(range(len(frame))):
            raise ValueError('Invalid compound split membership')
        rows.append(dict(sample=split['fold'], train_rows=len(ti), validation_rows=len(vi),
                         validation_fraction=len(vi)/len(frame),
                         train_compounds=len(set(groups[ti])), validation_compounds=len(set(groups[vi])),
                         overlapping_compounds=overlap,
                         overlapping_validation_smiles=int(frame.iloc[vi].SMILES.isin(frame.iloc[ti].SMILES).sum())))
    return {'identity_rule': 'connected groups sharing canonical SMILES or nonmissing Connectivity_Key',
            'selection': 'random group permutation; prefix nearest requested validation row count; no label use',
            'group_by_position': groups.tolist(), 'samples': rows}


def shared_activity_pca_dimensions(corpora, args):
    audit = []
    widths = np.array([44, 7])
    for corpus, frame in corpora.items():
        for split in repeated_splits(frame, 'activity', args.samples,
                                     args.validation_fraction, args.seed):
            training = frame.iloc[split['train_idx']]
            counts = np.array([training.POI_UniProt.nunique(), training.Ligase_UniProt.nunique()])
            if np.min(counts) < 2:
                raise ValueError(f'{corpus}: Activity PCA needs at least two distinct POI/E3 proteins per training partition')
            widths = np.minimum(widths, counts - 1)
            audit.append(dict(corpus=corpus, sample=split['fold'],
                              training_unique_poi=int(counts[0]),
                              training_unique_ligase=int(counts[1])))
    result = tuple(int(x) for x in widths)
    return result, {'requested_dimensions': [44, 7], 'used_dimensions': list(result),
                    'policy': 'minimum of requested width and training unique-protein count minus one, across both corpora and all samples; test data unused',
                    'training_counts': audit}


def validate_predictions(frame, expected, sample):
    if len(frame) != len(expected) or frame['_Benchmark_Row_ID'].duplicated().any():
        raise ValueError(f'Invalid prediction row count/IDs for sample {sample}')
    if frame['_Benchmark_Row_ID'].tolist() != expected['_Benchmark_Row_ID'].tolist():
        raise ValueError(f'Prediction row alignment changed for sample {sample}')
    if not np.isfinite(frame[['target', 'pred']].to_numpy(float)).all():
        raise ValueError(f'Nonfinite prediction for sample {sample}')
    if not (frame['fold'] == sample).all():
        raise ValueError(f'Wrong sample ID in predictions for sample {sample}')


def resolve_splits(train, corpus, task, args):
    grouped = getattr(args, 'holdout_unit', 'row') == 'compound'
    generated = list(compound_splits(train, args.samples, args.validation_fraction, args.seed)
                     if grouped else repeated_splits(train, task, args.samples, args.validation_fraction, args.seed))
    if args.split_reference is None:
        return generated
    path = args.split_reference / 'official_runs' / corpus.lower() / task / 'splits.json'
    reference = json.loads(path.read_text())
    if reference['row_id_by_position'] != train['_Benchmark_Row_ID'].tolist():
        raise ValueError(f'Reference row order mismatch: {corpus}/{task}')
    if grouped:
        if [s['model_seed'] for s in reference['splits']] != [s['model_seed'] for s in generated]:
            raise ValueError('Reference model seeds mismatch')
        return generated
    if reference['splits'] != generated:
        raise ValueError(f'Reference split membership or seeds mismatch: {corpus}/{task}')

    return reference['splits']


def validate_reference_experiment(args, trains, tests, load_yaml, best_configs):
    reference = args.split_reference
    if reference.resolve() == args.output_dir.resolve():
        raise ValueError('Reference and output directories must differ')
    if args.split_strategy != 'repeated-holdout' or args.hyperparameter_mode != 'fixed':
        raise ValueError('Reference comparison requires fixed-parameter repeated holdout')
    completion = json.loads((reference / 'COMPLETED.json').read_text())
    if completion['sample_ids'] != list(range(args.samples)):
        raise ValueError('Reference experiment must contain all requested samples')
    manifest = json.loads((reference / 'run_manifest.json').read_text())
    direct = getattr(args, 'pdc50_target', 'dc50-quantile') == 'direct-pdc50'
    grouped = getattr(args, 'holdout_unit', 'row') == 'compound'
    if grouped and (manifest['arguments']['hyperparameter_mode'] != 'fixed'
                    or manifest['arguments'].get('holdout_unit', 'row') != 'row'
                    or manifest['arguments'].get('pdc50_target') != args.pdc50_target):
        raise ValueError('Compound control requires a fixed row-holdout reference with the same target')
    if direct and not grouped and (manifest['arguments']['hyperparameter_mode'] != 'fixed'
                   or manifest['arguments'].get('pdc50_target', 'dc50-quantile') != 'dc50-quantile'):
        raise ValueError('Direct pDC50 requires a fixed-parameter DC50-quantile reference')
    for path in (args.autotpdplus, args.protacdb, args.test):
        if manifest['input_sha256'].get(str(path.resolve())) != file_hash(path):
            raise ValueError(f'Reference input changed: {path}')
    for key in ['split_strategy', 'samples', 'validation_fraction', 'seed', 'pdc50_model',
                'batch_size', 'num_proc', 'threads', 'leakage_policy']:
        if manifest['arguments'][key] != getattr(args, key):
            raise ValueError(f'Reference setting changed: {key}')
    for path, digest in manifest['code_sha256'].items():
        if Path(path).is_relative_to(args.tack_repo) and file_hash(Path(path)) != digest:
            raise ValueError(f'Official TACK code/config changed: {path}')
    audit = json.loads((reference / 'official_code_audit.json').read_text())
    checked = []
    for (corpus, task), train in trains.items():
        for name in [f'{corpus.lower()}_{task}_train.csv', f'heldout_{task}.csv']:
            old = reference / 'adapted_data' / name
            new = args.output_dir / 'adapted_data' / name
            if file_hash(old) != file_hash(new):
                raise ValueError(f'Adapted data differ from reference: {name}')
        splits = resolve_splits(train, corpus, task, args)
        _, _, data_config, model_config, _ = best_configs(task, args, load_yaml)
        data_config.update(batch_size=args.batch_size, num_workers=0, num_proc=args.num_proc)
        old_run = next(r for r in audit['official_runs'] if r['corpus'] == corpus and r['task'] == task)
        expected_config = copy.deepcopy(old_run['data_config'])
        if direct and not grouped and task == 'pdc50':
            if not expected_config['normalize_labels'] or expected_config['standardize_labels']:
                raise ValueError('Reference does not use the expected quantile label normalization')
            expected_config.update(normalize_labels=False, standardize_labels=False)
        if expected_config != data_config or old_run['model_config'] != model_config:
            raise ValueError(f'Feature or initial model configuration changed: {corpus}/{task}')
        checked.append(dict(corpus=corpus, task=task, samples=len(splits), train_rows=len(train),
                            test_rows=len(tests[task]), data_identical=True,
                            data_and_splits_identical=not grouped, model_seeds_identical=True))
        if grouped:
            split_audit = compound_split_audit(train, splits)
            (args.output_dir / f'{corpus.lower()}_{task}_compound_split_audit.json').write_text(
                json.dumps(split_audit, indent=2) + '\n')
    (args.output_dir / 'reference_comparison_audit.json').write_text(json.dumps({
        'reference_run': str(reference.resolve()),
        'intervention': ('compound-group holdouts instead of row holdouts; training target and fixed parameters unchanged'
                         if grouped else 'direct pDC50 labels without quantile normalization; fixed parameters unchanged'
                         if direct else 'fixed paper-selected parameters instead of per-sample Optuna tuning'),
        'checks': checked}, indent=2) + '\n')
    print('REFERENCE CHECK PASSED: identical inputs, adapted rows, model seeds, features and TACK code; '
          + ('new compound-disjoint splits verified.' if grouped else 'saved splits identical.'), flush=True)


def prepare_training_target(frame, task, args):
    if task != 'pdc50' or getattr(args, 'pdc50_target', 'dc50-quantile') != 'direct-pdc50':
        return frame
    from tack_official_benchmark import direct_pdc50_from_nm
    result = frame.copy()
    result['pDC50'] = direct_pdc50_from_nm(result['DC50'])
    result['_Prediction_Space'] = 'pdc50'
    return result


def run_repeated_holdout(corpus, task, train, test, args, run_cv, load_yaml, best_configs):
    from datasets import Dataset
    model_type, labels, data_config, model_config, method = best_configs(task, args, load_yaml)
    data_config = copy.deepcopy(data_config)
    data_config.update(batch_size=args.batch_size, num_workers=0, num_proc=args.num_proc)
    root = args.output_dir / 'official_runs' / corpus.lower() / task
    root.mkdir(parents=True, exist_ok=True)
    splits = resolve_splits(train, corpus, task, args)
    train = prepare_training_target(train, task, args)
    test = prepare_training_target(test, task, args)
    direct = task == 'pdc50' and getattr(args, 'pdc50_target', 'dc50-quantile') == 'direct-pdc50'
    if direct:
        (root / 'target_contract.json').write_text(json.dumps({
            'label': 'pDC50', 'training_formula': '9 - log10(DC50_nM)',
            'normalize_labels': False, 'standardize_labels': False,
            'early_stopping_metric': 'RMSE in exact pDC50 units',
            'prediction_space': 'pdc50', 'raw_measurement_column': 'DC50',
            'evaluation': 'previous pDC50 metric including 1e-12 M offset, unchanged',
            'censored_training_values': 'bounds treated as point labels, unchanged from reference',
        }, indent=2) + '\n')
    tune = args.hyperparameter_mode == 'tuned'
    grouped = getattr(args, 'holdout_unit', 'row') == 'compound'

    (root / 'splits.json').write_text(json.dumps({
        'protocol': ('random complete-compound groups; approximate validation row fraction' if grouped
                     else 'random row holdout; Activity stratified by label'),
        'validation_fraction': args.validation_fraction,
        'row_id_by_position': train['_Benchmark_Row_ID'].tolist(), 'splits': splits}, indent=2) + '\n')
    selected = set(range(args.samples) if args.fold_ids is None else args.fold_ids)
    official_globals = run_cv.__globals__
    original_splitter = official_globals['create_cv_splits']
    original_tuner = official_globals['tune_xgboost_hyperparameters']
    original_trainer = official_globals['train_xgboost']
    outputs = []
    for split in splits:
        sample = split['fold']
        if sample not in selected:
            continue
        sample_dir = root / f'sample_{sample:02d}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        checkpoints, predictions = sample_dir / 'checkpoints', sample_dir / 'predictions'

        def splitter(*_args, **_kwargs):
            yield split

        def trainer(dtrain, dval, config):
            config = copy.deepcopy(config)
            config['model_config']['seed'] = split['model_seed']
            if direct:
                for matrix, indices in [(dtrain, split['train_idx']), (dval, split['test_idx'])]:
                    np.testing.assert_allclose(matrix.get_label(), train.iloc[indices]['pDC50'],
                                               rtol=1e-6, atol=1e-6)
            model = original_trainer(dtrain, dval, config)
            if grouped:
                diagnostics = {'metric_space': 'exact pDC50; all training/validation records',
                               'best_iteration_zero_based': model.best_iteration,
                               'saved_boosting_rounds': model.num_boosted_rounds(),
                               'prediction_tree_policy': 'all saved trees, unchanged'}
                for name, matrix in [('train', dtrain), ('validation', dval)]:
                    residual = model.predict(matrix) - matrix.get_label()
                    diagnostics[f'{name}_rmse'] = float(np.sqrt(np.mean(residual**2)))
                    diagnostics[f'{name}_mae'] = float(np.mean(np.abs(residual)))
                (sample_dir / 'fit_diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
            return model

        def tuner(dtrain, dval, n_trials, base_config, seed):
            if not tune:
                raise AssertionError('HPO must not run in fixed-parameter mode')
            base_config = copy.deepcopy(base_config)
            base_config['model_config']['seed'] = split['model_seed']
            result = original_tuner(dtrain, dval, n_trials=n_trials,
                                    base_config=base_config, seed=split['model_seed'])
            (sample_dir / 'best_config.json').write_text(json.dumps(result, indent=2) + '\n')
            return result

        if not tune:
            fixed_config = copy.deepcopy(model_config)
            fixed_config['model_config']['seed'] = split['model_seed']
            (sample_dir / 'fixed_config.json').write_text(json.dumps(fixed_config, indent=2) + '\n')
        print(f'\nHOLDOUT {corpus}/{task} sample={sample + 1}/{args.samples} '
              f'train={len(split["train_idx"])} validation={len(split["test_idx"])} '
              f'mode={args.hyperparameter_mode} trials={args.tuning_trials if tune else 0} '
              f'seed={split["model_seed"]}', flush=True)
        official_globals['create_cv_splits'] = splitter
        official_globals['tune_xgboost_hyperparameters'] = tuner
        official_globals['train_xgboost'] = trainer
        try:
            run_cv(dataset=Dataset.from_pandas(train, preserve_index=False),
                   held_out_dataset=Dataset.from_pandas(test, preserve_index=False),
                   data_config=copy.deepcopy(data_config), model_type=model_type,
                   model_config=copy.deepcopy(model_config),
                   task_name={'pdc50': 'dc50', 'activity': 'bin'}.get(task, task),
                   labels=labels, group='random', n_splits=2, n_repeats=1,
                   checkpoints_dir=checkpoints, results_dir=predictions,
                   tune_hyperparameters=tune, n_tuning_trials=args.tuning_trials if tune else 0,
                   tune_first_fold_only=False)
        finally:
            official_globals['create_cv_splits'] = original_splitter
            official_globals['tune_xgboost_hyperparameters'] = original_tuner
            official_globals['train_xgboost'] = original_trainer
        for name, expected in [('test', test), ('val', train.iloc[split['test_idx']])]:
            files = list(predictions.glob(f'*split={name}.csv'))
            if len(files) != 1:
                raise ValueError(f'Expected one {name} prediction file in {predictions}')
            frame = pd.read_csv(files[0])
            validate_predictions(frame, expected, sample)
            if direct:
                if not frame['_Prediction_Space'].eq('pdc50').all():
                    raise ValueError('Direct pDC50 prediction-space marker missing')
                np.testing.assert_allclose(frame['target'], expected['pDC50'], rtol=1e-6, atol=1e-6)
            if name == 'test':
                frame['official_prediction_file'] = str(files[0])
                frame['split_seed'] = split['split_seed']
                frame['model_seed'] = split['model_seed']
                outputs.append(frame)
        config_name = 'best_config.json' if tune else 'fixed_config.json'
        if not (sample_dir / config_name).exists():
            raise ValueError(f'Missing per-sample configuration: {sample_dir}')
    return pd.concat(outputs, ignore_index=True), {
        'corpus': corpus, 'task': task, 'paper_method': method,
        'official_model_type': model_type, 'official_labels': labels,
        'data_config': data_config, 'model_config': model_config,
        'prediction_files': len(outputs), 'split_strategy': 'repeated-holdout',
        'holdout_unit': getattr(args, 'holdout_unit', 'row'),
        'tuning_trials_per_sample': args.tuning_trials if tune else 0,
        'hyperparameter_mode': args.hyperparameter_mode,
        'tune_each_sample': tune, 'sample_ids': sorted(selected),
        'final_training': 'training partition only; validation early stopping',
        'training_target': 'direct pDC50' if direct else 'official TACK label pipeline',
        'hpo_objective': ('1 - accuracy' if task == 'activity' else 'RMSE on TACK-normalized labels') if tune else None,
        'prediction_tree_policy': 'official TACK model.predict (all saved trees)',
    }


def paired_bootstrap(task, y, auto_pred, baseline_pred, clusters, n_boot, seed):
    a, b = metric_values(task, y, auto_pred), metric_values(task, y, baseline_pred)
    clusters = np.asarray(clusters).astype(str)
    unique = np.unique(clusters)
    groups = [np.flatnonzero(clusters == value) for value in unique]
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in a}
    def delta(metric, av, bv):
        return bv - av if metric in {'RMSE', 'MAE', 'LogLoss'} else av - bv
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if task == 'activity' and len(np.unique(y[idx])) < 2:
            continue
        aa, bb = metric_values(task, y[idx], auto_pred[idx]), metric_values(task, y[idx], baseline_pred[idx])
        for metric in a:
            value = delta(metric, aa[metric], bb[metric])
            if np.isfinite(value):
                draws[metric].append(value)
    rows = []
    for metric in a:
        boot = np.asarray(draws[metric])
        lo, hi = np.quantile(boot, [.025, .975]) if len(boot) else (np.nan, np.nan)
        p = min(1., 2 * (min(np.sum(boot <= 0), np.sum(boot >= 0)) + 1) / (len(boot) + 1)) if len(boot) else np.nan
        rows.append(dict(task=task, metric=metric, AutoTPDplus=a[metric], PROTACDB=b[metric],
                         delta_favoring_AutoTPDplus=delta(metric, a[metric], b[metric]),
                         ci95_low=lo, ci95_high=hi, bootstrap_p_two_sided=p,
                         n_test=len(y), n_test_doi_clusters=len(unique), n_valid_bootstrap=len(boot)))
    return pd.DataFrame(rows)


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    indices = np.flatnonzero(np.isfinite(values))
    order = indices[np.argsort(values[indices])]
    result[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(order), 0, -1)))
    return result


def save_repeat_metrics(long, output):
    from tack_official_benchmark import ensemble_predictions, interval_bounds, interval_metric_values
    rows = []
    for (corpus, task, sample), part in long.groupby(['corpus', 'benchmark_task', 'fold']):
        values = ensemble_predictions(part, task)
        primary = values if task == 'activity' else values[values.is_exact_primary.astype(bool)]
        metrics = metric_values(task, primary.y_true.to_numpy(), primary.prediction.to_numpy())
        for metric, value in metrics.items():
            rows.append(dict(corpus=corpus, task=task, sample=sample, evaluation='point-valued', metric=metric, value=value))
        if task != 'activity':
            lower, upper = interval_bounds(task, values.y_true.to_numpy(), values.value_operator.to_numpy())
            metrics = interval_metric_values(values.prediction.to_numpy(), lower, upper)
            for metric, value in metrics.items():
                rows.append(dict(corpus=corpus, task=task, sample=sample, evaluation='interval-aware', metric=metric, value=value))
    result = pd.DataFrame(rows)
    result.to_csv(output / 'per_sample_metrics.csv', index=False)
    result.groupby(['corpus', 'task', 'evaluation', 'metric']).value.agg(['count', 'mean', 'std']).to_csv(output / 'per_sample_metric_summary.csv')


def write_reference_comparison(args):
    direct = getattr(args, 'pdc50_target', 'dc50-quantile') == 'direct-pdc50'
    current_name, reference_name = ('direct_pdc50', 'dc50_quantile') if direct else ('fixed', 'hpo')
    prefix = 'quantile_vs_direct_pdc50' if direct else 'hpo_vs_fixed'
    if getattr(args, 'holdout_unit', 'row') == 'compound':
        current_name, reference_name = 'compound', 'row'
        prefix = 'row_vs_compound_pdc50'
    delta_name = f'delta_favoring_{current_name}'
    keys = ['corpus', 'task', 'sample', 'evaluation', 'metric']
    fixed = pd.read_csv(args.output_dir / 'per_sample_metrics.csv')
    tuned = pd.read_csv(args.split_reference / 'per_sample_metrics.csv')
    paired = fixed.merge(tuned, on=keys, how='left', validate='one_to_one',
                         suffixes=(f'_{current_name}', f'_{reference_name}'), indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Missing reference model metrics for a fixed-model sample')
    paired = paired.drop(columns='_merge')
    lower_better = {'RMSE', 'MAE', 'Interval_RMSE', 'Interval_MAE', 'LogLoss'}
    a_value, b_value = f'value_{current_name}', f'value_{reference_name}'
    paired[delta_name] = np.where(paired.metric.isin(lower_better),
        paired[b_value] - paired[a_value], paired[a_value] - paired[b_value])
    paired.to_csv(args.output_dir / f'{prefix}_per_model.csv', index=False)
    paired.groupby(['corpus', 'task', 'evaluation', 'metric']).agg(**{
        'count': ('sample', 'size'), f'{current_name}_mean': (a_value, 'mean'),
        f'{current_name}_sd': (a_value, 'std'), f'{reference_name}_mean': (b_value, 'mean'),
        f'{reference_name}_sd': (b_value, 'std'), f'mean_{delta_name}': (delta_name, 'mean'),
    }).to_csv(args.output_dir / f'{prefix}_summary.csv')
    if args.fold_ids is None:
        rows = []
        for filename in ['metrics_all.csv', 'metrics_interval_aware.csv']:
            a = pd.read_csv(args.output_dir / filename)
            b = pd.read_csv(args.split_reference / filename)
            if a.empty:
                continue
            value_cols = ['AutoTPDplus', 'PROTACDB']
            a = a.melt(id_vars=['task', 'metric'], value_vars=value_cols,
                       var_name='corpus', value_name=current_name)
            b = b.melt(id_vars=['task', 'metric'], value_vars=value_cols,
                       var_name='corpus', value_name=reference_name)
            combined = a.merge(b, on=['task', 'metric', 'corpus'], validate='one_to_one')
            combined[delta_name] = np.where(combined.metric.isin(lower_better),
                combined[reference_name] - combined[current_name], combined[current_name] - combined[reference_name])
            rows.append(combined)
        pd.concat(rows, ignore_index=True).to_csv(
            args.output_dir / f'{prefix}_ensemble.csv', index=False)
