"""Post-review, CPU-only statistics from a frozen, provenance-checked snapshot.

No torch, GPU, checkpoint loading, training controls, network access or source
result mutation. CSV files are numerical intermediates for the parent report.
All inferential analyses here were selected after review, not preregistered.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

SEEDS = (936137, 529212, 448132, 942529, 110107)
DATASETS = ('cifar100', 'dermamnist', 'organsmnist')
MODELS = ('ResNet18', 'MobileNetV2', 'EfficientNetV2S')
PRIMARY = 'test_f1_macro'
REFERENCE = 'LGTS-f4-Alpha'
BASELINES = ('LGTS-f4-Fixed', 'LGTS-f4-BaseSquareplus',
    'LGTS-f4-BaseSquareplusLearned', 'ELU', 'PELU', 'PReLU', 'Swish',
    'Mish', 'ReLU', 'LeakyReLU')
BASE_SEED = 20261102
BOOTSTRAPS = 10000
RANK_DECIMALS = 12
METRICS = (PRIMARY, 'test_accuracy', 'test_balanced_accuracy',
    'test_precision_macro', 'test_recall_macro', 'test_roc_auc_macro_ovr',
    'test_pr_auc_macro', 'test_nll', 'test_ece_15', 'test_brier_score',
    'train_time_sec', 'mean_epoch_time_sec', 'inference_ms_per_image',
    'throughput_images_sec', 'peak_training_gpu_memory_mb',
    'peak_inference_gpu_memory_mb', 'total_parameters', 'trainable_parameters',
    'lgts_alpha', 'lgts_alpha_layer_min', 'lgts_alpha_layer_max',
    'lgts_alpha_near_lower_fraction', 'validation_f1_macro', 'val_f1_macro')
TIMING = {'train_time_sec', 'mean_epoch_time_sec', 'inference_ms_per_image',
    'throughput_images_sec', 'peak_training_gpu_memory_mb', 'peak_inference_gpu_memory_mb'}

# Planned policy is compared, while actual fallback precision and resume counts
# remain reported observation attributes, not exclusion criteria.
METRIC_CONFIG = ('code_sha256', 'engine_version', 'study_stage', 'protocol',
    'selection_policy', 'evaluation_split', 'split_seed', 'pretrained',
    'weight_initialization', 'activation_scope', 'normalization', 'class_weight',
    'loss_name', 'stem_type', 'activation_site_policy', 'optimizer', 'scheduler',
    'learning_rate_initial', 'weight_decay', 'fused_optimizer', 'tf32',
    'cudnn_benchmark', 'channels_last', 'image_size', 'batch_size',
    'dataset_fingerprint_sha256', 'evaluation_precision_policy',
    'evaluation_tf32', 'evaluation_cudnn_tf32', 'train_count_before_aug',
    'validation_count', 'test_count', 'torch_version', 'torchvision_version',
    'cuda_version', 'cudnn_version', 'gpu_model')
REQUIRED_CONFIG = ('code_sha256', 'engine_version', 'protocol', 'selection_policy',
    'evaluation_split', 'optimizer', 'scheduler', 'learning_rate_initial',
    'weight_decay', 'batch_size', 'image_size', 'dataset_fingerprint_sha256',
    'evaluation_precision_policy')
RUN_CONFIG = ('patience', 'min_epochs', 'min_learning_rate', 'no_amp',
    'class_weight', 'normalization', 'small_input_stem', 'num_workers',
    'prefetch_factor', 'performance_mode', 'channels_last', 'tf32',
    'cudnn_benchmark', 'fused_optimizer')
ACTIVATION_CONFIG = ('lgts_alpha_min', 'lgts_alpha_max', 'lgts_beta',
    'lgts_beta_min', 'lgts_beta_max', 'lgts_learning', 'lgts_fixed')
CONDITION_FIELDS = ('stage', 'panel', 'dataset', 'model', 'activation', 'epochs',
    'alpha', 'sharing', 'augmentation')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def keyed_seed(key):
    return int.from_bytes(hashlib.sha256((str(BASE_SEED) + ':' + str(key)).encode()).digest()[:8], 'big')


def _vector(values):
    out = np.asarray(values, dtype=float)
    if out.ndim != 1 or out.size == 0 or not np.isfinite(out).all():
        raise ValueError('Expected a nonempty finite one-dimensional vector')
    return out


def midranks(values):
    """Average ranks for exactly tied values, independent of their order."""
    a = _vector(values)
    order = np.argsort(a, kind='stable')
    result = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        result[order[i:j]] = ((i + 1) + j) / 2.0
        i = j
    return result


def exact_signed_rank(differences):
    """Exact conditional sign-permutation test, two-sided, Wilcox zero rule.

    Differences are rounded to 12 decimals only for ranking, with zero values
    removed and tied magnitudes assigned average ranks. Every one of 2**n
    equally likely sign assignments is enumerated. This is not SciPy's
    no-ties-only exact reference distribution. Symmetric sign-exchangeability
    under the null is assumed. At five nonzero pairs the smallest p is .0625.
    """
    raw = _vector(differences)
    d = np.round(raw, RANK_DECIMALS)
    d = d[d != 0]
    if len(d) > 20:
        raise ValueError('Exact enumeration is restricted to at most 20 nonzero pairs')
    if not len(d):
        return dict(n=len(raw), n_nonzero=0, w_plus=0.0, w_minus=0.0,
            statistic=0.0, p_exact=1.0, rank_biserial=None,
            permutations=1, rank_rounding_decimals=RANK_DECIMALS)
    ranks = midranks(np.abs(d))
    plus = float(ranks[d > 0].sum())
    minus = float(ranks[d < 0].sum())
    total = float(ranks.sum())
    distance = abs(plus - total / 2)
    extreme = 0
    for signs in itertools.product((0, 1), repeat=len(d)):
        perm_plus = float(np.dot(signs, ranks))
        if abs(perm_plus - total / 2) >= distance - 1e-12:
            extreme += 1
    return dict(n=len(raw), n_nonzero=len(d), w_plus=plus, w_minus=minus,
        statistic=min(plus, minus), p_exact=extreme / (2 ** len(d)),
        rank_biserial=(plus - minus) / total, permutations=2 ** len(d),
        rank_rounding_decimals=RANK_DECIMALS)


def holm(pvalues):
    """Holm family-wise adjusted p values in the original order."""
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('p values must be finite and between zero and one')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for i, index in enumerate(order):
        running = max(running, (len(p) - i) * float(p[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_bootstrap(differences, key, reps=BOOTSTRAPS):
    """Percentile interval of the paired mean; n=5 intervals are exploratory."""
    d = _vector(differences)
    if type(reps) is not int or reps < 100:
        raise ValueError('At least 100 integer bootstrap replicates required')
    seed = keyed_seed(key)
    rng = np.random.default_rng(seed)
    values = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(axis=1)
    lower, upper = np.quantile(values, [.025, .975], method='linear')
    return dict(mean_difference=float(d.mean()),
        sd_difference=float(d.std(ddof=1)) if len(d) > 1 else None,
        ci_lower=float(lower), ci_upper=float(upper), bootstrap_reps=reps,
        rng_seed=seed, ci_method='paired percentile bootstrap 95%; exploratory; unadjusted')


def crossed_bootstrap(cube, key, reps=BOOTSTRAPS):
    """Product resampling of dataset, backbone and common paired-seed levels.

    Samples each axis independently with replacement and averages the Cartesian
    product, preserving the observed crossed array. It does not resample the
    45 cells independently. Only 3 dataset/architecture levels are available,
    so the resulting interval is a descriptive sensitivity, not robust
    population-level evidence or an exactly calibrated bootstrap.
    """
    a = np.asarray(cube, dtype=float)
    if a.ndim != 3 or min(a.shape) < 2 or not np.isfinite(a).all():
        raise ValueError('A finite complete three-factor cube is required')
    if type(reps) is not int or reps < 100:
        raise ValueError('At least 100 integer bootstrap replicates required')
    seed = keyed_seed(key)
    rng = np.random.default_rng(seed)
    ix = [rng.integers(0, n, size=(reps, n)) for n in a.shape]
    vals = a[ix[0][:, :, None, None], ix[1][:, None, :, None], ix[2][:, None, None, :]].mean(axis=(1, 2, 3))
    lower, upper = np.quantile(vals, [.025, .975], method='linear')
    return dict(mean_difference=float(a.mean()), ci_lower=float(lower), ci_upper=float(upper),
        bootstrap_reps=reps, rng_seed=seed, factor_sizes=list(a.shape),
        ci_method='crossed product percentile bootstrap 95%; exploratory; unadjusted',
        limitation='Only three dataset and three backbone levels; not 45 independent observations; no population generalization established')


def ni_rows(comparison, margins=(.005, .01, .02)):
    """Post-review NI sensitivity, not a declaration based on nonsignificance."""
    rows = []
    for margin in margins:
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError('Non-inferiority margins must be positive')
        lower = comparison['ci_lower']
        rows.append(dict(comparison_id=comparison.get('id', comparison.get('comparison_id')),
            family=comparison.get('family'), candidate=comparison.get('candidate'),
            comparator=comparison.get('comparator'), dataset=comparison.get('dataset'),
            model=comparison.get('model'), margin=margin,
            mean_difference=comparison['mean_difference'], ci_lower=lower,
            ci_upper=comparison['ci_upper'], threshold=-margin,
            lower_ci_above_negative_margin=bool(lower > -margin),
            interpretation='Post-review exploratory sensitivity only; unadjusted small-sample CI; not confirmatory NI',
            ci_method=comparison.get('ci_method'), direction='candidate minus comparator; higher macro-F1 is better'))
    return rows


def normalize_activation(name):
    return {'LGTS-f4-Base-Squareplus': 'LGTS-f4-BaseSquareplus',
        'LGTS-f4-Base-Squareplus-Learned': 'LGTS-f4-BaseSquareplusLearned'}.get(name, name)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def profile(record):
    m, cfg = record['metrics'], record.get('run_config') or {}
    return {'metrics': {k: m.get(k) for k in METRIC_CONFIG},
        'run_config': {k: cfg.get(k) for k in RUN_CONFIG}}


def condition(record):
    value = {k: record.get(k) for k in CONDITION_FIELDS}
    value['activation'] = normalize_activation(value['activation'])
    value['science_profile'] = profile(record)
    cfg = record.get('run_config') or {}
    value['activation_config'] = {k: cfg.get(k) for k in ACTIVATION_CONFIG}
    return value


def make_groups(records):
    grouped = defaultdict(list)
    definitions = {}
    identities = set()
    for record in records:
        identity = (record['stage'], record['job_id'])
        if identity in identities:
            raise ValueError('Duplicate record identity: ' + repr(identity))
        identities.add(identity)
        spec = condition(record)
        key = canonical(spec)
        definitions[key] = spec
        grouped[key].append(record)
    result = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda x: int(x['seed']))
        seeds = [int(x['seed']) for x in rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Duplicate seed in a condition: ' + key)
        result.append(dict(id=hashlib.sha256(key.encode()).hexdigest()[:20],
            condition=definitions[key], records=rows, seeds=seeds,
            complete_five_seed=set(seeds) == set(SEEDS)))
    return result


def summarize(group):
    c, rows = group['condition'], group['records']
    out = {k: c.get(k) for k in CONDITION_FIELDS}
    out.update(group_id=group['id'], n=len(rows), seed_ids=group['seeds'],
        complete_five_seed=group['complete_five_seed'],
        analysis_role='descriptive only' if c['stage'] != 'base' else 'post-review descriptive summary',
        timing_role='observed training/runtime records; not controlled common-GPU benchmark',
        host_ids=sorted({r['host_id'] for r in rows}),
        source_job_ids=[r['job_id'] for r in rows],
        source_sha256=[r['source_sha256'] for r in rows],
        science_profile=c['science_profile'], activation_config=c['activation_config'],
        actual_evaluation_precisions=sorted({str(r['metrics'].get('evaluation_precision_used')) for r in rows}),
        resumed_runs=sum(int(r['metrics'].get('resume_count', 0) or 0) > 0 for r in rows),
        values={})
    for metric in METRICS:
        values = [r['metrics'].get(metric) for r in rows]
        valid = np.array([v for v in values if finite_number(v)], dtype=float)
        if len(valid):
            out['values'][metric] = dict(n=len(valid), mean=float(valid.mean()),
                sd=float(valid.std(ddof=1)) if len(valid) > 1 else None,
                min=float(valid.min()), max=float(valid.max()),
                missing=len(rows)-len(valid))
    return out


def compatibility(left, right, allowed_axes=()):
    a, b = left['condition'], right['condition']
    if a['stage'] != 'base' or b['stage'] != 'base':
        return 'Only completed revision base panels are inferentially eligible'
    if not left['complete_five_seed'] or not right['complete_five_seed']:
        return 'Incomplete exact five-seed pairing'
    if a['science_profile'] != b['science_profile']:
        differing = [section + '.' + k for section in a['science_profile']
            for k in a['science_profile'][section]
            if a['science_profile'][section][k] != b['science_profile'][section][k]]
        return 'Scientific configuration differs: ' + ','.join(differing)
    if any(a['science_profile']['metrics'].get(k) is None for k in REQUIRED_CONFIG):
        return 'Required scientific configuration unavailable'
    if a['science_profile']['metrics']['evaluation_split'] != 'test':
        return 'Not final-test evidence'
    for k in ('dataset', 'model', 'epochs', 'augmentation', 'sharing', 'alpha'):
        if k not in allowed_axes and a[k] != b[k]:
            return 'Unintended axis differs: ' + k
    for g in (left, right):
        for r in g['records']:
            if not finite_number(r['metrics'].get(PRIMARY)):
                return 'Primary test metric unavailable or nonfinite'
            if int(r['metrics'].get('epochs_run', -1)) != int(g['condition']['epochs']):
                return 'Incomplete final epoch'
            if int(r['metrics'].get('seed', r['seed'])) != int(r['seed']):
                return 'Metric/plan seed mismatch'
    return None


def compare(left, right, family, allowed_axes=(), reps=BOOTSTRAPS):
    reason = compatibility(left, right, allowed_axes)
    if reason:
        return None, dict(family=family, candidate_group_id=left['id'],
            comparator_group_id=right['id'], reason=reason)
    a = {int(r['seed']): r for r in left['records']}
    b = {int(r['seed']): r for r in right['records']}
    values_a = [float(a[s]['metrics'][PRIMARY]) for s in SEEDS]
    values_b = [float(b[s]['metrics'][PRIMARY]) for s in SEEDS]
    diff = np.subtract(values_a, values_b).tolist()
    identifier = family + ':' + left['id'] + ':' + right['id']
    ca, cb = left['condition'], right['condition']
    row = dict(id=identifier, family=family, metric=PRIMARY,
        dataset=ca['dataset'], model=ca['model'],
        candidate=ca['activation'], comparator=cb['activation'],
        candidate_group_id=left['id'], comparator_group_id=right['id'],
        candidate_condition=ca, comparator_condition=cb,
        direction='candidate minus comparator; higher macro-F1 is better',
        analysis_timing='post-review exploratory analysis; not preregistered',
        seed_ids=list(SEEDS), candidate_values=values_a, comparator_values=values_b,
        paired_differences=diff,
        candidate_source_jobs=[a[s]['job_id'] for s in SEEDS],
        comparator_source_jobs=[b[s]['job_id'] for s in SEEDS],
        candidate_mean=float(np.mean(values_a)), comparator_mean=float(np.mean(values_b)),
        higher_seed_count=sum(x > 0 for x in diff), lower_seed_count=sum(x < 0 for x in diff),
        equal_seed_count=sum(x == 0 for x in diff),
        exact=exact_signed_rank(diff),
        superiority_conclusion='No superiority conclusion from the percentile CI alone; consult exact test and family correction')
    row.update(paired_bootstrap(diff, identifier, reps))
    return row, None


def adjust_families(rows):
    families = defaultdict(list)
    for row in rows:
        families[row['family']].append(row)
    for family, members in families.items():
        for row, p in zip(members, holm([r['exact']['p_exact'] for r in members])):
            row.update(p_holm=p, holm_family_size=len(members),
                reject_two_sided_after_holm=bool(p < .05))


def find_groups(groups, **criteria):
    return [g for g in groups if all(g['condition'].get(k) == v for k, v in criteria.items())]


def analyze(snapshot, reps=BOOTSTRAPS):
    groups = make_groups(snapshot['records'])
    summaries = [summarize(g) for g in groups]
    comparisons, omissions = [], []

    def add(left, right, family, axes=()):
        row, issue = compare(left, right, family, axes, reps)
        if issue:
            omissions.append(issue)
        else:
            comparisons.append(row)

    for dataset, model in itertools.product(DATASETS, MODELS):
        refs = find_groups(groups, stage='base', panel='components_baselines',
            dataset=dataset, model=model, activation=REFERENCE, epochs=100,
            alpha=.4, sharing='site', augmentation='standard')
        if len(refs) != 1:
            omissions.append(dict(context=[dataset, model], reason='Expected one scientific LGTS baseline condition, found ' + str(len(refs))))
            continue
        ref = refs[0]
        for activation in BASELINES:
            controls = find_groups(groups, stage='base', panel='components_baselines',
                dataset=dataset, model=model, activation=activation, epochs=100,
                alpha=.4, sharing='site', augmentation='standard')
            if len(controls) == 1:
                add(ref, controls[0], 'revision_components_all_90')
            else:
                omissions.append(dict(context=[dataset, model, activation], reason='Missing or ambiguous component condition'))
        for panel, axes in [('fixed_historical_mean', ('alpha',)), ('alpha_grid', ('alpha',)),
            ('alpha_sharing', ('sharing',)), ('alpha_zero_bound', ('alpha',)),
            ('organsmnist_no_vflip', ('augmentation',)),
            ('organsmnist_no_flip', ('augmentation',)),
            ('no_hflip_precision_bridge', ('augmentation',))]:
            controls = find_groups(groups, stage='base', panel=panel, dataset=dataset, model=model, epochs=100)
            for control in controls:
                if panel.startswith('organsmnist') or panel == 'no_hflip_precision_bridge':
                    if control['condition']['activation'] != REFERENCE:
                        continue
                add(ref, control, 'revision_' + panel, axes)

    long_refs = find_groups(groups, stage='base', panel='longer_budget', activation=REFERENCE, epochs=200)
    for ref in long_refs:
        c = ref['condition']
        for control in find_groups(groups, stage='base', panel='longer_budget',
                dataset=c['dataset'], model=c['model'], epochs=200):
            if control['id'] != ref['id']:
                add(ref, control, 'revision_200epoch_activations')
        old = find_groups(groups, stage='base', panel='components_baselines',
            dataset=c['dataset'], model=c['model'], activation=REFERENCE, epochs=100,
            alpha=c['alpha'], sharing=c['sharing'], augmentation=c['augmentation'])
        if len(old) == 1:
            add(ref, old[0], 'revision_LGTS_200_vs_100', ('epochs',))
    adjust_families(comparisons)

    aggregate, crossed = [], []
    for baseline in BASELINES:
        selected = [r for r in comparisons if r['family'] == 'revision_components_all_90'
                    and r['comparator'] == baseline]
        indexed = {(r['dataset'], r['model']): r for r in selected}
        if len(indexed) != 9 or len(selected) != 9:
            omissions.append(dict(aggregate_comparator=baseline, reason='Incomplete unique 3x3x5 crossed panel'))
            continue
        cube = np.array([[indexed[(d, m)]['paired_differences'] for m in MODELS] for d in DATASETS])
        means = cube.mean(axis=(0, 1)).tolist()
        identifier = 'revision_seed_aggregate:' + baseline
        row = dict(id=identifier, family='revision_seed_aggregate_all_10',
            metric=PRIMARY, candidate=REFERENCE, comparator=baseline,
            dataset='equal-weight three datasets', model='equal-weight three backbones',
            n_contexts=9, n_seeds=5, seed_ids=list(SEEDS), paired_differences=means,
            exact=exact_signed_rank(means), context_comparison_ids=[r['id'] for r in selected],
            analysis_timing='post-review exploratory analysis; not preregistered',
            limitation='Five seed means over fixed nine contexts; not 45 independent observations')
        row.update(paired_bootstrap(means, identifier, reps))
        aggregate.append(row)
        row2 = dict(id='revision_crossed:' + baseline, family='revision_crossed_all_10',
            candidate=REFERENCE, comparator=baseline, metric=PRIMARY,
            dataset_ids=list(DATASETS), model_ids=list(MODELS), seed_ids=list(SEEDS),
            paired_difference_cube=cube.tolist(), context_comparison_ids=[r['id'] for r in selected])
        row2.update(crossed_bootstrap(cube, row2['id'], reps))
        crossed.append(row2)
    adjust_families(aggregate)
    ni = [item for row in comparisons + aggregate + crossed for item in ni_rows(row)]
    return dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        analysis_timing='All methods and families selected post-review, not preregistered',
        method=dict(primary_metric=PRIMARY, reference=REFERENCE, seed_ids=list(SEEDS),
            confidence_level=.95, bootstrap_reps=reps, base_random_seed=BASE_SEED,
            rank_rounding_decimals=RANK_DECIMALS,
            exact_test='Two-sided conditional sign permutation of Wilcoxon signed ranks; Wilcox zero removal; midrank ties',
            exact_test_assumption='Independent seed-level differences with sign exchangeability under a zero-centered symmetric null',
            p_resolution='Five nonzero pairs imply minimum exact two-sided p=0.0625; these tests cannot reject at .05',
            correction='Holm within explicitly named families; not a global correction across all different analyses',
            ci_warning='Percentile intervals are exploratory and unadjusted, particularly fragile at n=5; CI exclusion of zero is not a superiority declaration',
            ni_warning='Margins 0.005/0.01/0.02 are post-review sensitivity settings. Flag is lower two-sided95% CI > -margin; no nonsignificance-based NI inference',
            crossed_warning='Product resampling of 3 datasets, 3 backbones and 5 common seeds. Few factor levels limit reliability and generalization',
            timing_warning='Observed runtime and memory are not a controlled common-GPU cost benchmark',
            incomplete_policy='LR and ImageNet descriptive only; no ranking, selection, significance or NI from incomplete panels',
            references=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html',
                'https://doi.org/10.1214/07-AOAS122', 'https://doi.org/10.1214/12-AOAS547']),
        input_coverage=snapshot.get('coverage'), input_issues=snapshot.get('issues'),
        counts=dict(input_records=len(snapshot['records']), groups=len(groups),
            paired_comparisons=len(comparisons), seed_aggregate_comparisons=len(aggregate),
            crossed_comparisons=len(crossed), omitted=len(omissions)),
        group_summary=summaries, paired_comparisons=comparisons,
        seed_aggregate_comparisons=aggregate, crossed_bootstrap=crossed,
        ni_sensitivity=ni, omissions=omissions)


def flat_rows(rows):
    return [{k: canonical(v) if isinstance(v, (dict, list, tuple)) else v
             for k, v in row.items()} for row in rows]


def write_csv(path, rows):
    rows = flat_rows(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def outputs(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'statistics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    summaries = []
    for source in result['group_summary']:
        row = {k: v for k, v in source.items() if k != 'values'}
        for metric, stats in source['values'].items():
            row.update({metric + '_' + key: value for key, value in stats.items()})
        summaries.append(row)
    write_csv(folder / 'group_summary.csv', summaries)
    write_csv(folder / 'paired_comparisons.csv', result['paired_comparisons'])
    write_csv(folder / 'seed_aggregate_comparisons.csv', result['seed_aggregate_comparisons'])
    write_csv(folder / 'ni_sensitivity.csv', result['ni_sensitivity'])
    write_csv(folder / 'crossed_bootstrap.csv', result['crossed_bootstrap'])
    write_csv(folder / 'omitted_comparisons.csv', result['omissions'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, default=Path(__file__).with_name('snapshot.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    result = analyze(json.loads(raw))
    result['snapshot_sha256'] = hashlib.sha256(raw).hexdigest()
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['numpy_version'] = np.__version__
    outputs(result, args.output_dir)
    print(json.dumps(result['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
