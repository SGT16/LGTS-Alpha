import argparse
import ast
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_fingerprint(names, hashes):
    result = hashlib.sha256()
    for name in names:
        result.update(name.encode('utf-8') + b'\0' + hashes[name].encode('ascii') + b'\0')
    return result.hexdigest()


def verify_engines(root, records):
    for name, record in records.items():
        directory = root / 'engines' / name
        tree = ast.parse((directory / 'train_benchmark.py').read_text(encoding='utf-8'))
        files = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FINGERPRINT_FILES' for t in node.targets))
        if list(files) != record['ordered_files']:
            raise ValueError(f'Fingerprint file order differs: {name}')
        for version in ('original', 'packaged'):
            identity = record[version]
            if set(identity['source_file_sha256']) != set(files):
                raise ValueError(f'Fingerprint file set differs: {name}/{version}')
            actual = combined_fingerprint(files, identity['source_file_sha256'])
            if actual != identity['engine_sha256']:
                raise ValueError(f'Fingerprint differs: {name}/{version}')
        actual_files = {relative: sha256(directory / relative) for relative in files}
        if actual_files != record['packaged']['source_file_sha256']:
            raise ValueError(f'Packaged training source differs: {name}')


def verify_plans(root, records):
    plans = {}
    for name, count in [('gelu_lr', 65), ('alpha_roles', 15), ('bounds', 30)]:
        plan = read(root / 'protocols' / (name + '.json'))
        jobs = plan['jobs']
        if len(jobs) != count or len({j['id'] for j in jobs}) != count:
            raise ValueError(f'Job coverage differs: {name}')
        if any(j['epochs'] != 100 or j['seed'] not in plan['seed_order'] for j in jobs):
            raise ValueError(f'Epoch or seed setting differs: {name}')
        engine = 'two_scale' if name == 'alpha_roles' else 'standard'
        if plan['engine_sha256'] != records[engine]['original']['engine_sha256']:
            raise ValueError(f'Recorded training source differs: {name}')
        plans[name] = plan
    return plans


def verify_initialization(root, lineage, bounds):
    metadata = lineage['initialization']
    audit_path = root / 'initialization' / 'initial_alpha_audit.json'
    audit = read(audit_path)
    if sha256(audit_path) != metadata['original_audit_sha256']:
        raise ValueError('Recorded initialization file differs')
    if audit['audit_script_sha256'] != metadata['original_audit_script_sha256']:
        raise ValueError('Recorded initialization script fingerprint differs')
    if audit['plan_sha256'] != sha256(root / 'protocols' / 'bounds.json') or audit['plan_sha256'] != metadata['original_bounds_plan_sha256']:
        raise ValueError('Recorded initialization plan differs')
    if audit['engine_sha256'] != lineage['engines']['standard']['original']['engine_sha256']:
        raise ValueError('Recorded initialization engine differs')
    if len(audit['runs']) != 35 or len(set(audit['sites'])) != 17 or len(audit['sites']) != 17:
        raise ValueError('Initialization coverage differs')
    if len({r['job'] for r in audit['runs']}) != len(audit['runs']):
        raise ValueError('Duplicate initialization record')
    jobs = {j['id']: j for j in bounds['jobs']}
    refs = {r['seed']: r for r in audit['runs'] if r['source'] == 'archive_reconstruction'}
    if set(refs) != set(bounds['seed_order']) or {r['paired_archive_id'] for r in refs.values()} != set(bounds['archive_reference']['run_ids']):
        raise ValueError('Archived initialization controls differ')
    planned = {r['job'] for r in audit['runs'] if r['source'] == 'planned_reconstruction'}
    if planned != set(jobs):
        raise ValueError('Planned initialization records differ')
    values = []
    for run in audit['runs']:
        reference = refs[run['seed']]
        if run['paired_archive_id'] != reference['paired_archive_id'] or run['paired_archive_job'] != reference['job']:
            raise ValueError(f"Paired archive differs: {run['job']}")
        settings = jobs.get(run['job'], bounds['archive_reference'])
        if any(run[k] != settings[k] for k in ('beta', 'alpha_lower', 'alpha_upper')) or run['alpha_initial_target'] != settings['alpha_initial'] or run['seed'] != settings.get('seed', run['seed']):
            raise ValueError(f"Initialization configuration differs: {run['job']}")
        observed = run['alpha_by_site']
        if len(observed) != 17 or not all(math.isfinite(v) for v in observed):
            raise ValueError(f"Invalid initialization: {run['job']}")
        if min(observed) != run['alpha_initial_min'] or max(observed) != run['alpha_initial_max']:
            raise ValueError(f"Initialization summary differs: {run['job']}")
        target = run['alpha_initial_target']
        raw = math.log((target - run['alpha_lower']) / (run['alpha_upper'] - target))
        if max(abs(x-target) for x in observed) > 1e-7 or abs(run['raw_alpha_initial'] - raw) > 1e-7 or abs(run['beta_initial'] - run['beta']) > 2e-7:
            raise ValueError(f"Unexpected initialized parameters: {run['job']}")
        values.extend(observed)
    return {'models': len(audit['runs']), 'sites': len(audit['sites']), 'values': len(values),
            'maximum_distance_from_0_40': max(abs(v - 0.4) for v in values)}


def verify(root):
    manifest = read(root / 'manifest.json')['sha256']
    for name, expected in manifest.items():
        if sha256(root / name) != expected:
            raise ValueError(f'Package file checksum differs: {name}')
    lineage = read(root / 'source_fingerprints.json')
    engines = lineage['engines']
    verify_engines(root, engines)
    plans = verify_plans(root, engines)
    return {'status': 'passed', 'manifested_files': len(manifest),
            'engine_sha256': {k: {v: r[v]['engine_sha256'] for v in ('original', 'packaged')} for k, r in engines.items()},
            'new_jobs': {k: len(v['jobs']) for k, v in plans.items()},
            'initialization': verify_initialization(root, lineage, plans['bounds'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))

