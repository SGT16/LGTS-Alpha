from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess
import sys


def command(root, study, job_id, data_dir, output_dir):
    plan = json.loads((root / 'protocols' / (study+'.json')).read_text())
    matches = [j for j in plan['jobs'] if j['id'] == job_id] if job_id else plan['jobs'][:1]
    if len(matches) != 1:
        raise ValueError('Select exactly one existing job ID')
    job = matches[0]
    settings = {**plan['fixed_config'], 'datasets':[job['dataset']], 'models':[job['model']],
                'activations':[job['activation']], 'seeds':[job['seed']],
                'seed_order':plan['seed_order'], 'learning_rate':job['learning_rate']}
    if study == 'alpha_roles':
        settings.update(split_width=job['width'],split_depth=job['depth'],lgts_alphas=[job['width']])
    elif study == 'bounds':
        settings.update(lgts_beta=job['beta'],lgts_alpha_min=job['alpha_lower'],
                        lgts_alpha_max=job['alpha_upper'],lgts_alphas=[job['alpha_initial']])
    engine = 'two_scale' if study == 'alpha_roles' else 'standard'
    argv = [sys.executable,'-u',str(root/'engines'/engine/'train_benchmark.py')]
    for key,value in settings.items():
        if isinstance(value,bool):
            if value:
                argv.append('--'+key)
        else:
            argv.append('--'+key)
            argv.extend(str(v) for v in (value if isinstance(value,list) else [value]))
    argv += ['--data_dir',str(data_dir),'--output_dir',str(output_dir)]
    return argv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',choices=['gelu_lr','alpha_roles','bounds'],required=True)
    parser.add_argument('--job')
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--format',choices=['json','shell'],default='json')
    args = parser.parse_args()
    argv = command(Path(__file__).resolve().parent,args.study,args.job,args.data_dir,args.output_dir)
    print(json.dumps(argv,indent=2) if args.format == 'json' else
          (subprocess.list2cmdline(argv) if os.name == 'nt' else shlex.join(argv)))
