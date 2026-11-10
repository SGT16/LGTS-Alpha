from pathlib import Path
import argparse
import json
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def reconstruct(root):
    import torch
    sys.path.insert(0, str(root / 'engines/standard'))
    from lgts_benchmark.models import build_model
    from lgts_benchmark.activations import lgts_parameter_rows
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    original = json.loads((root / 'initialization/initial_alpha_audit.json').read_text())
    plan = json.loads((root / 'protocols/bounds.json').read_text())
    fixed = plan['fixed_config']
    rows, errors = [], []
    for recorded in original['runs']:
        torch.manual_seed(recorded['seed'])
        model = build_model(
            'ResNet18', 3, 100, 'LGTS-f4-Alpha', log_t=.6,
            lgts_learnable=fixed['lgts_learning'] != 'none',
            lgts_learn_alpha=fixed['lgts_learning'] in {'both', 'alpha'},
            lgts_learn_beta=fixed['lgts_learning'] in {'both', 'beta'},
            lgts_alpha=recorded['alpha_initial_target'], lgts_beta=recorded['beta'],
            lgts_alpha_min=recorded['alpha_lower'], lgts_alpha_max=recorded['alpha_upper'],
            lgts_beta_min=fixed['lgts_beta_min'], lgts_beta_max=fixed['lgts_beta_max'],
            activation_scope=fixed['activation_scope'], activation_sharing=fixed['activation_sharing'],
            pretrained=False, small_input_stem=True,
        ).cpu()
        actual_sites = {r['layer'] for r in lgts_parameter_rows(model)}
        if actual_sites != set(original['sites']):
            raise ValueError('Unexpected activation sites: ' + recorded['job'])
        values, raw_values, beta_values = [], [], []
        for name in original['sites']:
            module = model.get_submodule(name)
            for tensor in (module.alpha, module.raw_alpha, module.beta):
                if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or tensor.numel() != 1:
                    raise ValueError('Unexpected tensor: ' + name)
            values.append(module.alpha.detach().item())
            raw_values.append(module.raw_alpha.detach().item())
            beta_values.append(module.beta.detach().item())
        difference = max(abs(a-b) for a,b in zip(values,recorded['alpha_by_site']))
        raw_difference = max(abs(a-recorded['raw_alpha_initial']) for a in raw_values)
        beta_difference = max(abs(a-recorded['beta_initial']) for a in beta_values)
        if difference > 1e-7 or raw_difference > 2e-7 or beta_difference > 2e-7:
            raise ValueError('Initialization differs from the original audit: ' + recorded['job'])
        errors.append(difference)
        rows.append({'job':recorded['job'],'seed':recorded['seed'],'alpha_by_site':values,
                     'raw_alpha_by_site':raw_values,'beta_by_site':beta_values,
                     'maximum_absolute_difference_from_original_alpha':difference})
        del model
    return {'method':'CPU reconstruction from the recorded configuration with the packaged model constructor.',
            'torch_version':str(torch.__version__),'device':'cpu','dtype':'float32',
            'sites':original['sites'],'models':len(rows),'values':sum(len(r['alpha_by_site']) for r in rows),
            'maximum_absolute_difference_from_original_alpha':max(errors),'runs':rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = reconstruct(Path(__file__).resolve().parents[1])
    text = json.dumps(result,indent=2) + '\n'
    if args.output:
        args.output.write_text(text,encoding='utf8')
    else:
        print(text,end='')
