from __future__ import annotations

import copy
import gc
import math
import random

PROTOCOL = {
    'schema_version': 1,
    'name': 'paired_amp_loss_scale_calibration_v1',
    'initial_scale': 65536.0,
    'minimum_scale': 1.0,
    'max_trials': 17,
    'scale_factor': 0.5,
    'original_equivalence_batch': 2,
    'additional_check': 'actual measurement batch; matched one optimizer step',
    'timing_included': False,
    'reset_policy': 'Immutable checkpoint + fresh AdamW/scaler + identical RNG for every side/trial',
    'retry_reason': 'Only nonfinite gradients or a skipped optimizer step; finite mismatch is final',
    'input_gradient_units': 'Unscaled: raw input gradient divided by the recorded forward loss scale',
    'scientific_training_modified': False,
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def scales():
    values = tuple(PROTOCOL['initial_scale'] * PROTOCOL['scale_factor'] ** n
                   for n in range(PROTOCOL['max_trials']))
    require(values[-1] == PROTOCOL['minimum_scale'], 'Calibration scale policy changed')
    return values


def steps_executed(step_values):


    return bool(step_values) and all(type(v) in (int, float) and math.isfinite(v) and v == 1
                                   for v in step_values.values())


def _snapshot(runtime, cpu_model, cell, backend, batch, amp, scale, seed):
    torch = runtime.torch
    model = wrapped = inputs = target = optimizer = scaler = criterion = output = loss = None
    runtime.guard.verify()
    try:
        model = copy.deepcopy(cpu_model).to('cuda').train()
        if cell['precision']['channels_last']:
            model = model.to(memory_format=torch.channels_last)
        wrapped = runtime.compiler(model, backend)
        inputs = runtime.input(cell, batch, requires_grad=True)
        target = torch.arange(batch, device='cuda') % cell['num_classes']
        optimizer = runtime.optimizer(model, cell)
        scaler = torch.amp.GradScaler('cuda', enabled=amp, init_scale=float(scale))
        criterion = runtime.criterion(cell)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        optimizer.zero_grad(set_to_none=True)
        used_scale = float(scaler.get_scale())
        with runtime.autocast(amp):
            output = wrapped(inputs)
            loss = criterion(output, target)
        scaler.scale(loss).backward()


        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        runtime.activations.project_activation_parameters_(model)
        torch.cuda.synchronize()
        scale_after = float(scaler.get_scale())
        gradients = {name: None if p.grad is None else p.grad.detach().cpu().clone()
                     for name, p in model.named_parameters()}
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        input_gradient = None if inputs.grad is None else (inputs.grad.detach().float() / used_scale).cpu().clone()
        output_cpu = output.detach().cpu().clone()
        loss_finite = bool(torch.isfinite(loss.detach()).all().item())
        step_values = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                value = optimizer.state.get(parameter, {}).get('step')
                step_values[name] = float(value.item()) if torch.is_tensor(value) else value
        input_finite = input_gradient is not None and bool(torch.isfinite(input_gradient).all().item())
        bad_gradients = [name for name, value in gradients.items()
                         if value is not None and not bool(torch.isfinite(value).all().item())]
        bad_states = [name for name, value in state.items() if not bool(torch.isfinite(value).all().item())]
        output_finite = bool(torch.isfinite(output_cpu).all().item())
        executed = steps_executed(step_values)
        evidence = {
            'backend': backend, 'batch_size': batch, 'used_scale': used_scale,
            'scale_after': scale_after, 'loss_finite': loss_finite,
            'output_finite': output_finite, 'input_gradient_finite': input_finite,
            'nonfinite_parameter_gradients': bad_gradients,
            'nonfinite_updated_state': bad_states, 'optimizer_step_values': step_values,
            'optimizer_step_executed': executed,
            'amp_skip': bool(amp and scale_after < used_scale),
        }
        finite = loss_finite and output_finite and input_finite and not bad_gradients and not bad_states
        return {'output': output_cpu, 'input_gradient': input_gradient,
                'parameter_gradients': gradients, 'updated_state': state,
                'finite': bool(finite), 'step_executed': executed,
                'safe_step': bool(finite and executed and scale_after == used_scale),
                'evidence': evidence}
    finally:
        model = wrapped = inputs = target = optimizer = scaler = criterion = output = loss = None
        gc.collect()
        torch.cuda.empty_cache()
        runtime.guard.verify()


def _pair(runtime, cpu_model, cell, backend, batch, amp, scale, seed, tolerance):


    left = _snapshot(runtime, cpu_model, cell, 'eager', batch, amp, scale, seed)
    right = _snapshot(runtime, cpu_model, cell, backend, batch, amp, scale, seed)
    result = {'batch_size': batch, 'scale': scale, 'sides': [left['evidence'], right['evidence']],
              'safe_step': left['safe_step'] and right['safe_step'], 'passed': False}
    if not result['safe_step']:
        result['status'] = 'nonfinite_or_skipped_step'
        return result
    result['output'] = runtime.error_metrics(left['output'], right['output'], tolerance)
    result['input_gradient'] = runtime.error_metrics(left['input_gradient'], right['input_gradient'], tolerance)
    result['parameter_gradients'], result['updated_state'] = [], []
    for category in ('parameter_gradients', 'updated_state'):
        require(set(left[category]) == set(right[category]), 'Training calibration parameter/state identities differ')
        for name in left[category]:
            a, b = left[category][name], right[category][name]
            metric = {'passed': a is None and b is None, 'both_unused': a is None and b is None} if a is None or b is None else runtime.error_metrics(a, b, tolerance)
            result[category].append({'name': name, **metric})
    result['passed'] = (result['output']['passed'] and result['input_gradient']['passed']
                        and all(row['passed'] for row in result['parameter_gradients'] + result['updated_state']))
    result['status'] = 'passed' if result['passed'] else 'finite_numerical_mismatch'
    return result


def select_scale(evaluate, *, amp, batches):

    trials = []
    for scale in scales() if amp else (1.0,):
        checks = []
        for batch in batches:
            check = evaluate(batch, scale)
            checks.append(check)
            if check.get('safe_step') and not check.get('passed'):
                trials.append({'scale': scale, 'checks': checks, 'status': 'finite_numerical_mismatch'})
                return {'passed': False, 'selected_scale': None, 'trials': trials,
                        'reason': 'Finite numerical mismatch; no loss-scale retry permitted'}
            if not check.get('safe_step'):
                break
        safe = len(checks) == len(batches) and all(c.get('safe_step') and c.get('passed') for c in checks)
        trials.append({'scale': scale, 'checks': checks,
                       'status': 'passed' if safe else 'nonfinite_or_skipped_step'})
        if safe:
            return {'passed': True, 'selected_scale': scale, 'trials': trials, 'checks': checks}
    return {'passed': False, 'selected_scale': None, 'trials': trials,
            'reason': 'No finite executed matched step within the fixed loss-scale calibration budget'}


def training_equivalence(runtime, cpu_model, cell, backend):
    policy = runtime.manifest['sampling_protocol']
    batches = list(dict.fromkeys((PROTOCOL['original_equivalence_batch'], int(cell['batch_size']))))
    results, calibrations = [], {}
    for label, amp in (('float32', False), ('amp_float16', True)):
        def evaluate(batch, scale):
            return _pair(runtime, cpu_model, cell, backend, batch, amp, scale,
                         policy['order_seed'], policy['tolerances'][label])
        calibration = select_scale(evaluate, amp=amp, batches=batches)
        calibrations[label] = calibration


        original = calibration.get('checks', [{}])[0]
        entry = {key: original[key] for key in ('output', 'input_gradient', 'updated_state', 'parameter_gradients') if key in original}
        entry.update(precision=label, passed=calibration['passed'],
                     selected_scale=calibration['selected_scale'],
                     amp_scales=[calibration['selected_scale']] * 2 if calibration['passed'] else [],
                     reason=calibration.get('reason'))
        results.append(entry)
    return {'passed': all(r['passed'] for r in results), 'checks': results,
            'mode': 'train; original dropout/BN retained; fresh matched one-step state per calibration trial',
            'selected_scale': {label: value['selected_scale'] for label, value in calibrations.items()},
            'calibration_protocol': copy.deepcopy(PROTOCOL), 'calibration': calibrations}


def training_scale(gate, *, amp):

    require(gate.get('passed') is True, 'Training timing cannot bypass its calibrated equivalence gate')
    key = 'amp_float16' if amp else 'float32'
    value = gate.get('selected_scale', {}).get(key)
    require(type(value) in (int, float) and math.isfinite(value)
            and value in (scales() if amp else (1.0,)), 'Missing or invalid calibrated scale')
    return float(value)
