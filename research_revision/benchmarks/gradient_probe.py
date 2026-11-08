import math


PROTOCOL_NAME = "supplementary_paired_model_gradient_scale_v1"


SCALES = tuple(2.0 ** -n for n in range(9))


def _stats(torch, values):
    finite = torch.isfinite(values)
    total = values.numel()
    nfinite = int(finite.sum().item())
    report = dict(elements=total, finite_elements=nfinite, nan=0, posinf=0, neginf=0,
                  min_finite=None, max_finite=None)
    if nfinite != total:
        report.update(nan=int(torch.isnan(values).sum().item()),
                      posinf=int(torch.isposinf(values).sum().item()),
                      neginf=int(torch.isneginf(values).sum().item()))
        values = values[finite]
    if nfinite:
        lo, hi = torch.aminmax(values)
        report.update(min_finite=float(lo.item()), max_finite=float(hi.item()))
    return report


def _compare(runtime, left, right, scale, tolerance):
    if left is None or right is None:
        return dict(passed=left is right, both_unused=left is right,
                    finite_mismatch=left is not right, nonfinite=False)
    if left.shape != right.shape:
        raise ValueError("Mismatched diagnostic gradient shapes")
    torch = runtime.torch
    a, b = left.detach().float() / scale, right.detach().float() / scale
    sa, sb = _stats(torch, a), _stats(torch, b)
    nonfinite = sa['finite_elements'] < a.numel() or sb['finite_elements'] < b.numel()
    if nonfinite:
        mask = torch.isfinite(a) & torch.isfinite(b)
        a, b = a[mask], b[mask]
    comparison = runtime.error_metrics(a, b, tolerance)
    finite_mismatch = not comparison["passed"]
    return dict(reference=sa, candidate=sb, finite_pair_comparison=comparison,
                finite_mismatch=finite_mismatch, nonfinite=nonfinite,
                passed=not nonfinite and not finite_mismatch)


def _trial(runtime, eager, candidate, source, precision, scale, tolerance, baseline):
    runtime.guard.verify()
    torch = runtime.torch
    eager.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    xa, xb = (source.detach().clone().requires_grad_() for _ in range(2))
    with runtime.autocast(precision == 'amp_float16'):
        ya, yb = eager(xa), candidate(xb)
        la, lb = ya.float().square().mean(), yb.float().square().mean()
    unchanged = baseline is None or (torch.equal(ya.detach(),baseline[0]) and torch.equal(yb.detach(),baseline[1]))
    if baseline is None:
        baseline = (ya.detach().clone(), yb.detach().clone())
    output = _compare(runtime, ya, yb, 1.0, tolerance)
    trial = dict(precision=precision, scale=scale, output=output, unscaled_objective=[la.item(), lb.item()],
                 outputs_unchanged=unchanged, parameter_gradients=[], passed=False)
    if not output["passed"] or not unchanged or not all(math.isfinite(v) for v in trial["unscaled_objective"]):
        trial["decision"] = 'output_failure'
        return trial, baseline
    (la * scale).backward()
    (lb * scale).backward()
    trial["input_gradient"] = _compare(runtime, xa.grad, xb.grad, scale, tolerance)
    left = dict(eager.named_parameters())
    right = dict(getattr(candidate, "_orig_mod", candidate).named_parameters())
    if left.keys() != right.keys():
        raise ValueError("Mismatched diagnostic parameter names")
    for name, parameter in left.items():
        pair = _compare(runtime, parameter.grad, right[name].grad, scale, tolerance)
        trial["parameter_gradients"].append(dict(name=name, **pair))
    checks = [trial['input_gradient'], *trial['parameter_gradients']]
    if any(item["finite_mismatch"] for item in checks):
        trial["decision"] = "finite_mismatch"
    elif any(item["nonfinite"] for item in checks):
        trial["decision"] = "nonfinite_gradient"
    else:
        trial.update(decision="passed", passed=True)
    return trial, baseline


def paired_model_probe(runtime, eager, candidate, cell, backend, *, protocol):

    if protocol != PROTOCOL_NAME:
        raise ValueError("Explicit supplementary diagnostic protocol required")
    from runtime import POLICY

    base = getattr(candidate, "_orig_mod", candidate)
    if eager.training or candidate.training or base.training:
        raise ValueError("Diagnostic requires original eval-mode models")
    if any(p.grad is not None for model in (eager,base) for p in model.parameters()):
        raise ValueError("Native gradient diagnostics must release their gradients first")
    runtime.guard.verify()


    original = {name:t.detach().cpu().clone() for name,t in eager.state_dict().items()}

    def verify_state(model):
        state = model.state_dict()
        if (original.keys() != state.keys() or any(
                state[k].dtype != t.dtype or not runtime.torch.equal(state[k].detach().cpu(), t)
                for k,t in original.items())):
            raise ValueError("Diagnostic model state differs from the original checkpoint copy")

    verify_state(base)
    result = dict(backend=backend, model=[], passed=False, protocol=protocol, supplementary=True,
                  cell_id=cell['id'], checkpoint_sha256=cell['checkpoint_sha256'], scales=list(SCALES),
                  comparison_units="FP32 gradients divided by the trial loss scale",
                  loss_definition="mean(square(logits.float()))", batch_size=min(2,cell['batch_size']),
                  native_benchmark_modified=False, attempts=[])
    try:
        for label, scales in [('float32', (1.0,)), ('amp_float16', SCALES)]:
            source = runtime.input(cell,result['batch_size'])
            baseline = None
            for scale in scales:
                trial, baseline = _trial(runtime,eager,candidate,source,label,scale,POLICY['tolerances'][label],baseline)
                result['attempts'].append(trial)
                if trial['decision'] != 'nonfinite_gradient':
                    break
            result['model'].append(dict(precision=label, passed=trial['passed'], decision=trial['decision'],
                                        scale=trial['scale'], attempt_index=len(result['attempts'])-1))
            if not trial['passed']:
                return result
        result['passed'] = True
        return result
    finally:
        eager.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        verify_state(eager)
        verify_state(base)
        runtime.guard.verify()
