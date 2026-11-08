import copy
import sys
import types
from runtime import POLICY, KINDS, require

NATIVE="original_eager"
STABLE="supplementary_slope_fp32_intermediate_fp16_io"
def stable_slope(self, x):

    torch = sys.modules["torch"]
    if x.dtype in (torch.float16, torch.bfloat16):
        work = x.float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            gate = torch.sigmoid(self.beta * work)
            negative = self.alpha * torch.tanh(work / self.alpha)
            return (negative + gate * (work - negative)).to(x.dtype)
    return type(self).forward(self, x)


def slope_reference(rt, source, stable, dtype, points, reference_dtype):
    torch = rt.torch
    native = copy.deepcopy(source).to("cuda")
    reference = torch.tensor(points, device="cuda", dtype=dtype).to(reference_dtype).requires_grad_()
    observed = reference.detach().to(dtype).requires_grad_()
    candidate = copy.deepcopy(stable).to("cuda")
    with rt.autocast(reference_dtype == torch.float16):
        a = native(reference)
    with rt.autocast(dtype == torch.float16):
        b = candidate(observed)
    a.float().mean().backward()
    b.float().mean().backward()
    tol = POLICY["tolerances"]["amp_float16"]
    checks = {"output": rt.error_metrics(a, b, tol), "input_gradient": rt.error_metrics(reference.grad, observed.grad, tol)}
    expected = dict(native.named_parameters())
    for name, p in candidate.named_parameters():
        checks[name] = rt.error_metrics(expected[name].grad, p.grad, tol)
    return dict(passed=all(c["passed"] for c in checks.values()), tensors=checks)


def gated_module(rt, source, activation):
    original = rt.activation_equivalence(source, "eager")
    if original["passed"] or activation != "LGTS-f2-Slope":
        detail = dict(passed=original["passed"], original=original)
        if not original["passed"] and activation == "PELU":
            slope = float((source.a.detach() / source.b.detach()).float().item())
            detail["positive_tail_check"] = dict(slope=slope, input=60000., exact_output=60000.*slope,
                  maximum_finite_float16=65504., outside_float16_range=60000.*slope > 65504.)
        return source, NATIVE, detail

    stable = copy.deepcopy(source)
    stable.forward = types.MethodType(stable_slope, stable)
    supplementary = rt.activation_equivalence(stable, "eager")
    points = [-10., -1., -.0001, 0., .0001, 1., 10.]
    torch = rt.torch
    checks = dict(native_fp16_on_representable_region=slope_reference(rt, source, stable, torch.float16, points, torch.float16),
                  native_fp32_reference_at_fp16_extremes=slope_reference(rt, source, stable, torch.float16,
                            [-60000., -1000., *points, 1000., 60000.], torch.float32))
    return stable, STABLE, dict(passed=supplementary["passed"] and all(c["passed"] for c in checks.values()),
                                original=original, supplementary=supplementary, independent_checks=checks)


def build_operation(rt, module, cell, unit):
    torch = rt.torch
    module = copy.deepcopy(module).to("cuda")
    if hasattr(module, "inplace"):
        module.inplace = False
    dtype = getattr(torch, unit["dtype"].split(".")[-1])
    amp = cell["precision"]["autocast"] == "float16"
    x = torch.randn(unit["shape"], dtype=dtype, device="cuda", requires_grad=True)
    with rt.autocast(amp):
        fixed_y = module(x)
    delta = torch.ones_like(fixed_y)

    def forward():
        with torch.no_grad(), rt.autocast(amp):
            return module(x)

    def backward():
        module.zero_grad(set_to_none=True)
        x.grad = None
        fixed_y.backward(delta, retain_graph=True)

    def combined():
        module.zero_grad(set_to_none=True)
        x.grad = None
        with rt.autocast(amp):
            y = module(x)
        y.backward(delta)

    op = dict(zip(KINDS, (forward, backward, combined)))[unit["kind"]]
    op()
    finite = {"output": bool(torch.isfinite(fixed_y).all())}
    if unit["kind"] != KINDS[0]:
        finite["input_gradient"] = x.grad is not None and bool(torch.isfinite(x.grad).all())
        for key, parameter in module.named_parameters():
            if parameter.requires_grad:
                finite[key] = parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
    return op, finite


def source_at_site(rt, cell, sites, unit):
    name, source, shape, dtype = sites[unit["activation_site"]]
    require(shape[1:] == unit["shape"][1:] and dtype == unit["dtype"], "Checkpoint site differs from plan")
    if unit["parameter_state"] == "declared_initialization":
        kwargs = {k:v for k,v in cell["model_build_kwargs"].items() if k.startswith("lgts_") or k == "log_t"}
        return rt.activations.make_activation(cell["activation"], channels=shape[1], **kwargs)
    require(unit["parameter_state"] == "learned_checkpoint", "Unknown parameter state")
    return source
