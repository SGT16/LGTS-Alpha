import copy
import math
from itertools import islice
from runtime import POLICY, KINDS, require
import pelu as replay

INPUT_FILE="observed_pairs.pt"
PROTOCOL="prelu_checkpoint_unscaled_weighted_mean_ce_paired_gradient_v1"
def limits(tensor):
    low, high = tensor.detach().aminmax()
    low, high = float(low), float(high)
    require(math.isfinite(low) and math.isfinite(high), "Nonfinite PReLU replay input or loss derivative")
    return low, high


def capture_identity(cell, units, signature):
    sites = {}
    for u in units:
        require(u["cell_id"] == cell["id"] and u["checkpoint_sha256"] == cell["checkpoint_sha256"]
                and u["source_seed"] == cell["source_seed"] and u["model"] == cell["model"]
                and u["dataset"] == cell["dataset"] and u["activation"] == cell["activation"] == "PReLU"
                and u["parameter_state"] == "learned_checkpoint" and u["kind"] in KINDS[1:]
                and u["dtype"] == "torch.float16" and cell["precision"]["autocast"] == "float16"
                and 0 < u["batch_size"] == u["shape"][0] <= cell["batch_size"], "Unexpected PReLU capture unit")
        shape = [cell["batch_size"], *u["shape"][1:]]
        require(sites.setdefault(u["activation_site"], shape) == shape, "Conflicting original PReLU site shape")
    require(sites, "No PReLU capture sites")
    return dict(protocol=PROTOCOL, plan_sha256=signature, cell_id=cell["id"],
                checkpoint_sha256=cell["checkpoint_sha256"], dataset=cell["dataset"], model=cell["model"],
                source_seed=cell["source_seed"], dataset_fingerprint_sha256=cell["dataset_fingerprint_sha256"],
                training_batch_size=cell["batch_size"], full_site_shapes=sites, tensor_file=INPUT_FILE)


def sample_pairs(rt, model, cell, identity, progress):

    torch = rt.torch
    device = next(model.parameters()).device
    engine, loader, sampler = replay.training_loader(rt, cell)
    criterion = rt.criterion(cell)
    require(criterion.reduction == "mean", "Source training loss reduction changed")
    mode = engine.resolve_class_weight_mode(cell["config"]["class_weight"], cell["dataset"])
    require(mode == cell["resolved_class_weight"] and (criterion.weight is not None) == (mode == "balanced"),
            "Source training class weights differ")
    sites = identity["full_site_shapes"]
    frames, hooks, batches, pairs = {}, [], [], {}

    def record(site):
        def hook(module, args, output):
            x = args[0]
            require(site not in frames and list(x.shape) == sites[site] and x.shape == output.shape
                    and x.dtype == output.dtype == torch.float16 and output.requires_grad,
                    "PReLU site repeated or primitive tensor changed")
            frames[site] = (x.detach(), output)
        return hook

    try:
        for site in sites:
            hooks.append(model.get_submodule(site).register_forward_hook(record(site)))
        for index, (images, labels) in enumerate(islice(loader, replay.BATCHES)):
            progress()
            rt.guard.verify()
            frames.clear()
            size = cell["batch_size"]
            indices = list(sampler.indices[index * size:(index + 1) * size])
            require(len(labels) == len(indices) == size, "Incomplete original training batch")
            batch_hash = replay.hash_tensors([("images", images), ("labels", labels)])
            x, targets = engine.move_batch(images, labels, device, cell["precision"]["channels_last"])
            with torch.enable_grad():
                with rt.autocast(True):
                    loss = criterion(model(x), targets)
                require(set(frames) == set(sites) and bool(loss.isfinite()), "Invalid training-site loss replay")


                upstream = torch.autograd.grad(loss, tuple(frames[site][1] for site in sites))
            row = dict(batch=index + 1, training_indices=indices, transformed_batch_sha256=batch_hash,
                       loss=float(loss.detach()), sites={})
            for site, dy in zip(sites, upstream):
                site_x = frames[site][0]
                require(site_x.shape == dy.shape and dy.dtype == site_x.dtype, "Mismatched upstream derivative")
                lo, hi = limits(site_x)
                dlo, dhi = limits(dy)
                row["sites"][site] = dict(minimum=lo, maximum=hi, gradient_minimum=dlo, gradient_maximum=dhi,
                                          upstream_nonzero=int(dy.count_nonzero()))
                if index == 0:
                    pairs[site] = dict(x=site_x.to("cpu", copy=True), dy=dy.detach().to("cpu", copy=True))
            batches.append(row)
    finally:
        for hook in hooks:
            hook.remove()
        frames.clear()
        del loader
    require(len(batches) == replay.BATCHES and pairs.keys() == sites.keys(), "Incomplete training-gradient capture")
    ranges = {}
    for site, pair in pairs.items():
        rows = [b["sites"][site] for b in batches]
        ranges[site] = dict(shape=sites[site], minimum=min(r["minimum"] for r in rows),
                            maximum=max(r["maximum"] for r in rows), gradient_minimum=min(r["gradient_minimum"] for r in rows),
                            gradient_maximum=max(r["gradient_maximum"] for r in rows),
                            input_sha256=replay.hash_tensors([("x", pair["x"])]),
                            upstream_sha256=replay.hash_tensors([("dy", pair["dy"])]))
    meta = dict(**identity, sites=ranges, batches=batches, sampled_images=len(batches) * cell["batch_size"],
                timed_batch=1, split="train_val", model_mode="train", autocast="float16", parameter_updates=0,
                loss="CrossEntropyLoss", reduction="mean", class_weight_mode=mode,
                class_weights=criterion.weight.detach().cpu().tolist() if criterion.weight is not None else None,
                loss_scale=1.0, augmentation=cell["config"]["augmentation"], normalization=cell["config"]["normalization"],
                input_origin="Paired native activation inputs and derivatives of the original full-batch weighted mean "
                             "loss replayed with frozen learned weights. No historical training gradients were logged.",
                range_scope="Eight real training batches; time matched first-batch prefix without renormalizing dL/dy.")
    return dict(metadata=meta, pairs=pairs)


def operation(rt, module, full_x_cpu, full_dy_cpu, cell, unit, capture_meta, *, device="cuda"):
    torch = rt.torch
    size = unit["batch_size"]
    require(unit["kind"] in KINDS[1:] and size <= capture_meta["training_batch_size"]
            and unit["activation_site"] in capture_meta["sites"], "Invalid PReLU operation")
    rt.set_policy(cell)
    tx = full_x_cpu[:size]
    tdy = full_dy_cpu[:size]
    require(list(tx.shape) == unit["shape"] and tx.shape == tdy.shape and tx.dtype == tdy.dtype == torch.float16,
            "PReLU timing pair differs from original native operation")
    x = tx.detach().to(device=device, copy=True).requires_grad_()
    dy = tdy.detach().to(device=device)
    layer = copy.deepcopy(module).to(device)
    with rt.autocast(True):
        fixed = layer(x)

    def backward():
        x.grad = None
        layer.zero_grad(set_to_none=True)
        fixed.backward(dy, retain_graph=True)

    def combined():
        x.grad = None
        layer.zero_grad(set_to_none=True)
        with rt.autocast(True):
            output = layer(x)
        output.backward(dy)

    op = backward if unit["kind"] == KINDS[1] else combined
    ref = copy.deepcopy(module).to(device=device, dtype=torch.float32)
    ref_x = x.detach().float().requires_grad_()
    with rt.autocast(False):
        ref_y = ref(ref_x)
    ref_y.backward(dy.float())
    op()
    tol = POLICY["tolerances"]["amp_float16"]
    checks = {name: rt.error_metrics(a, b, tol) for name, a, b in (
        ("output", ref_y, fixed), ("input_gradient", ref_x.grad, x.grad),
        ("slope_gradient", ref.weight.grad, layer.weight.grad))}
    nonzero = int(dy.count_nonzero())
    active = int(((x.detach() < 0) & (dy != 0)).count_nonzero())
    finite_weight = bool(layer.weight.grad.isfinite().all())
    dx_nonzero = int(x.grad.count_nonzero())
    lost_weight_signal = bool(ref.weight.grad.detach().half().ne(0).any() and layer.weight.grad.eq(0).all())
    gate = dict(protocol=PROTOCOL, passed=all(c["passed"] for c in checks.values()) and active > 0
                and dx_nonzero > 0 and not lost_weight_signal, input_gradient_nonzero=dx_nonzero,
                checks=checks, upstream_nonzero=nonzero, negative_inputs_with_nonzero_upstream=active,
                representable_slope_gradient_lost=lost_weight_signal,
                native_weight_gradient=float(layer.weight.grad.detach().item()) if finite_weight else None,
                reference_weight_gradient=float(ref.weight.grad.detach().item()),
                training_batch_size=capture_meta["training_batch_size"], timed_batch_size=size, loss_scale=1.0,
                paired_input_upstream_sha256=replay.hash_tensors([("x", tx), ("dy", tdy)]),
                reference="Native FP32 PReLU on the same FP16-representable input and unscaled loss derivative.")
    if not gate["passed"]:
        gate["reason"] = "No negative-site loss-gradient signal" if not active else "Native FP16 training-gradient gate failed"
    return op, gate

def decode_number(n):
    return float(n) if math.isfinite(n) else ("NaN" if math.isnan(n) else ("Infinity" if n > 0 else "-Infinity"))


def gradient_audit(torch, module, x, output):
    weight_grad = module.weight.grad
    require(weight_grad is not None and weight_grad.numel()==1 and x.grad is not None, "Missing native PReLU gradients")
    slope=float(weight_grad.detach().item())
    dx_nonfinite=int((~torch.isfinite(x.grad)).sum().item())
    return dict(output_finite=bool(torch.isfinite(output).all().item()),input_gradient_finite=dx_nonfinite==0,
                input_gradient_nonfinite_count=dx_nonfinite, input_gradient_dtype=str(x.grad.dtype),
                slope_gradient_finite=math.isfinite(slope),slope_gradient_nonfinite_count=int(not math.isfinite(slope)),
                slope_gradient=decode_number(slope),slope_gradient_dtype=str(weight_grad.dtype))
