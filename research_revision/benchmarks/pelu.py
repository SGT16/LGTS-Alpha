import copy
from itertools import islice
import math
from pathlib import Path
from types import SimpleNamespace
from runtime import ROOT, POLICY, KINDS, require, sha
from real_epoch import ObservedSampler, hash_tensors

BATCHES=8
INPUT_FILE="observed_input.pt"
PROTOCOL="pelu_learned_checkpoint_train_loader_8_batches_v1"

def training_loader(rt, cell):
    import train_benchmark as engine
    from lgts_benchmark import data

    for module in (engine, data, rt.models, rt.activations):
        require(Path(module.__file__).resolve().is_relative_to(ROOT / "engine"),
                        "Training replay imported an unsealed engine")
    cfg = SimpleNamespace(**cell["config"])
    require(cfg.protocol == "full_train" and not cfg.no_amp, "Unsupported source training protocol")
    if cell["dataset"] == "imagenet100":
        from imagenet import readonly_bundle
        bundle = readonly_bundle(rt.data_dir)
    else:
        bundle = data.load_bundle(cell["dataset"], rt.data_dir, med_size=cfg.med_size, download=False)
    manifest = data.dataset_manifest(bundle)
    require(sha(cell["dataset_manifest_path"]) == cell["dataset_manifest_sha256"]
                    and manifest["dataset_fingerprint_sha256"] == cell["dataset_fingerprint_sha256"],
                    "Replay dataset differs from the sealed training data")
    size = engine.resolve_image_size(cfg, bundle.name, bundle.native_size)
    batch = engine.resolve_batch_size(cfg, bundle.name)
    require([bundle.in_channels, size, size] == cell["input_shape"] and batch == cell["batch_size"],
                    "Replay images or batch size differ from source training")
    transform, _ = data.build_transforms(bundle.name, size, bundle.in_channels, cfg.augmentation, cfg.normalization)
    dataset = data.TransformDataset(bundle.train_val, transform)
    loader = engine.loader(dataset, cfg, True, batch, data_seed=cell["source_seed"])
    sampler = ObservedSampler(loader.batch_sampler.sampler)
    loader.batch_sampler.sampler = sampler
    engine.seed_everything(cell["source_seed"], cfg.performance_mode, cfg.cudnn_benchmark)
    return engine, loader, sampler


def capture_identity(cell, unit, signature):
    require(unit["cell_id"] == cell["id"] and unit["activation"] == cell["activation"] == "PELU"
                    and unit["checkpoint_sha256"] == cell["checkpoint_sha256"]
                    and unit["dataset"] == cell["dataset"] and unit["source_seed"] == cell["source_seed"]
                    and unit["parameter_state"] == "learned_checkpoint" and cell["precision"]["autocast"] == "float16"
                    and unit["dtype"] == "torch.float16" and unit["shape"][0] == unit["batch_size"] == cell["batch_size"],
                    "Unexpected observed-range microbenchmark")
    return dict(protocol=PROTOCOL, plan_sha256=signature, cell_id=cell["id"],
                checkpoint_sha256=cell["checkpoint_sha256"], dataset=cell["dataset"], model=unit["model"],
                dataset_fingerprint_sha256=cell["dataset_fingerprint_sha256"], source_seed=cell["source_seed"],
                activation_site=unit["activation_site"], shape=unit["shape"], dtype=unit["dtype"],
                tensor_file=INPUT_FILE)


def sample_input(rt, model, cell, unit, identity, progress):
    torch = rt.torch
    engine, loader, sampler = training_loader(rt, cell)
    source = model.get_submodule(unit["activation_site"])
    batch_stats, observed = [], []
    first = None

    def capture_input(_module, args):
        nonlocal first
        x = args[0].detach()
        require(list(x.shape) == unit["shape"] and str(x.dtype) == unit["dtype"],
                        "Observed activation shape or dtype changed")
        lo, hi = torch.aminmax(x)
        limits = [float(lo), float(hi)]
        require(all(math.isfinite(v) for v in limits), "Nonfinite observed training activation")
        observed.append(limits)
        if first is None:
            first = x.to("cpu", copy=True)

    hook = source.register_forward_pre_hook(capture_input)
    try:
        with torch.no_grad(), rt.autocast(True):
            for index, (images, labels) in enumerate(islice(loader, BATCHES)):
                progress()
                rt.guard.verify()
                observed.clear()
                batch = len(labels)
                indices = list(sampler.indices[index * batch:(index + 1) * batch])
                require(batch == unit["batch_size"] and len(indices) == batch, "Incomplete training batch")
                batch_digest = hash_tensors([("images", images), ("labels", labels)])
                x, _ = engine.move_batch(images, labels, torch.device("cuda"), cell["precision"]["channels_last"])
                model(x)
                require(len(observed) == 1, "Training site must execute exactly once per batch")
                low, high = observed[0]
                batch_stats.append(dict(batch=index + 1, images=batch, minimum=low, maximum=high,
                                        training_indices=indices, transformed_batch_sha256=batch_digest))
    finally:
        hook.remove()

        del loader
    require(len(batch_stats) == BATCHES and first is not None, "Incomplete real training replay")
    cfg = cell["config"]
    meta = dict(**identity, batches=batch_stats, sampled_images=BATCHES * cell["batch_size"],
                minimum=min(b["minimum"] for b in batch_stats), maximum=max(b["maximum"] for b in batch_stats),
                timed_batch=1, input_sha256=hash_tensors([("activation_input", first)]), split="train_val",
                augmentation=cfg["augmentation"], normalization=cfg["normalization"], num_workers=cfg["num_workers"],
                prefetch_factor=cfg["prefetch_factor"], autocast="float16", model_mode="train",
                parameter_updates=0, input_origin="Forward pre-hook on a frozen learned checkpoint replaying real training batches",
                range_scope="First eight full shuffled batches of this replay; historical epoch extrema were not recorded.")
    return {"metadata": meta, "tensor": first}


def compare(rt, module, x, *, backward=True):

    torch = rt.torch
    results = []
    for dtype in (torch.float32, torch.float16):
        candidate = copy.deepcopy(module).to(x.device)
        candidate.zero_grad(set_to_none=True)
        value = x.detach().to(dtype=dtype).requires_grad_(backward)
        with rt.autocast(dtype == torch.float16):
            output = candidate(value)
        observed = dict(output=output.detach())
        if backward:
            output.backward(torch.ones_like(output))
            observed["input_gradient"] = value.grad
            observed.update({name: p.grad for name, p in candidate.named_parameters()})
        results.append(observed)
    tol = POLICY["tolerances"]["amp_float16"]
    checks = {name: rt.error_metrics(reference, results[1][name], tol) for name, reference in results[0].items()}
    return dict(passed=all(v["passed"] for v in checks.values()), tensors=checks)


def operation(rt, module, tensor, cell, unit, evidence):

    torch = rt.torch
    rt.set_policy(cell)
    x = tensor.detach().cuda().requires_grad_()
    extrema = [value for row in evidence["batches"] for value in (row["minimum"], row["maximum"])]
    sample = torch.tensor(extrema, dtype=x.dtype, device=x.device)
    check_gradients = unit["kind"] != KINDS[0]
    tests = dict(captured_timing_tensor=compare(rt, module, x, backward=check_gradients),
                 measured_batch_extrema=compare(rt, module, sample, backward=check_gradients))
    gate = dict(passed=all(v["passed"] for v in tests.values()), protocol=PROTOCOL, checks=tests,
                minimum=evidence["minimum"], maximum=evidence["maximum"], input_sha256=evidence["input_sha256"],
                reference="Eager FP32 PELU on the identical FP16-representable inputs; unit upstream gradient")
    layer = copy.deepcopy(module).cuda()
    with rt.autocast(True):
        fixed = layer(x)
    delta = torch.ones_like(fixed)

    def forward():
        with torch.no_grad(), rt.autocast(True):
            return layer(x)

    def backward():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        fixed.backward(delta, retain_graph=True)

    def combined():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with rt.autocast(True):
            y = layer(x)
        y.backward(delta)

    op = dict(zip(KINDS, (forward, backward, combined)))[unit["kind"]]
    op()
    finite = {"output": torch.isfinite(fixed).all().item()}
    if unit["kind"] != KINDS[0]:
        finite["input_gradient"] = x.grad is not None and torch.isfinite(x.grad).all().item()
        finite.update({name: p.grad is not None and torch.isfinite(p.grad).all().item()
                       for name, p in layer.named_parameters()})
    gate.update(workload_finite=finite, passed=gate["passed"] and all(finite.values()))
    return op, gate
