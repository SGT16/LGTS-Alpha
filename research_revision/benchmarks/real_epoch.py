import copy
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import re
import threading
import time
from types import SimpleNamespace

from runtime import POLICY, ROOT, require, sha, write_json as durable_json

DATASETS={"cifar100":(32,50000),"dermamnist":(64,8010),"organsmnist":(64,16384)}
MODELS={"ResNet18","MobileNetV2","EfficientNetV2S"}
ACTIVATIONS={"Native","GELU","LGTS-f4-Alpha"}
BACKENDS={"eager","cudagraphs"}
INITIAL_SCALE=256.

class NumericExclusion(Exception):


    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def validate_unit(unit):
    size, images = DATASETS[unit["dataset"]]
    require(unit["model"] in MODELS and unit["activation"] in ACTIVATIONS
            and unit["backend"] in BACKENDS, "Unsupported epoch arm")
    require(unit["seed"] == 936137 and unit["batch_size"] == 128 and unit["image_size"] == size
            and unit["expected_images"] == images and unit["expected_steps"] == math.ceil(images / 128),
            "Unexpected seed, image geometry, sample count or batch policy")
    require(re.fullmatch(r"ep-[a-f0-9]{20}", unit["id"]) is not None, "Invalid real epoch unit id")
    for field in ("protocol_sha256", "dataset_fingerprint_sha256"):
        require(re.fullmatch(r"[a-f0-9]{64}", unit[field]) is not None, f"Invalid {field}")


def batch_shapes(images, batch_size):
    return list(dict.fromkeys(n for n in (min(images, batch_size), images % batch_size) if n))


class ObservedSampler:


    def __init__(self, source):
        self.source = source
        self.indices = array("I")

    def __iter__(self):
        for index in self.source:
            self.indices.append(index)
            yield index

    def __len__(self):
        return len(self.source)


def hash_tensors(items):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda entry: entry[0]):
        metadata = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(metadata).to_bytes(4, "big"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(raw)
    return digest.hexdigest()


def backbone_digest(model):
    from torch import nn
    kinds = (nn.Conv2d, nn.Linear, nn.modules.batchnorm._BatchNorm)
    items = [(f"{name}.{key}", value) for name, layer in model.named_modules() if isinstance(layer, kinds)
             for key, value in layer.state_dict().items()]
    require(items, "No backbone tensors")
    return hash_tensors(items)


def data_evidence(sampler, probes, images):
    indices = sampler.indices
    require(len(indices) == images and len(set(indices)) == images and min(indices) == 0
            and max(indices) == images - 1, "Full train split visits each example exactly once")
    return {"data_order_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "augmentation_probe_sha256": hash_tensors(probes.items())}


class StepProgress:


    def __init__(self, callback, total, guard):
        self.callback, self.total = callback, total
        self.guard = guard
        self.steps = 0
        self.failure = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.publish, name="real-epoch-progress", daemon=True)

    def publish(self):
        last_check = time.monotonic()
        try:
            while not self.stop.wait(8):
                self.callback("timing", self.steps, self.total)
                if time.monotonic() - last_check >= 30:
                    self.guard.verify()
                    last_check = time.monotonic()
        except Exception as error:
            self.failure = error

    def __enter__(self):
        self.callback("timing", 0, self.total)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop.set()
        self.thread.join()
        if exc is None and self.failure is not None:
            raise self.failure


def numerical_gate(runtime, cpu, cell, backend, shapes, evidence_path, progress):
    from calibration import _pair, select_scale


    trials = {}
    for precision, amp in (("float32", False), ("amp_float16", True)):
        def compare(batch, scale):
            progress("validating", 0, cell["expected_steps"])
            return _pair(runtime, cpu, cell, backend, batch, amp, scale, cell["source_seed"],
                         POLICY["tolerances"][precision])


        calibration = select_scale(compare, amp=amp, batches=shapes)
        trials[precision] = {key: value for key, value in calibration.items() if key != "checks"}
        if not calibration["passed"]:
            break
    passed = len(trials) == 2 and all(trial["passed"] for trial in trials.values())
    durable_json(evidence_path, {"passed": passed, "batches": shapes, "precisions": trials}, exclusive=True)
    summary = {"passed": passed, "evidence_file": evidence_path.name,
               "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(), "batches": shapes,
               "selected_check_scales": {key: value["selected_scale"] for key, value in trials.items()}}
    if not passed:
        raise NumericExclusion("Paired training equivalence failed", summary)
    return summary


def prepare(runtime, unit):
    import train_benchmark as engine
    from lgts_benchmark import data

    for module in (engine, data, runtime.models, runtime.activations):
        require(Path(module.__file__).resolve().is_relative_to(ROOT / "engine"), "Unpinned engine import")
    dataset, size, seed = unit["dataset"], unit["image_size"], unit["seed"]
    root = runtime.data_dir
    require((root / ("torchvision" if dataset == "cifar100" else "medmnist")).is_dir(), "Local dataset is absent")
    bundle = data.load_bundle(dataset, root, med_size=size, download=False)
    manifest = data.dataset_manifest(bundle)
    require(manifest["dataset_fingerprint_sha256"] == unit["dataset_fingerprint_sha256"], "Dataset fingerprint changed")
    transform, _ = data.build_transforms(dataset, size, bundle.in_channels, "standard", "dataset")
    train = data.TransformDataset(bundle.train_val, transform)
    cfg = SimpleNamespace(batch_size=128, num_workers=4, prefetch_factor=4, learning_rate=.001,
                          weight_decay=.0001, fused_optimizer=True, performance_mode=False, cudnn_benchmark=True,
                          channels_last=False, tf32=True)
    cell = {"id": f"epoch-gate:{dataset}:{unit['model']}:{seed}", "source_seed": seed, "dataset": dataset,
            "config": vars(cfg), "input_shape": [bundle.in_channels, size, size], "batch_size": unit["batch_size"],
            "expected_steps": unit["expected_steps"], "num_classes": len(bundle.class_names),
            "precision": {"channels_last": False, "tf32": True, "cudnn_benchmark": True}}
    runtime.set_policy(cell)
    loader = engine.loader(train, cfg, True, unit["batch_size"], data_seed=seed)
    require(len(train) == unit["expected_images"] and len(loader) == unit["expected_steps"], "Loader counts differ")
    sampler = ObservedSampler(loader.batch_sampler.sampler)
    loader.batch_sampler.sampler = sampler
    engine.seed_everything(seed, cfg.performance_mode, cfg.cudnn_benchmark)
    cpu = runtime.models.build_model(unit["model"], bundle.in_channels, cell["num_classes"], unit["activation"],
                                     lgts_alpha=.4, lgts_learn_alpha=True, lgts_learn_beta=False, pretrained=False,
                                     small_input_stem=True, activation_scope="full_backbone", activation_sharing="site")
    weights = None
    if engine.resolve_class_weight_mode("auto", dataset) == "balanced":
        weights = engine.balanced_weights(bundle.labels_train_val, cell["num_classes"], runtime.torch.device("cuda"))
    runtime.criterion = lambda _: runtime.torch.nn.CrossEntropyLoss(weight=weights)
    return engine, cpu, cell, loader, sampler


def one_step(rt, model, optimizer, scaler, criterion, x, y):
    optimizer.zero_grad(set_to_none=True)
    with rt.autocast(True):
        loss = criterion(model(x), y)
    if not rt.torch.isfinite(loss).item():
        raise NumericExclusion("Nonfinite real epoch loss", {"passed": False, "reason": "nonfinite_training_loss"})
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.detach().item() * len(y)


def warm_and_prove(rt, cpu, cell, backend, shapes, path, progress):

    torch = rt.torch
    model = copy.deepcopy(cpu).to("cuda").train()
    candidate = rt.compiler(model, backend)
    optimizer = rt.optimizer(model, cell)
    scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
    criterion = rt.criterion(cell)
    proofs = []
    for batch in shapes:
        rt.guard.verify()
        progress("warming", 0, cell["expected_steps"])
        x = rt.input(cell, batch)
        y = torch.arange(batch, device="cuda") % cell["num_classes"]

        def step():
            return one_step(rt, candidate, optimizer, scaler, criterion, x, y)

        for _ in range(3):
            step()
        if backend == "cudagraphs":
            evidence = rt.graph_evidence(step, backend, training=True)
            proofs.append({"batch": batch, **evidence})
    torch.cuda.synchronize()
    proof = {"passed": all(row["passed"] for row in proofs), "backend": backend, "required": backend != "eager",
             "batches": shapes}
    durable_json(path, {**proof, "captures": proofs}, exclusive=True)
    proof.update(evidence_file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if not proof["passed"]:
        raise NumericExclusion("Compiled forward/backward CUDA graph launches unverified", proof)
    model.load_state_dict(cpu.state_dict(), strict=True)
    model.zero_grad(set_to_none=True)
    return model, candidate, proof


def compilation_counters(torch):


    from torch._inductor.cudagraph_trees import get_manager
    manager = get_manager(0, create_if_none_exists=False)
    require(manager is not None, "No live CUDA graph tree for the warmed candidate")
    counters = torch._dynamo.utils.counters
    return {"dynamo_graphs": counters["stats"]["unique_graphs"], "skips": counters["inductor"]["cudagraph_skips"],
            "captured_graphs": sum(1 + n.num_descendants() for roots in manager.roots.values() for n in roots),
            "mutation_fallbacks": len(manager.warned_mutation)}


def measure(unit, output_dir, runtime, progress):

    from imagenet import validate_unit as validate_image, prepare as prepare_image
    is_image = unit["dataset"] == "imagenet100"
    (validate_image if is_image else validate_unit)(unit)
    guard = runtime.guard
    guard.verify()


    total = unit["expected_steps"]
    progress("preparing", 0, total)
    folder = Path(output_dir)
    torch = runtime.torch
    import numpy as np
    loader = None
    try:
        environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)}
        engine, cpu, cell, loader, sampler = (prepare_image if is_image else prepare)(runtime, unit)


        rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        initial = hash_tensors(cpu.state_dict().items())
        backbone = backbone_digest(cpu)
        backend = unit["backend"]
        shapes = batch_shapes(unit["expected_images"], unit["batch_size"])
        progress("validating", 0, total)
        gate = matched_gate(runtime, cpu, cell, backend, shapes, folder / "training_equivalence.json", progress) if INITIAL_SCALE == 256 else numerical_gate(runtime, cpu, cell, backend, shapes, folder / "training_equivalence.json", progress)
        runtime.captures.clear()
        model, candidate, proof = warm_and_prove(runtime, cpu, cell, backend, shapes, folder / "graph_evidence.json", progress)
        require(hash_tensors(model.state_dict().items()) == initial, "Warm-up restoration changed initial weights or buffers")
        optimizer = runtime.optimizer(model, cell)

        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda", init_scale=INITIAL_SCALE)
        criterion = runtime.criterion(cell)
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state(rng[1])
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        guard.verify()
        torch.cuda.synchronize()
        graphs_before = compilation_counters(torch) if backend == "cudagraphs" else None
        images, loss_sum, probes = 0, 0.0, {}
        from lgts_benchmark.numerics import NumericalError, validate_model_state
        with StepProgress(progress, total, guard) as report:
            started = time.perf_counter()
            model.train()
            for i, (x, y) in enumerate(loader, 1):
                if i == 1 or i == total:
                    probes[f"{i}/images"], probes[f"{i}/labels"] = x, y
                x, y = engine.move_batch(x, y, torch.device("cuda"), False)
                loss_sum += one_step(runtime, candidate, optimizer, scaler, criterion, x, y)
                images += len(y)
                report.steps = i
                if report.failure:
                    raise report.failure
            try:
                validate_model_state(model)
            except NumericalError as error:
                raise NumericExclusion(str(error), {"passed": False, "reason": "invalid_final_model"}) from error
            runtime.activations.lgts_parameter_snapshot(model)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
        guard.verify()
        progress("checking", report.steps, total)
        require(report.steps == total and images == len(loader.dataset), "Incomplete epoch")
        require(math.isfinite(elapsed) and elapsed > 0 and math.isfinite(loss_sum), "Invalid epoch timing or loss")
        updates = {int(state["step"].item()) for state in optimizer.state.values() if "step" in state}
        if len(updates) != 1 or not 0 < next(iter(updates)) <= total:
            raise NumericExclusion("Epoch did not produce consistent AdamW steps", {"passed": False, "update_counts": sorted(updates)})
        if backend == "cudagraphs":
            graphs_after = compilation_counters(torch)
            if graphs_before != graphs_after:
                raise NumericExclusion("Compilation, recording or fallback occurred inside the measured epoch",
                                       {"passed": False, "before": graphs_before, "after": graphs_after})
            proof.update(before_epoch=graphs_before, after_epoch=graphs_after)
        return {**unit, "status": "measured", "measured": True, "epoch": 1, "epoch_count": 1,
                "seconds": elapsed, "images": images, "steps": report.steps, "initial_state_sha256": initial,
                "backbone_sha256": backbone, **data_evidence(sampler, probes, images),
                "initial_scale": INITIAL_SCALE, "amp_skipped_steps": total - next(iter(updates)),
                "final_scale": float(scaler.get_scale()), "mean_training_loss_diagnostic_only": loss_sum / images,
                "proof": proof, "numerical_gate": gate, "environment": environment}
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


def matched_gate(runtime, cpu, cell, backend, shapes, path, progress):
    from calibration import _pair


    checks = []
    for batch in shapes:
        progress("validating", 0, cell["expected_steps"])
        trial = _pair(runtime, cpu, cell, backend, batch, True, INITIAL_SCALE,
                      cell["source_seed"], POLICY["tolerances"]["amp_float16"])
        checks.append(trial)
        if not trial["passed"] or not trial["safe_step"]:
            durable_json(path, dict(passed=False, initial_scale=INITIAL_SCALE, trials=checks), exclusive=True)
            raise NumericExclusion("AMP equivalence failed at the prespecified control scale",
                                   dict(passed=False, evidence_file=path.name, sha256=sha(path)))
    extra = path.with_name("control_scale_equivalence.json")
    durable_json(extra, dict(passed=True, initial_scale=INITIAL_SCALE, trials=checks), exclusive=True)
    gate = numerical_gate(runtime, cpu, cell, backend, shapes, path, progress)
    return dict(gate, control_scale_sha256=sha(extra))
