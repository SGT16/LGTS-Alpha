import copy
import gc
import hashlib
import json
import math
import importlib.util
import statistics
from pathlib import Path
import random
import sys
import time

from native_batch_norm import compile_builtin_training_graph
from calibration import training_equivalence as calibrated_training_equivalence, training_scale

ROOT = Path(__file__).resolve().parents[1]
FILES = Path(__file__).resolve().parent / "protocol_data"
sys.path.insert(0, str(ROOT / "engine"))
POLICY = {"order_seed": 20261102, "fullgraph": True, "dynamic": False,
          "tolerances": {"float32": {"atol": 1e-5, "rtol": 1e-4},
                         "amp_float16": {"atol": .005, "rtol": .05}}}
KINDS = ("activation_forward", "activation_backward_retained_graph", "activation_forward_plus_backward")

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def write_json(path, value, *, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)

def read(path):
    return json.loads(Path(path).read_bytes())

class DeviceCheck:
    def __init__(self, torch):
        self.torch = torch
        require(torch.cuda.is_available(), "CUDA is required")
        self.device = torch.cuda.current_device()
        require(self.device == 0, "Select one CUDA device before starting")
    def verify(self):
        require(self.torch.cuda.current_device() == self.device, "CUDA device changed")

def configure_torch_threads(torch):

    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.set_num_threads(1)


def class_weight_mode(cell):
    requested = cell["config"]["class_weight"]
    expected = ("balanced" if cell["dataset"] in ("pathmnist", "dermamnist", "organsmnist") else "none") if requested == "auto" else requested
    require(expected in ("none", "balanced"), "Unknown class-weight mode")
    require(cell.get("resolved_class_weight") == expected, "Pinned resolved class weighting disagrees with sealed engine rule")
    return expected


def checkpoint_metadata(payload, cell):
    require(isinstance(payload, dict) and payload.get("format_version") == 1, "Wrong final checkpoint format")
    require(payload.get("purpose") == "final_evaluation_and_mechanistic_analysis", "Wrong checkpoint purpose")
    meta = payload.get("metadata", {})
    for key, expected in (("dataset", cell["dataset"]), ("model", cell["model"]),
                          ("seed", cell["source_seed"]), ("epochs_run", cell["source_epochs"]),
                          ("code_sha256", cell["engine_code_sha256"]),
                          ("dataset_fingerprint_sha256", cell["dataset_fingerprint_sha256"]),
                          ("activation_scope", cell["activation_scope"]), ("protocol", "full_train")):
        require(meta.get(key) == expected, "Checkpoint metadata mismatch: " + key)
    norm = lambda s: str(s).replace("-", "").replace("_", "").lower()
    require(norm(meta.get("activation")) == norm(cell["activation"]), "Checkpoint activation mismatch")
    require(isinstance(payload.get("state_dict"), dict), "Missing checkpoint state_dict")
    cfg = meta.get("configuration", {})
    for key in ("activation_sharing", "lgts_alpha_min", "lgts_alpha_max", "lgts_beta_min", "lgts_beta_max",
                "lgts_beta", "no_amp", "optimizer", "weight_decay", "scheduler", "small_input_stem",
                "augmentation", "epochs"):
        require(cfg.get(key) == cell["config"].get(key), "Checkpoint config mismatch: " + key)
    return payload["state_dict"]


def aggregate(values):
    require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), "Invalid timing repeats")
    ordered = sorted(values)
    def q(p):
        pos = (len(ordered)-1)*p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)
    return {"repeat_count": len(values), "mean": statistics.mean(values),
            "sample_standard_deviation": statistics.stdev(values) if len(values)>1 else None,
            "median": statistics.median(values), "q25": q(.25), "q75": q(.75), "iqr": q(.75)-q(.25)}

def direction(event):
    parent = event.cpu_parent
    while parent is not None:
        name = parent.name.casefold()
        if "backward" in name:
            return "backward"
        if name == "alpha_proof_model_forward":
            return "forward"
        parent = parent.cpu_parent
    return "unassigned"

class Runtime:
    def __init__(self, output, data_dir):
        import torch
        from lgts_benchmark import activations, models
        self.torch, self.guard = torch, DeviceCheck(torch)
        self.output = Path(output)
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.activations, self.models = activations, models
        configure_torch_threads(torch)
        self.rows, self.gaps, self.captures = [], [], []
        self.variant, self.profiling = "cudagraphs", False
        self.manifest = {"sampling_protocol": POLICY}
        self.measurement_runner_sha256 = sha(__file__)
        self.journal = (self.output / "blocks.jsonl").open("x", encoding="utf-8")
    def load(self, cell):
        torch = self.torch
        self.guard.verify()
        require(sha(cell["checkpoint_path"]) == cell["checkpoint_sha256"], "Checkpoint changed before load")
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            payload = torch.load(cell["checkpoint_path"], map_location="cpu", weights_only=True)
        state = checkpoint_metadata(payload, cell)
        require(all(not torch.is_tensor(v) or torch.isfinite(v).all().item() for v in state.values()), "Nonfinite checkpoint tensor")
        model = self.models.build_model(**cell["model_build_kwargs"])
        model.load_state_dict(state, strict=True)
        counts = self.models.count_parameters(model)
        require(counts == (cell["expected_total_parameters"], cell["expected_trainable_parameters"]), "Parameter-count mismatch")
        return model, state


    def set_policy(self, cell):
        torch = self.torch
        torch.manual_seed(POLICY["order_seed"])
        torch.cuda.manual_seed_all(POLICY["order_seed"])
        torch.backends.cuda.matmul.allow_tf32 = bool(cell["precision"]["tf32"])
        torch.backends.cudnn.allow_tf32 = bool(cell["precision"]["tf32"])
        torch.backends.cudnn.benchmark = bool(cell["precision"]["cudnn_benchmark"])
        torch.backends.cudnn.deterministic = not bool(cell["config"]["performance_mode"])
        torch.set_float32_matmul_precision("high" if cell["precision"]["tf32"] else "highest")
        torch._dynamo.config.suppress_errors = False


    def autocast(self, enabled):
        return self.torch.autocast(device_type="cuda", dtype=self.torch.float16, enabled=enabled)


    def input(self, cell, batch, *, requires_grad=False):
        torch = self.torch
        seed = int.from_bytes(hashlib.sha256((cell["id"]+":"+str(batch)).encode()).digest()[:8], "big")
        generator = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn((batch, *cell["input_shape"]), device="cuda", dtype=torch.float32, generator=generator)
        if cell["precision"]["channels_last"]:
            x = x.contiguous(memory_format=torch.channels_last)
        return x.requires_grad_(requires_grad)


    def error_metrics(self, reference, observed, tolerance):
        torch = self.torch
        require(reference.shape == observed.shape, "Equivalence tensor shape mismatch")
        a, b = reference.detach().float(), observed.detach().float()
        finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
        if not finite:
            return {"passed": False, "finite": False, "elements": a.numel()}
        error = (a-b).abs().reshape(-1)


        sample = error[::max(1, error.numel()//100000)]
        bound = tolerance["atol"] + tolerance["rtol"]*a.abs()
        return {"passed": bool((error.reshape(a.shape) <= bound).all().item()), "finite": True,
                "elements": a.numel(), "max_absolute_error": float(error.max().item()) if error.numel() else 0.,
                "p99_absolute_error_deterministic_sample": float(torch.quantile(sample, .99).item()) if sample.numel() else 0.,
                "max_tolerance_ratio": float(((a-b).abs()/bound).max().item()) if a.numel() else 0.,
                "failed_elements": int(((a-b).abs()>bound).sum().item()), **tolerance}


    def optimizer(self, model, cell):
        torch = self.torch
        cfg = cell["config"]
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if p.requires_grad:
                (no_decay if p.ndim <= 1 or name.endswith("bias") else decay).append(p)
        return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                  {"params": no_decay, "weight_decay": 0.}],
                                 lr=cfg["learning_rate"], fused=cfg["fused_optimizer"])


    def criterion(self, cell):
        torch = self.torch
        manifest = json.loads(Path(cell["dataset_manifest_path"]).read_text(encoding="utf-8-sig"))
        balanced = class_weight_mode(cell) == "balanced"
        weights = None
        if balanced:
            counts = [a+b for a,b in zip(manifest["train_class_counts"], manifest["validation_class_counts"])]
            require(all(n > 0 for n in counts), "Empty weighted training class")
            weights = torch.tensor([sum(counts)/(len(counts)*n) for n in counts], device="cuda", dtype=torch.float32)
        return torch.nn.CrossEntropyLoss(weight=weights)


    def training_equivalence(self, cpu_model, cell, backend):
        return calibrated_training_equivalence(self, cpu_model, cell, backend)


    def timer(self, operation, cell, backend, kind, batch, *, iterations=100, warmup=50,
              blocks=10, minimum_seconds=0., extra=None, state_probe=None):
        torch = self.torch
        source_version = getattr(self, "measurement_runner_sha256", None) or sha(Path(__file__))
        self.guard.verify()
        begin = time.perf_counter()
        for _ in range(warmup): operation()
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter()-begin
        torch.cuda.reset_peak_memory_stats()
        baseline = {"baseline_allocated_bytes": torch.cuda.memory_allocated(),
                    "baseline_reserved_bytes": torch.cuda.memory_reserved()}
        previous_state = state_probe() if state_probe else None
        rows = []
        for block in range(blocks):
            self.guard.verify()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            count = 0; started = time.perf_counter(); start.record()


            while count < iterations or time.perf_counter()-started < minimum_seconds:
                operation(); count += 1
            end.record(); torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
            elapsed_wall = time.perf_counter()-started
            observed_state = state_probe() if state_probe else None
            self.guard.verify()
            row = {"cell_id": cell["id"], "backend": backend, "kind": kind,
                   "source_measurement_runner_sha256": source_version,
                   "reused_measurement": False,
                   "backend_attempt_id": getattr(self, "backend_attempt_id", None),
                   "batch_size": batch, "block": block+1, "iterations": count,
                   "total_cuda_ms": elapsed_ms, "per_operation_ms": elapsed_ms/count,
                   "wall_seconds": elapsed_wall,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **baseline,
                   "warmup_iterations": warmup, "warmup_seconds": warmup_seconds,
                   "memory_scope": "Absolute/incremental relative to this cell after warmup; model/compiler caches from prior batches may remain; not isolated-process batch memory",
                   "state_before": previous_state, "state_after": observed_state,
                   "precision": cell["precision"]["autocast"], **(extra or {})}
            rows.append(row); self.rows.append(row)
            row["incremental_peak_allocated_bytes"] = row["peak_allocated_bytes"]-baseline["baseline_allocated_bytes"]
            row["incremental_peak_reserved_bytes"] = row["peak_reserved_bytes"]-baseline["baseline_reserved_bytes"]
            self.journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            self.journal.flush()
            if state_probe:
                require(observed_state["loss_finite"], "Nonfinite synthetic loss: raw timing preserved, no accepted speed result")
                require(observed_state["amp_scale"] >= previous_state["amp_scale"], "AMP skipped steps in timed block: raw timing preserved, no accepted speed result")
                previous_state = observed_state
        return {"kind": kind, "batch_size": batch, "summary_ms": aggregate([r["per_operation_ms"] for r in rows]), "raw_repeats": rows}


    def measure_training_component(self, cpu_model, state, cell, backend, folder):

        torch = self.torch

        gate = self.training_equivalence(cpu_model, cell, backend)
        write_json(folder / "training_step_equivalence.json", gate)
        if not gate["passed"]:
            proof = {"passed": False, "reason": "Calibrated training equivalence gate failed"}
            result = {"status": "optimization_not_verified_no_training_timing", "proof": proof}
            write_json(folder / "training_optimization_evidence.json", proof)
            write_json(folder / "training_results.json", result)
            self.gaps.append({"cell_id":cell["id"],"backend":backend,"reason":proof["reason"]})
            return gate, result
        amp = cell["precision"]["autocast"] == "float16"
        initial_scale = training_scale(gate, amp=amp)
        model = candidate = optimizer = scaler = criterion = x = y = last_loss = None
        try:
            torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
            model = copy.deepcopy(cpu_model).to("cuda")
            if cell["precision"]["channels_last"]:
                model = model.to(memory_format=torch.channels_last)
            candidate = self.compiler(model.train(), backend)
            x = self.input(cell, cell["batch_size"])
            y = torch.arange(cell["batch_size"], device="cuda") % cell["num_classes"]
            optimizer = self.optimizer(model, cell)
            scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=initial_scale)
            criterion = self.criterion(cell)
            def step():
                nonlocal last_loss
                optimizer.zero_grad(set_to_none=True)
                with self.autocast(amp): loss = criterion(candidate(x), y)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                self.activations.project_activation_parameters_(model)
                last_loss = loss.detach()
            random.seed(POLICY["order_seed"])
            torch.manual_seed(POLICY["order_seed"]); torch.cuda.manual_seed_all(POLICY["order_seed"])
            started = time.perf_counter()
            for _ in range(3): step()
            torch.cuda.synchronize()
            first_calls = time.perf_counter() - started
            proof = self.graph_evidence(step, backend, training=True)
            write_json(folder / "training_optimization_evidence.json", proof)
            if proof["passed"]:
                model.load_state_dict(state, strict=True)
                optimizer = self.optimizer(model, cell)
                scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=initial_scale)
                random.seed(POLICY["order_seed"])
                torch.manual_seed(POLICY["order_seed"]); torch.cuda.manual_seed_all(POLICY["order_seed"])

                result = self.timer(step, cell, backend, "synthetic_training_step", cell["batch_size"],
                    extra={"optimizer":"AdamW", "projection_included":True, "data_loading_included":False,
                           "per_step_finite_validator_included":False,
                           "definition_note":"Forward/loss/backward/scaler/optimizer/projection/zero_grad only; engine gradient logging and epoch-level finite validation excluded; finite loss checked between blocks",
                           "state_policy":"Disposable state; repeated steps; never saved to training results",
                           "calibrated_initial_loss_scale":initial_scale,
                           "calibration_in_timing":False},
                    state_probe=lambda:{"amp_scale":scaler.get_scale(),"loss_finite":bool(torch.isfinite(last_loss).item())})
                result.update(amp_scale_before=initial_scale,amp_scale_after=scaler.get_scale(),
                              first_three_training_calls_seconds=first_calls,
                              calibrated_initial_loss_scale=initial_scale,
                              source_measurement_runner_sha256=self.measurement_runner_sha256)
            else:
                result = {"status":"optimization_not_verified_no_training_timing","proof":proof}
                self.gaps.append({"cell_id":cell["id"],"backend":backend,"reason":"Training forward/backward optimization not verified"})
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite disposable model after timing")
            write_json(folder / "training_results.json", result)
            return gate, result
        finally:
            model = candidate = optimizer = scaler = criterion = x = y = last_loss = None
            gc.collect(); torch.cuda.empty_cache()

    def compiler(self, model, backend):
        if backend == "eager":
            return model
        require(backend == self.variant, "Unexpected backend")


        self.torch._dynamo.reset()
        gc.collect()
        self.torch.cuda.empty_cache()
        event = {}
        self.captures.append(event)
        candidate = compile_builtin_training_graph(self.torch, model, event)

        def invoke(x):
            if self.profiling:
                with self.torch.profiler.record_function("alpha_proof_model_forward"):
                    return candidate(x)
            return candidate(x)
        return invoke


    def graph_evidence(self, operation, backend, *, training=False):
        if backend == "eager":
            return {"required": False, "passed": True}
        require(training and backend == self.variant, "Training-only profiler")
        torch = self.torch
        self.profiling = True
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                operation()
                torch.cuda.synchronize()
        finally:
            self.profiling = False
        events = list(prof.events())
        launches = [e for e in events if "cudagraphlaunch" in e.name.casefold()]
        directions = [direction(e) for e in launches]
        names = sorted({e.name for e in events})
        compiled = [n for n in names if "torch-compiled region" in n.casefold() or "compiledfunction" in n.casefold()]
        return dict(required=True, passed=bool(compiled) and {"forward", "backward"} <= set(directions),
                    backend=backend, cuda_graph_launch_count=len(launches), launch_directions=directions,
                    compiled_regions_observed_in_this_call=compiled, event_names=names,
                    training_forward_backward_evidence_required=True, compiler_evidence=self.captures)


    def activation_sites(self, model, cell):
        torch = self.torch
        allowed = (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU, torch.nn.GELU,
                   torch.nn.ELU, torch.nn.PReLU, torch.nn.Mish, torch.nn.LeakyReLU)
        rows, handles = [], []
        def hook(name):
            def capture(module, args, output):
                if args and torch.is_tensor(args[0]):
                    rows.append((name, module, list(args[0].shape), str(args[0].dtype)))
            return capture
        for name, module in model.named_modules():
            if isinstance(module, allowed) or module.__class__.__module__ == self.activations.__name__:
                if not list(module.children()):
                    handles.append(module.register_forward_hook(hook(name)))
        try:
            model.eval()
            with torch.inference_mode(), self.autocast(cell["precision"]["autocast"] == "float16"):
                model(self.input(cell, cell["batch_size"]))
        finally:
            for handle in handles: handle.remove()
        require(rows, "No eligible activation sites captured")


        selected = [rows[0]] + ([rows[-1]] if rows[-1][0] != rows[0][0] else [])
        return [(name, copy.deepcopy(module).cpu(), shape, dtype) for name,module,shape,dtype in selected]


    def activation_equivalence(self, source, backend):
        torch = self.torch
        checks = []
        for label, dtype in (("float32", torch.float32), ("amp_float16", torch.float16)):
            a, b = copy.deepcopy(source).to("cuda"), copy.deepcopy(source).to("cuda")
            if hasattr(a, "inplace"): a.inplace = False
            if hasattr(b, "inplace"): b.inplace = False
            compiled = self.compiler(b, backend)

            values = torch.tensor([-60000., -1000., -10., -1., -.0001, 0., .0001, 1., 10., 1000., 60000.], device="cuda", dtype=dtype)
            channels = max((p.numel() for p in a.parameters()), default=1)
            x = values.reshape(1, 1, 1, -1).expand(1, channels, 1, -1).clone().requires_grad_()
            y = x.detach().clone().requires_grad_()
            with self.autocast(label == "amp_float16"):
                oa, ob = a(x), compiled(y)

            oa.float().mean().backward(); ob.float().mean().backward()
            tol = POLICY["tolerances"][label]
            entry = {"precision": label, "output": self.error_metrics(oa, ob, tol),
                     "input_dtype": str(dtype), "autocast_enabled": label == "amp_float16",
                     "input_gradient": self.error_metrics(x.grad, y.grad, tol), "parameter_gradients": []}
            for (an, ap), (bn, bp) in zip(a.named_parameters(), b.named_parameters()):
                require(an == bn, "Activation parameter names differ")
                check = {"passed": ap.grad is None and bp.grad is None} if ap.grad is None or bp.grad is None else self.error_metrics(ap.grad, bp.grad, tol)
                entry["parameter_gradients"].append({"name": an, **check})
            entry["passed"] = entry["output"]["passed"] and entry["input_gradient"]["passed"] and all(p["passed"] for p in entry["parameter_gradients"])
            checks.append(entry)
            del a, b, compiled, x, y, oa, ob
        gradcheck = {"status": "not_custom_squareplus_backward", "passed": True}
        if isinstance(source, (self.activations.AlgebraicBaseLGTS, self.activations.AlgebraicLogisticLGTS)):
            module = copy.deepcopy(source).cpu().double()
            params = dict(module.named_parameters())
            channels = max((p.numel() for p in params.values()), default=1)
            gx = torch.linspace(-.7, .8, channels*2, dtype=torch.float64).reshape(1,channels,1,2).requires_grad_()
            names = list(params)
            def call(value, *parameters):
                return torch.func.functional_call(module, dict(zip(names,parameters)), (value,))
            try:
                passed = torch.autograd.gradcheck(call, (gx,*params.values()), eps=1e-6, atol=1e-4,
                                                 rtol=1e-3, fast_mode=True, raise_exception=True)
                gradcheck = {"status": "float64_input_and_parameter_directional_gradcheck", "passed": bool(passed),
                             "fast_mode": True, "epsilon": 1e-6, "atol": 1e-4, "rtol": 1e-3,
                             "note": "Directional fast-mode gradcheck away from alpha-zero kink; no smoothness claim at kink"}
            except Exception as exc:
                gradcheck = {"status": "failed", "passed": False, "error": str(exc)[:4000]}
            del module, params, gx
        return {"passed": all(e["passed"] for e in checks) and gradcheck["passed"], "checks": checks,
                "float64_gradcheck": gradcheck,
                "input_regions": "negative tail, near zero, positive tail, finite large magnitude"}

class InferenceRuntime(Runtime):
    def compiler(self, model, backend):
        torch = self.torch
        if backend == "eager":
            return model
        if backend == "inductor" and importlib.util.find_spec("triton") is None:
            raise RuntimeError("UNSUPPORTED: TorchInductor CUDA requires Triton, which is not installed; no installation performed")
        return torch.compile(model, backend=backend, fullgraph=POLICY["fullgraph"], dynamic=POLICY["dynamic"])


    def graph_evidence(self, operation, backend, *, training=False):
        if backend == "eager": return {"required": False, "passed": True}
        torch = self.torch
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            operation(); torch.cuda.synchronize()
        events = list(prof.events())
        names = sorted({e.name for e in events})
        graph_launches = [name for name in names if "cudagraphlaunch" in name.lower()]
        launch_count = sum("cudagraphlaunch" in e.name.lower() for e in events)
        compiled_regions = [name for name in names if "torch-compiled region" in name.lower() or "compiledfunction" in name.lower()]
        counters = {str(k): dict(v) for k, v in torch._dynamo.utils.counters.items()}
        if backend == "cudagraphs":
            passed = launch_count >= (2 if training else 1)
        else:
            passed = bool(compiled_regions) and (not training or any("backward" in name.lower() for name in compiled_regions))
        return {"required": True, "passed": passed, "cuda_graph_launch_events": graph_launches,
                "cuda_graph_launch_count": launch_count, "training_forward_backward_evidence_required": training,
                "compiled_regions_observed_in_this_call": compiled_regions,
                "dynamo_counters": counters, "event_names": names,
                "note": "Cudagraph backend requires actual cudaGraphLaunch evidence; wrapper creation alone is insufficient"}


    def equivalence(self, eager, candidate, cell, backend):

        torch = self.torch
        eager.eval(); candidate.eval()
        result = {"backend": backend, "model": [], "passed": True}
        for label, amp in (("float32", False), ("amp_float16", True)):
            eager.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
            source = self.input(cell, min(2, cell["batch_size"]))
            xa = source.detach().clone().requires_grad_(); xb = source.detach().clone().requires_grad_()
            with self.autocast(amp):
                ya = eager(xa); yb = candidate(xb)
                la, lb = ya.float().square().mean(), yb.float().square().mean()
            la.backward(); lb.backward()
            tol = POLICY["tolerances"][label]
            entry = {"precision": label, "output": self.error_metrics(ya, yb, tol),
                     "input_gradient": self.error_metrics(xa.grad, xb.grad, tol), "parameter_gradients": []}

            base = getattr(candidate, "_orig_mod", candidate)
            left, right = dict(eager.named_parameters()), dict(base.named_parameters())
            require(set(left) == set(right), "Equivalence parameter identity mismatch")
            for name in left:
                a, b = left[name].grad, right[name].grad
                if a is None or b is None:
                    check = {"passed": a is None and b is None, "both_unused": a is None and b is None}
                else:
                    check = self.error_metrics(a, b, tol)
                entry["parameter_gradients"].append({"name": name, **check})
            passed = entry["output"]["passed"] and entry["input_gradient"]["passed"] and all(p["passed"] for p in entry["parameter_gradients"])
            entry["passed"] = passed; result["passed"] &= passed; result["model"].append(entry)
            del xa, xb, ya, yb, la, lb, source
        eager.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
        return result

    def inference(self, cell, backend, cpu_model, state, folder, scaled_gate=False):
        torch=self.torch
        torch._dynamo.utils.counters.clear()
        model = copy.deepcopy(cpu_model).to("cuda")
        eager = copy.deepcopy(cpu_model).to("cuda")
        if cell["precision"]["channels_last"]:
            model = model.to(memory_format=torch.channels_last); eager = eager.to(memory_format=torch.channels_last)
        compile_started = time.perf_counter()
        candidate = self.compiler(model, backend)
        candidate.eval(); eager.eval()
        with torch.no_grad(), self.autocast(cell["precision"]["autocast"] == "float16"):
            candidate(self.input(cell, 1))
        torch.cuda.synchronize()
        first_call = time.perf_counter()-compile_started
        gate = self.equivalence(eager, candidate, cell, backend)
        write_json(folder / "model_equivalence.json", gate)
        if not gate["passed"] and scaled_gate:
            from gradient_probe import PROTOCOL_NAME, paired_model_probe
            gate = paired_model_probe(self, eager, candidate, cell, backend, protocol=PROTOCOL_NAME)
            write_json(folder / "scaled_model_equivalence.json", gate)
        require(gate["passed"], "Model forward/input/parameter-gradient equivalence gate failed")
        del eager

        model.load_state_dict(state, strict=True); model.zero_grad(set_to_none=True)
        self.activation_sites(model, cell)
        results = {"backend": backend, "compile_and_first_call_seconds": first_call,
                   "compile_seconds_isolated": None,
                   "compile_time_note": "First-call wall time includes compilation and first execution; training warmup/first-call times recorded separately, not included in steady-state events",
                   "fullgraph": POLICY["fullgraph"], "dynamic": POLICY["dynamic"],
                   "inference": [], "batch_failures": [], "equivalence_passed": True}
        amp = cell["precision"]["autocast"] == "float16"
        for batch in BATCHES:
            self.guard.verify()
            try:
                if backend != "eager":
                    candidate=None
                    torch._dynamo.reset();gc.collect();torch.cuda.empty_cache()
                    candidate=self.compiler(model.eval(),backend)
                x = self.input(cell, batch)
                candidate.eval()
                def forward():
                    with torch.inference_mode(), self.autocast(amp): candidate(x)
                batch_prepare=time.perf_counter()
                for _ in range(3): forward()
                torch.cuda.synchronize()
                batch_prepare_seconds=time.perf_counter()-batch_prepare
                proof = self.graph_evidence(forward, backend)
                require(proof["passed"], "Optimized backend lacked actual optimization evidence; not counted as optimized")
                write_json(folder / ("optimization_evidence_batch"+str(batch)+".json"), proof)
                row = self.timer(forward, cell, backend, "isolated_model_forward", batch,
                                 extra={"compiler_reset_before_batch":backend!="eager",
                                        "batch_first_three_calls_seconds":batch_prepare_seconds})
                row["per_image_latency_ms"] = row["summary_ms"]["median"] / batch
                row["throughput_images_per_second"] = batch*1000 / row["summary_ms"]["median"]
                results["inference"].append(row)

                del x
            except torch.cuda.OutOfMemoryError as exc:
                results["batch_failures"].append({"batch_size": batch, "status": "OOM_unavailable_in_recorded_process_state",
                                                 "compiler_reset_before_batch":backend!="eager","error": str(exc)[:2000]})
                if "x" in locals(): del x
                gc.collect(); torch.cuda.empty_cache()
            self.guard.verify()
        return results

BATCHES = (1, 16, 32, 64, 128)
