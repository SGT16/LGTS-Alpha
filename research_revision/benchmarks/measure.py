import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import pelu
import prelu
import real_epoch
import stress
from runtime import FILES, KINDS, ROOT, InferenceRuntime, Runtime, read, require, sha, write_json


def epoch_rows():
    with (ROOT / "results" / "real_epochs.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def epoch_arm(args):
    source, = [r for r in epoch_rows() if not r["source_unit_id"]
               and (r["dataset"], r["model"], r["activation"], r["backend"]) ==
               (args.dataset, args.model, args.activation, args.backend)]
    unit = {k:source[k] for k in ("dataset", "model", "activation", "backend", "protocol_sha256",
                                  "dataset_fingerprint_sha256")}
    unit.update(id=source["unit_id"], seed=int(source["seed"]), batch_size=int(source["batch_size"]),
                expected_images=int(source["images"]), image_size=int(source["image_size"]),
                expected_steps=int(source["steps"]))
    return unit, source


def epoch_measurement(args, rt):
    unit, prior = epoch_arm(args)
    real_epoch.INITIAL_SCALE = 256. if args.amp_policy == "matched-updates" else 65536.
    result = real_epoch.measure(unit, args.output, rt, lambda phase, step, total: None)
    result.update(protocol=prior["protocol"], qualified_timing=True)
    if args.amp_policy == "matched-updates":
        matched, = [r for r in epoch_rows() if r["source_unit_id"] == unit["id"]]
        result.update(id=matched["unit_id"], source_unit_id=unit["id"], protocol=matched["protocol"],
                      protocol_sha256=matched["protocol_sha256"])
        keys = ("initial_state_sha256", "backbone_sha256", "data_order_sha256", "augmentation_probe_sha256")
        mismatches = [key for key in keys if result[key] != prior[key]]
        result["historical_hash_mismatches"] = mismatches
        result["matches_reference"] = not mismatches
        if result["amp_skipped_steps"] or result["final_scale"] != 256. or mismatches:
            result.update(status="excluded", qualified_timing=False)
    return result


def cell_and_weights(args, cell_id):
    cells = read(FILES / "cells.json")
    require(cell_id in cells, "Unknown measurement cell")
    cell = cells[cell_id]
    cell["dataset_manifest_path"] = FILES / (cell["dataset_fingerprint_sha256"] + ".json")
    require(sha(cell["dataset_manifest_path"]) == cell["dataset_manifest_sha256"], "Dataset record changed")
    require(args.checkpoint.is_file() and sha(args.checkpoint) == cell["checkpoint_sha256"],
            "Checkpoint does not match the recorded SHA-256")
    cell["checkpoint_path"] = args.checkpoint
    return cell


def operation_time(rt, unit, cell, operation):
    return rt.timer(operation, cell, "eager", unit["kind"], unit["batch_size"],
                    iterations=100, warmup=50, blocks=10, minimum_seconds=.5,
                    extra={"activation_site":unit["activation_site"], "shape":unit["shape"],
                           "parameter_state":unit["parameter_state"], "dtype":unit["dtype"]})


def observed(rt, cpu, cell, unit, units):
    site = unit["activation_site"]
    module = copy.deepcopy(cpu.get_submodule(site))
    model = copy.deepcopy(cpu).cuda().train()
    if cell["precision"]["channels_last"]:
        model = model.to(memory_format=rt.torch.channels_last)
    if unit["activation"] == "PReLU":
        require(type(module) is rt.torch.nn.PReLU and module.num_parameters == 1, "Unexpected PReLU site")
        siblings = [u for u in units if u["cell_id"] == cell["id"] and u["protocol"] == unit["protocol"]]
        identity = prelu.capture_identity(cell, siblings, sha(FILES/"activation_units.json"))
        payload = prelu.sample_pairs(rt, model, cell, identity, rt.guard.verify)
        meta = payload["metadata"]
        pair = payload["pairs"][site]
        del model, cpu
        operation, gate = prelu.operation(rt, module, pair["x"], pair["dy"], cell, unit, meta)
    else:
        require(type(module) is rt.activations.PELU, "Unexpected PELU site")
        full = dict(unit, batch_size=cell["batch_size"], shape=[cell["batch_size"], *unit["shape"][1:]])
        identity = pelu.capture_identity(cell, full, sha(FILES/"activation_units.json"))
        payload = pelu.sample_input(rt, model, cell, full, identity, rt.guard.verify)
        meta = payload["metadata"]
        tensor = payload["tensor"][:unit["batch_size"]].clone()
        meta.update(shape=unit["shape"], training_batch_size=cell["batch_size"], timed_batch_size=unit["batch_size"],
                    timed_training_indices=meta["batches"][0]["training_indices"][:unit["batch_size"]], input_sha256=
                    real_epoch.hash_tensors([("activation_input", tensor)]))
        payload["tensor"] = tensor
        del model, cpu
        operation, gate = pelu.operation(rt, module, tensor, cell, unit, meta)
    rt.torch.save(payload, rt.output/meta["tensor_file"])
    write_json(rt.output/"capture.json", meta)
    write_json(rt.output/"gradient_gate.json", gate)
    if not gate["passed"]:
        return {"status":"excluded", "gate":gate}
    return {"status":"accepted", "timing":operation_time(rt, unit, cell, operation)}


def legacy(rt, cpu, cell, unit):
    torch = rt.torch
    site = unit["activation_site"]
    rt.set_policy(cell)
    for name, shape in (("relu", [128,64,64,64]), ("layer4.1.act2", [128,512,8,8])):
        x = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
        if name == site:
            break
        del x
        scratch = torch.randn(shape, device="cuda", dtype=torch.float16)
        del scratch
    require(site == name and list(x.shape) == unit["shape"], "Legacy activation location changed")
    module = copy.deepcopy(cpu.get_submodule(site)).cuda()
    require(type(module) is torch.nn.PReLU and module.num_parameters == 1, "Original learned PReLU site missing")
    with rt.autocast(True):
        fixed = module(x)
    dy = torch.ones_like(fixed)
    require(fixed.dtype == torch.float16, "Native output precision changed")
    historical, = [row for row in read(ROOT / "results" / "prelu_fp16_slope_replays.json")["units"]
                   if row["unit_id"] == unit["id"]]
    hashes = {key: hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()
              for key, tensor in (("input_sha256", x), ("upstream_sha256", dy))}
    require(all(historical[key] == value for key, value in hashes.items()), "Legacy probe tensor hash mismatch")

    def operation():
        module.zero_grad(set_to_none=True)
        x.grad = None
        if unit["kind"] == KINDS[1]:
            fixed.backward(dy, retain_graph=True)
        else:
            with rt.autocast(True):
                y = module(x)
            y.backward(dy)

    operation()
    probe = lambda: prelu.gradient_audit(torch,module,x,fixed)
    audits = [probe()]
    base = rt.journal

    class BlockAudit:
        def write(self, line):
            check = probe()
            audits.append(check)
            base.write(json.dumps(dict(json.loads(line), gradient_audit=check, **hashes), allow_nan=False) + "\n")
        def flush(self):
            base.flush()

    rt.journal = BlockAudit()
    try:
        timing = operation_time(rt, unit, cell, operation)
    finally:
        rt.journal = base
    passed = all(check["output_finite"] and check["input_gradient_finite"] and
                 check["slope_gradient_finite"] for check in audits)
    return {"status":"accepted" if passed else "diagnostic_nonfinite", "qualified_timing":passed,
            "timing":timing, "audits":audits, "source_measurement_id":historical["source_measurement_id"], **hashes}


def measure_cell(args, rt):
    units = read(FILES/"activation_units.json")
    if args.command == "activation-replay":
        selected, = [u for u in units if u["id"] == args.unit]
        cell = cell_and_weights(args,selected["cell_id"])
    else:
        cell = cell_and_weights(args,args.cell)
    rt.set_policy(cell)
    cpu, state = rt.load(cell)
    if args.command == "training-step":
        gate, result = rt.measure_training_component(cpu, state, cell, args.backend, rt.output)
        return {"status":"accepted" if gate["passed"] and result.get("raw_repeats") else "excluded",
                "cell_id":cell["id"], "result":result}
    if args.command == "inference":
        return rt.inference(cell, args.backend, cpu, state, rt.output, args.scaled_gate)
    if args.command == "activation-replay":
        if selected["protocol"] == "prelu_legacy_fp16_validation_20261108":
            result = legacy(rt, cpu, cell, selected)
        else:
            require(args.data_dir is not None, "--data-dir is required for training captures")
            result = observed(rt, cpu, cell, selected, units)
        return dict(result, unit=selected["id"], protocol=selected["protocol"])
    sites = {row[0]:row for row in rt.activation_sites(copy.deepcopy(cpu).cuda(),cell)}
    name = list(sites)[0 if args.site == "early" else -1]
    _, _, shape, dtype = sites[name]
    unit = dict(cell_id=cell["id"], activation_site=name, parameter_state=args.state, batch_size=args.batch,
                kind=args.kind, shape=[args.batch,*shape[1:]], dtype=dtype)
    layer = stress.source_at_site(rt,cell,sites,unit)
    module, protocol, gate = stress.gated_module(rt,layer,cell["activation"])
    write_json(rt.output/"activation_equivalence.json", gate)
    if not gate["passed"]:
        return {"status":"excluded", "gate":gate, "protocol":protocol, "unit":unit}
    rt.set_policy(cell)
    operation, finite = stress.build_operation(rt,module,cell,unit)
    if not all(finite.values()):
        return {"status":"excluded", "finite":finite, "unit":unit, "protocol":protocol}
    return {"status":"accepted", "unit":unit, "protocol":protocol, "timing":operation_time(rt,unit,cell,operation)}


def arguments():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    listing = subs.add_parser("list")
    listing.add_argument("items", choices=["epochs", "cells", "activation-replays"])
    for command in ("epoch","training-step","inference","operator","activation-replay"):
        sub = subs.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        if command in ("epoch","training-step","inference"):
            sub.add_argument("--backend", choices=["eager","cudagraphs"], required=True)
        if command == "epoch":
            sub.add_argument("--dataset", choices=sorted({*real_epoch.DATASETS,"imagenet100"}),required=True)
            sub.add_argument("--model", choices=sorted(real_epoch.MODELS),required=True)
            sub.add_argument("--activation", choices=sorted(real_epoch.ACTIVATIONS),required=True)
            sub.add_argument("--amp-policy", choices=["matched-updates","original"], default="matched-updates")
            sub.add_argument("--data-dir", type=Path, required=True)
        else:
            sub.add_argument("--checkpoint", type=Path, required=True)
            if command == "activation-replay":
                sub.add_argument("--unit", required=True)
                sub.add_argument("--data-dir", type=Path)
            else:
                sub.add_argument("--cell", required=True)
        if command == "operator":
            sub.add_argument("--site",choices=["early","late"],required=True)
            sub.add_argument("--batch",type=int,choices=(1,32,64,128),required=True)
            sub.add_argument("--kind",choices=KINDS,required=True)
            sub.add_argument("--state",choices=("learned_checkpoint","declared_initialization"),default="learned_checkpoint")
        if command == "inference":
            sub.add_argument("--scaled-gate",action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    if args.command == "list":
        if args.items == "epochs":
            for row in epoch_rows():
                if not row["source_unit_id"]:
                    print(row["dataset"], row["model"], row["activation"], row["backend"])
        elif args.items == "cells":
            for cid,cell in read(FILES/"cells.json").items():
                print(cid,cell["dataset"],cell["model"],cell["activation"],cell["checkpoint_sha256"])
        else:
            for row in read(FILES/"activation_units.json"):
                print(row["id"],row["dataset"],row["model"],row["activation_site"],row["kind"],
                      row["batch_size"],row["checkpoint_sha256"])
        return
    args.output.mkdir(parents=True, exist_ok=False)
    cls = InferenceRuntime if args.command == "inference" else Runtime
    rt = cls(args.output,getattr(args,"data_dir",None))
    try:
        write_json(args.output/"environment.json",dict(python=platform.python_version(), torch=str(rt.torch.__version__),
                   torchvision=importlib.metadata.version("torchvision"), cuda=rt.torch.version.cuda,
                   cudnn=rt.torch.backends.cudnn.version(), gpu=rt.torch.cuda.get_device_name(0)))
        try:
            row = epoch_measurement(args,rt) if args.command == "epoch" else measure_cell(args,rt)
        except real_epoch.NumericExclusion as exc:
            row = {"status":"excluded","reason":str(exc),"evidence":exc.evidence}
        write_json(args.output/"result.json", row)
    finally:
        rt.journal.close()


if __name__ == "__main__":
    main()
