from pathlib import Path
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def read_runs():
    with (HERE / "runs.csv").open(encoding="utf-8", newline="") as handle:
        return {row["run_id"]: row for row in csv.DictReader(handle)}


def command_for(row, data_dir, output_dir):
    shared = json.loads((HERE / "shared_arguments.json").read_text(encoding="utf-8"))
    arguments = dict(shared[row["stage"]])
    arguments.update(json.loads(row["arguments"]))
    target = Path(output_dir).resolve() / ("run_" + row["run_id"])
    if row["stage"] == "lr":
        script = HERE / "validation_engine.py"
        arguments["--plan"] = [str(HERE / "validation_plan.json")]
        arguments["--data-dir"] = [str(Path(data_dir).resolve())]
        arguments["--output-dir"] = [str(target)]
    else:
        script = HERE.parent / "engine" / "train_benchmark.py"
        arguments["--data_dir"] = [str(Path(data_dir).resolve())]
        arguments["--output_dir"] = [str(target)]
    command = [sys.executable, str(script)]
    for flag, values in arguments.items():
        command.append(flag)
        command.extend(values)
    return command


def main():
    parser = argparse.ArgumentParser(description="Print or execute one archived reviewer-training recipe.")
    parser.add_argument("run_id", help="Numeric run_id from runs.csv or results/run_settings.csv")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Run the printed recipe on CUDA")
    options = parser.parse_args()
    row = read_runs().get(options.run_id)
    if row is None:
        parser.error(f"Unknown reviewer-training run_id: {options.run_id}")
    command = command_for(row, options.data_dir, options.output_dir)
    print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command), flush=True)
    if options.execute:
        import torch

        if not torch.cuda.is_available():
            parser.error("The recorded runs require CUDA")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
