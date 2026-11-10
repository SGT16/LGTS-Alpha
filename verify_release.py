
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NORMALIZATIONS = ROOT / "reproducibility" / "non_training_normalizations.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_result_checksums() -> None:
    manifest = ROOT / "results" / "CHECKSUMS.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        target = manifest.parent / relative.strip()
        if not target.is_file():
            raise AssertionError(f"Missing result file: {target.relative_to(ROOT)}")
        actual = sha256(target)
        if actual != expected:
            raise AssertionError(
                f"Checksum mismatch for {target.relative_to(ROOT)}: "
                f"expected {expected}, got {actual}"
            )
        checked += 1
    if checked == 0:
        raise AssertionError("The result checksum manifest is empty")
    print(f"Result checksums: {checked} files verified")


def verify_source_snapshot(result_dir: str, snapshot_dir: str) -> None:
    provenance_path = ROOT / "results" / result_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    snapshot = ROOT / "reproducibility" / snapshot_dir
    combined = hashlib.sha256()
    normalization_data = json.loads(NORMALIZATIONS.read_text(encoding="utf-8"))
    normalizations = {
        (scope, item["path"]): item
        for item in normalization_data["normalizations"]
        for scope in item["snapshot_scopes"]
    }
    normalized_files = 0

    for relative, expected in provenance["source_file_sha256"].items():
        target = snapshot / Path(relative)
        if not target.is_file():
            raise AssertionError(f"Missing source snapshot file: {target.relative_to(ROOT)}")
        actual = sha256(target)
        if actual != expected:
            normalization = normalizations.get((snapshot_dir, relative))
            if not normalization or not (
                normalization["recorded_sha256"] == expected
                and normalization["published_sha256"] == actual
                and normalization["training_semantics"] == "unchanged"
            ):
                raise AssertionError(
                    f"Source hash mismatch for {target.relative_to(ROOT)}: "
                    f"expected {expected}, got {actual}"
                )
            normalized_files += 1
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(expected.encode("ascii"))
        combined.update(b"\0")

    actual_combined = combined.hexdigest()
    expected_combined = provenance["code_sha256"]
    if actual_combined != expected_combined:
        raise AssertionError(
            f"Combined source hash mismatch for {snapshot_dir}: "
            f"expected {expected_combined}, got {actual_combined}"
        )
    suffix = f" ({normalized_files} documented metadata normalization)" if normalized_files else ""
    print(f"Source snapshot: {snapshot_dir} verified{suffix}")


def verify_run_table(relative: str, expected_rows: int, expected_datasets: set[str]) -> None:
    path = ROOT / relative
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != expected_rows:
        raise AssertionError(f"{relative}: expected {expected_rows} rows, found {len(rows)}")

    key_fields = ("dataset", "model", "activation", "training_seed", "fold")
    keys = {tuple(row[field] for field in key_fields) for row in rows}
    if len(keys) != expected_rows:
        raise AssertionError(f"{relative}: duplicated experimental conditions detected")

    datasets = {row["dataset"] for row in rows}
    if datasets != expected_datasets:
        raise AssertionError(
            f"{relative}: expected datasets {sorted(expected_datasets)}, found {sorted(datasets)}"
        )

    if any(row.get("status") != "confirmatory_done" for row in rows):
        raise AssertionError(f"{relative}: at least one run is not marked confirmatory_done")
    if any(row.get("epochs_run") != "100" for row in rows):
        raise AssertionError(f"{relative}: at least one run does not contain 100 epochs")

    print(f"Run table: {relative} contains {expected_rows} unique completed runs")


def main() -> None:
    verify_result_checksums()
    verify_source_snapshot("confirmatory", "source_snapshot_q2_confirmatory")
    verify_source_snapshot(
        "organsmnist_no_hflip", "source_snapshot_organsmnist_no_hflip"
    )
    verify_run_table(
        "results/confirmatory/run_results.csv",
        225,
        {"cifar100", "dermamnist", "organsmnist"},
    )
    verify_run_table(
        "results/organsmnist_no_hflip/run_results.csv",
        75,
        {"organsmnist"},
    )
    print("LGTS-Alpha release verification completed successfully.")


if __name__ == "__main__":
    main()
