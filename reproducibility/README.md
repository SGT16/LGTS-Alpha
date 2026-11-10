# Source snapshots

These directories contain the source files used to verify the two recorded
training panels. The distributed Python files omit comments and docstrings.
For each file whose bytes differ from the recorded source, `non_training_normalizations.json`
lists both SHA-256 values. The verifier uses the distributed hash to check the
file and the recorded hash to reconstruct the historical run fingerprint.
The [November 8 release tree](https://github.com/SGT16/LGTS-Alpha/tree/8a2f6e97dc92db1cc51080e97c2e20a0522fe96b/reproducibility)
retains the earlier published copies and their original mapping.

- `source_snapshot_q2_confirmatory/` reconstructs combined run-time code hash
  `988cfad419b8ae4613f37e9c4b4c513e39cd56e278ed0e23fc4bfd88d2bdf301`.
- `source_snapshot_organsmnist_no_hflip/` reconstructs combined run-time code hash
  `cc8753bdf13348a1cf6add379d5ad899da8b6dddada7e9c1f0d81496e3a38d72`.

Use the root source tree for normal execution and these snapshots when checking
the provenance of the published numerical results.
