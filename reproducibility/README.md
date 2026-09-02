# Source snapshots

These directories preserve the training-relevant files identified by the
SHA-256 maps recorded during the experiments. They are archival snapshots, not
the maintained command-line interface. One class docstring that does not enter
the training computation was shortened for a journal-neutral research package;
the recorded and published hashes are documented in
`non_training_normalizations.json`. Every other snapshotted file remains
byte-identical to the recorded source map.

- `source_snapshot_q2_confirmatory/` reconstructs combined run-time code hash
  `988cfad419b8ae4613f37e9c4b4c513e39cd56e278ed0e23fc4bfd88d2bdf301`.
- `source_snapshot_organsmnist_no_hflip/` reconstructs combined run-time code hash
  `cc8753bdf13348a1cf6add379d5ad899da8b6dddada7e9c1f0d81496e3a38d72`.

Use the root source tree for normal execution and these snapshots when checking
the provenance of the published numerical results.
