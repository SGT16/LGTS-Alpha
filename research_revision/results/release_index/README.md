# Per-run archive index

The 48 ZIP archives contain 6,368 files from 960 completed revision runs. Each ZIP contains 20 runs. To find a result, select a dataset, model, activation and seed in [run settings](../run_settings.csv). Look up the resulting `run_id` in [run_files.csv](run_files.csv). The `release_asset` column names the ZIP to download and `path` locates the file inside it. The same `run_id` links the observations to [seed results](../seed_results.csv).

Each run includes its epoch history, gradient observations, completion record and evaluation numerics. For the 930 runs with a final test assessment, the archives contain test predictions and per-class metrics. For the 30 validation-only learning-rate pilots, they contain validation predictions, per-class metrics and the split record. Parameter tables are available for applicable activations.

[run_archives.csv](run_archives.csv) records the byte size and SHA-256 digest of each complete ZIP. [run_files.csv](run_files.csv) also records the size and digest of each uncompressed member. The separate [interrupted ImageNet-100 attempt](interrupted_imagenet_attempt.json) records one training epoch and is excluded from the 960 completed runs and their analyses.
