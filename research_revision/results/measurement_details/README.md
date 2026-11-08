# Measurement details

| File | Rows | Link to the published results |
| --- | ---: | --- |
| [Microbenchmark links](micro_measurement_links.csv) | 87 | Each microbenchmark unit refers to one existing timing in [timing_measurements.csv](../timing_measurements.csv) by `measurement_id`. The cell IDs agree in all 87 pairs. Count the linked timing once. |
| [Model profiles](model_profiles.csv) | 316 | Each of the 158 [measurement cells](../measurement_cells.csv) has one eager and one CUDA Graphs profile. Each profile links to its batch-one isolated-forward timing via `measurement_id` and to the checkpoint's training via `run_id`. |
| [Operator profiles](operator_profiles.csv.gz) | 70,089 | Entries link to a model profile via `profile_id`. The pair `profile_id, entry` is unique. The entry counts and sums of supported operator FLOPs reproduce all 316 profile summaries. |
| [Representative epochs](representative_epochs.csv) | 157 | Each row links by `cell_id` to the source learned checkpoint. There are 133 measured eager epochs and 24 ImageNet-100 entries marked unavailable at this stage. |
| [Epoch interruption](epoch_interruptions.csv) | 1 | The GPU availability check interrupted one ImageNet-100/MobileNetV2/LGTS-Alpha eager timing attempt. Its failed status is preserved. The same unit subsequently completed in [real_epochs.csv](../real_epochs.csv). |
| [Training resource times](training_resource_times.csv) | 853 | Each `run_id` links to [run_settings.csv](../run_settings.csv) and [training_history.csv](../training_history.csv). The recorded epoch counts and summed `time_sec` reproduce all 853 reported totals. |

The profiler ran outside the primary timing blocks. `compile_and_first_call_seconds` includes compilation and the first execution. Isolated compile time was unavailable. FLOP counts cover supported operators such as convolution, linear layers and matrix multiplication. The operator table retains the PyTorch event or kernel name, input shapes, call count, CPU and device times in microseconds, FLOPs and memory in bytes. Decompress the file as gzip to read standard UTF-8 CSV.

Each measured representative epoch started from final learned weights at the recorded initial learning rate. Its timer included loader startup, augmentation, transfers, loss, backward propagation, optimizer and scaler steps, parameter projection and first-minibatch gradient diagnostics. Checkpoint writing and evaluation were outside this interval. The single epoch per cell provides one duration. For the four-dataset comparison with common initialization, use [the original and matched-update real epochs](../real_epochs.csv).

Training resource time sums the completed epoch wall times, including validation where enabled. GPU-hours equal this sum divided by 3,600 for a single-GPU job. The source flags retain their recorded values, including `epoch_history_verified`. The export separately reconciles those totals with the published epoch rows. The timer excludes startup, partial interrupted epochs, checkpoint writes, final evaluation and idle periods. `device_index` distinguishes the four RTX A5500 devices. An empty value denotes an unavailable or inapplicable measurement.
