from itertools import product
import math

from study_data import arguments, rows


def _render(result, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    axis = np.asarray(result["coordinates"])
    x, y = np.meshgrid(axis, axis, indexing="xy")
    matrices = []
    for surface in result["surfaces"]:
        matrix = np.full((len(axis), len(axis)), np.nan)
        for row in surface["points"]:
            matrix[row["iy"], row["ix"]] = row["cross_entropy"]
        if not np.isfinite(matrix).all():
            raise RuntimeError("Incomplete surface cannot be rendered as complete")
        matrices.append(matrix)
    lower = min(float(z.min()) for z in matrices)
    upper = max(float(z.max()) for z in matrices)
    norm = Normalize(vmin=lower, vmax=upper)
    fig = plt.figure(figsize=(10, 4.2), layout="constrained")
    for index, (surface, matrix) in enumerate(zip(result["surfaces"], matrices), 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.plot_surface(x, y, matrix, cmap="viridis", norm=norm, linewidth=0, antialiased=True)
        ax.set(xlabel="Direction 1 coefficient", ylabel="Direction 2 coefficient",
               zlabel="Cross-entropy", title=surface["activation"], zlim=(lower, upper))
        ax.view_init(elev=25, azim=-55)
    fig.savefig(output / "loss_landscape_3d.png", dpi=400)
    fig.savefig(output / "loss_landscape_3d.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained", sharex=True, sharey=True)
    levels = np.linspace(lower, upper if upper > lower else lower + 1, 25)
    for ax, surface, matrix in zip(axes, result["surfaces"], matrices):
        artist = ax.contourf(x, y, matrix, levels=levels, cmap="viridis", norm=norm)
        ax.scatter([0], [0], color="red", s=18)
        ax.set(title=surface["activation"], xlabel="Direction 1 coefficient",
               ylabel="Direction 2 coefficient", aspect="equal")
    fig.colorbar(artist, ax=axes, label="Cross-entropy")
    fig.savefig(output / "loss_landscape_contours.png", dpi=400)
    fig.savefig(output / "loss_landscape_contours.svg")
    plt.close(fig)


def main():
    args = arguments("Plot the two CIFAR-100 loss surfaces from the recorded FP32 grid.")
    coordinates = [round(-0.5 + i * 0.05, 12) for i in range(21)]
    surfaces = {"LGTS-f4-Alpha": {}, "ReLU": {}}
    metadata = {}
    for row in rows(args.results_dir / "loss_landscape_grid.csv"):
        act = row["activation"]
        if (act not in surfaces or row["dataset"] != "cifar100" or row["seed"] != "936137"
                or row["precision"] != "float32" or row["sample_count"] != "5000"):
            raise ValueError("Unexpected surface or evaluation protocol")
        identity = tuple(row[k] for k in ("run_id", "checkpoint_sha256", "direction_seed", "sampling_seed"))
        if metadata.setdefault(act, identity) != identity:
            raise ValueError(f"Mixed checkpoint or direction at {act}")
        i, j = int(row["ix"]), int(row["iy"])
        x, y, loss = (float(row[k]) for k in ("x", "y", "cross_entropy"))
        if ((i, j) in surfaces[act] or i not in range(21) or j not in range(21)
                or not math.isclose(x, coordinates[i], abs_tol=1e-12)
                or not math.isclose(y, coordinates[j], abs_tol=1e-12) or not math.isfinite(loss)):
            raise ValueError(f"Invalid grid point for {act} at ({i}, {j})")
        surfaces[act][i, j] = dict(ix=i, iy=j, cross_entropy=loss)
    grid = set(product(range(21), repeat=2))
    if any(points.keys() != grid for points in surfaces.values()):
        raise ValueError("Each surface must contain all 441 coordinates")
    if metadata["LGTS-f4-Alpha"][2:] != metadata["ReLU"][2:]:
        raise ValueError("The directions and samples must share the same seed between models")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _render(dict(coordinates=coordinates, surfaces=[dict(activation=a, points=list(p.values()))
                 for a, p in surfaces.items()]), args.output_dir)


if __name__ == "__main__":
    main()
