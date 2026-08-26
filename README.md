# stone-mojo

An annotation-free terrain traversability auto-labeler, in pure Mojo.

Reimplementation of the pipeline described in *STONE: A Scalable Multi-Modal
Surround-View 3D Traversability Dataset for Off-Road Robot Navigation*
(Park et al., ICRA 2026) — [paper](https://arxiv.org/abs/2603.09175) ·
[project page](https://konyul.github.io/STONE-dataset/).

There is no neural network in the pipeline. Every stage is classical geometry,
which is the reason it is worth writing in Mojo:

1. **ingest** — bin a LiDAR sweep into a 2.5-D grid, streaming per-cell first
   and second moments (count, z extent, 3×3 scatter).
2. **finalize** — per cell, fit a plane from the eigenvectors of the scatter
   matrix and derive elevation / slope / roughness / step.
3. **fit** — the driven trajectory is free ground truth. Cells the robot
   actually crossed define a Gaussian over feature space.
4. **label** — Mahalanobis distance from that Gaussian assigns traversability
   *everywhere the robot did not go*.

## API

`src/stone.mojo` is the whole library. The types, in pipeline order:

| type | role |
| --- | --- |
| `Vec3` | 3-D point |
| `GridSpec` | grid origin, resolution, extent |
| `CellAccumulator` | streaming per-cell moments during ingest |
| `Sym3` / `Eigen3` | symmetric 3×3 scatter matrix and its eigendecomposition |
| `TerrainCell` | a finalized cell: plane fit + derived features |
| `CellFeatures` | the (slope, roughness, step) feature vector |
| `TerrainGrid` | the binned sweep; ingest and finalize live here |
| `TraversabilityModel` | Gaussian fitted to the driven cells |
| `StoneLabels` | per-cell labels: `LABEL_UNKNOWN`, `LABEL_TRAVERSABLE`, `LABEL_NON_TRAVERSABLE` |

Tunables (minimum points per cell, the χ²(3) cutoff, per-feature noise floors)
are `comptime` constants at the top of the file.

## Use

```
mojo run -I src your_program.mojo
```
