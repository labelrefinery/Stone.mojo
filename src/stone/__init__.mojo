"""STONE-style annotation-free terrain traversability auto-labeling.

Reimplementation of the auto-labeling pipeline described in:

    STONE: A Scalable Multi-Modal Surround-View 3D Traversability Dataset
    for Off-Road Robot Navigation
    Park et al., ICRA 2026
    Paper:   https://arxiv.org/abs/2603.09175
    Project: https://konyul.github.io/STONE-dataset/

The pipeline has no neural network in it. Every stage is classical geometry,
which is the whole reason it is worth writing in Mojo:

    1. ingest   - bin a LiDAR sweep into a 2.5-D grid, streaming per-cell
                  first and second moments (count, z extent, 3x3 scatter).
    2. finalize - per cell, fit a plane via the eigenvectors of the scatter
                  matrix and derive elevation / slope / roughness / step.
    3. fit      - the driven trajectory is free ground truth. Cells the robot
                  actually crossed define a Gaussian over feature space.
    4. label    - Mahalanobis distance from that Gaussian assigns
                  traversability *everywhere the robot did not go*.

"""

from std.math import sqrt, acos, abs


# ===----------------------------------------------------------------------===
# Tunables
# ===----------------------------------------------------------------------===

comptime MIN_POINTS_PER_CELL = 8
"""Cells with fewer returns than this stay UNKNOWN. Plane fits on 3-4 points
are noise, and a confidently wrong label is worse than no label."""

comptime CHI2_3DOF_95 = 7.815
"""Chi-square critical value, 3 degrees of freedom, p=0.95. Feature vectors are
3-D, so a squared Mahalanobis distance above this is a 95%-confidence outlier
against the driven distribution."""

comptime SLOPE_NOISE_VAR = 3.0e-4
"""Variance floor on slope, rad^2. ~1 degree of angular uncertainty in a plane
fit from a real sweep."""

comptime ROUGHNESS_NOISE_VAR = 4.0e-4
"""Variance floor on roughness, m^2. ~2cm, the range noise of a typical
spinning LiDAR."""

comptime STEP_NOISE_VAR = 9.0e-4
"""Variance floor on step height, m^2. ~3cm: worse than range noise because
z_max - z_min is an extremal statistic and collects the tails.

These three exist because a machine that drives a uniform corridor produces a
near-singular sample covariance -- every driven cell is flat, so the fitted
distribution claims slope is known to eight decimal places. Invert that and any
other flat cell in the world sits a thousand sigma away, so the labeler marks
the entire site impassable. Adding a sensor-noise floor to the diagonal says
the obvious physical thing: the model may not claim more precision than the
LiDAR has. Tune these to your sensor, not to make a test pass."""

comptime COVARIANCE_RIDGE = 1e-12
"""Last-resort numerical ridge, applied at inversion. The noise floors above do
the real conditioning; this only guards against exact singularity."""


# ===----------------------------------------------------------------------===
# Labels
# ===----------------------------------------------------------------------===

comptime LABEL_UNKNOWN = 0
"""Not enough observations to decide."""

comptime LABEL_TRAVERSABLE = 1
comptime LABEL_NON_TRAVERSABLE = 2


# ===----------------------------------------------------------------------===
# Geometry primitives
# ===----------------------------------------------------------------------===


struct Vec3(ImplicitlyCopyable, Movable):
    """A point or direction in the sensor/map frame, meters."""

    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64, y: Float64, z: Float64):
        self.x = x
        self.y = y
        self.z = z

    def norm(self) -> Float64:
        return sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


struct GridSpec(ImplicitlyCopyable, Movable):
    """A 2.5-D grid over the ground plane.

    STONE reasons about voxel-level traversability, but the ground is a height
    field over the vast majority of a job site, so one cell per (x, y) column
    carrying its z statistics captures the same information far more cheaply.
    """

    var origin_x: Float64
    var origin_y: Float64
    var cell_size: Float64
    var nx: Int
    var ny: Int

    def __init__(
        out self,
        origin_x: Float64,
        origin_y: Float64,
        cell_size: Float64,
        nx: Int,
        ny: Int,
    ):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.cell_size = cell_size
        self.nx = nx
        self.ny = ny

    def cell_count(self) -> Int:
        return self.nx * self.ny

    def same_as(self, other: GridSpec) -> Bool:
        """The same lattice: one's cell indices mean the same cells in the
        other."""
        return (
            self.origin_x == other.origin_x
            and self.origin_y == other.origin_y
            and self.cell_size == other.cell_size
            and self.nx == other.nx
            and self.ny == other.ny
        )

    def index_of(self, x: Float64, y: Float64) -> Int:
        """Flat cell index for a world point, or -1 if outside the grid."""
        var ix = Int((x - self.origin_x) / self.cell_size)
        var iy = Int((y - self.origin_y) / self.cell_size)
        if ix < 0 or iy < 0 or ix >= self.nx or iy >= self.ny:
            return -1
        return iy * self.nx + ix


# ===----------------------------------------------------------------------===
# Stage 1: streaming accumulation
# ===----------------------------------------------------------------------===


struct CellAccumulator(ImplicitlyCopyable, Movable):
    """Per-cell running moments.

    Deliberately streaming: one pass over the sweep, no per-cell point lists.
    A 128-beam sweep is ~2e5 points and a 200x200 grid is 4e4 cells, so keeping
    the points would cost far more than the ten scalars that actually matter.
    """

    var count: Int
    var z_min: Float64
    var z_max: Float64
    var sum_x: Float64
    var sum_y: Float64
    var sum_z: Float64
    var sum_xx: Float64
    var sum_xy: Float64
    var sum_xz: Float64
    var sum_yy: Float64
    var sum_yz: Float64
    var sum_zz: Float64

    def __init__(out self):
        self.count = 0
        self.z_min = 0.0
        self.z_max = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_z = 0.0
        self.sum_xx = 0.0
        self.sum_xy = 0.0
        self.sum_xz = 0.0
        self.sum_yy = 0.0
        self.sum_yz = 0.0
        self.sum_zz = 0.0

    def add(mut self, p: Vec3):
        if self.count == 0:
            self.z_min = p.z
            self.z_max = p.z
        else:
            if p.z < self.z_min:
                self.z_min = p.z
            if p.z > self.z_max:
                self.z_max = p.z
        self.count += 1
        self.sum_x += p.x
        self.sum_y += p.y
        self.sum_z += p.z
        self.sum_xx += p.x * p.x
        self.sum_xy += p.x * p.y
        self.sum_xz += p.x * p.z
        self.sum_yy += p.y * p.y
        self.sum_yz += p.y * p.z
        self.sum_zz += p.z * p.z

    def merge(mut self, other: CellAccumulator):
        """Adds `other`'s points, as if they had been `add`ed here.

        Count, `z_min` and `z_max` come out exactly as streaming would give
        them, in any order. The sums are floating point, so they can differ
        from a stream in the last bits depending on how the points were
        grouped; merging the same groups in the same order always gives the
        same bits.
        """
        if other.count == 0:
            return
        if self.count == 0:
            self = other
            return
        if other.z_min < self.z_min:
            self.z_min = other.z_min
        if other.z_max > self.z_max:
            self.z_max = other.z_max
        self.count += other.count
        self.sum_x += other.sum_x
        self.sum_y += other.sum_y
        self.sum_z += other.sum_z
        self.sum_xx += other.sum_xx
        self.sum_xy += other.sum_xy
        self.sum_xz += other.sum_xz
        self.sum_yy += other.sum_yy
        self.sum_yz += other.sum_yz
        self.sum_zz += other.sum_zz


struct CellPatch(Copyable, Movable, Sized):
    """The cells one batch of points touched, and their moments: a sparse
    piece of a `TerrainGrid`.

    This is how a grid is built in pieces, one sweep per worker, and merged:
    a sweep touches a small part of a grid that covers a whole drive, so a
    dense grid per sweep would be mostly zeros. `indices` ascend, and
    `accumulators[i]` is the cell at `indices[i]` of `spec`.
    """

    var spec: GridSpec
    var indices: List[Int]
    var accumulators: List[CellAccumulator]

    def __init__(out self, spec: GridSpec):
        self.spec = spec
        self.indices = List[Int]()
        self.accumulators = List[CellAccumulator]()

    def __init__(
        out self,
        spec: GridSpec,
        var indices: List[Int],
        var accumulators: List[CellAccumulator],
    ):
        self.spec = spec
        self.indices = indices^
        self.accumulators = accumulators^

    @staticmethod
    def of(spec: GridSpec, points: List[Vec3]) -> CellPatch:
        """Bins `points` into `spec`'s cells, keeping only the touched ones.
        Points outside the grid are dropped, as `TerrainGrid.ingest` drops
        them."""
        var slot = Dict[Int, Int]()
        var touched = List[Int]()
        var accs = List[CellAccumulator]()
        for i in range(len(points)):
            var p = points[i]
            var idx = spec.index_of(p.x, p.y)
            if idx < 0:
                continue
            var s = slot.get(idx, -1)
            if s < 0:
                s = len(accs)
                slot[idx] = s
                touched.append(idx)
                accs.append(CellAccumulator())
            accs[s].add(p)
        sort(touched)
        var ordered = List[CellAccumulator](capacity=len(touched))
        for idx in touched:
            ordered.append(accs[slot.get(idx, 0)])
        return CellPatch(spec, touched^, ordered^)

    def __len__(self) -> Int:
        return len(self.indices)


# ===----------------------------------------------------------------------===
# Symmetric 3x3 linear algebra
# ===----------------------------------------------------------------------===


struct Sym3(ImplicitlyCopyable, Movable):
    """A symmetric 3x3 matrix stored as its six unique entries."""

    var xx: Float64
    var xy: Float64
    var xz: Float64
    var yy: Float64
    var yz: Float64
    var zz: Float64

    def __init__(
        out self,
        xx: Float64,
        xy: Float64,
        xz: Float64,
        yy: Float64,
        yz: Float64,
        zz: Float64,
    ):
        self.xx = xx
        self.xy = xy
        self.xz = xz
        self.yy = yy
        self.yz = yz
        self.zz = zz

    def get(self, i: Int, j: Int) -> Float64:
        if i == 0 and j == 0:
            return self.xx
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return self.xy
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return self.xz
        if i == 1 and j == 1:
            return self.yy
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return self.yz
        return self.zz


struct Eigen3(Movable):
    """Eigen-decomposition of a symmetric 3x3, ascending by eigenvalue."""

    var values: Array[Float64, 3]
    var vectors: Array[Float64, 9]
    """Column-major: eigenvector k is vectors[3*k + 0 .. 3*k + 2]."""

    def __init__(out self):
        self.values = Array[Float64, 3](fill=0.0)
        self.vectors = Array[Float64, 9](fill=0.0)

    def vector(self, k: Int) -> Vec3:
        return Vec3(
            self.vectors[3 * k + 0],
            self.vectors[3 * k + 1],
            self.vectors[3 * k + 2],
        )


def jacobi_eigen(m: Sym3) -> Eigen3:
    """Cyclic Jacobi eigen-decomposition of a symmetric 3x3 matrix.

    Jacobi rather than the closed-form trigonometric solution: the analytic
    route loses catastrophic precision when eigenvalues are near-degenerate,
    which is exactly the regime a flat patch of ground sits in. For a 3x3 the
    sweep count is tiny, so robustness is free.
    """
    # Working copy of the full matrix.
    var a = Array[Float64, 9](fill=0.0)
    for i in range(3):
        for j in range(3):
            a[3 * i + j] = m.get(i, j)

    # Accumulated rotations, initialized to identity.
    var v = Array[Float64, 9](fill=0.0)
    v[0] = 1.0
    v[4] = 1.0
    v[8] = 1.0

    for _sweep in range(24):
        # Largest off-diagonal magnitude drives both the pivot and the
        # convergence test.
        var off = 0.0
        var p = 0
        var q = 1
        for i in range(3):
            for j in range(i + 1, 3):
                var mag = abs(a[3 * i + j])
                if mag > off:
                    off = mag
                    p = i
                    q = j
        if off < 1e-15:
            break

        var apq = a[3 * p + q]
        var app = a[3 * p + p]
        var aqq = a[3 * q + q]

        # Rotation angle that annihilates a[p][q].
        var theta = (aqq - app) / (2.0 * apq)
        var root = sqrt(1.0 + theta * theta)
        var t = 1.0 / (theta + root) if theta >= 0.0 else -1.0 / (root - theta)
        var c = 1.0 / sqrt(1.0 + t * t)
        var s = t * c

        # Apply the rotation to the matrix from both sides.
        for k in range(3):
            var akp = a[3 * k + p]
            var akq = a[3 * k + q]
            a[3 * k + p] = c * akp - s * akq
            a[3 * k + q] = s * akp + c * akq
        for k in range(3):
            var apk = a[3 * p + k]
            var aqk = a[3 * q + k]
            a[3 * p + k] = c * apk - s * aqk
            a[3 * q + k] = s * apk + c * aqk

        # And accumulate it into the eigenvector basis.
        for k in range(3):
            var vkp = v[3 * k + p]
            var vkq = v[3 * k + q]
            v[3 * k + p] = c * vkp - s * vkq
            v[3 * k + q] = s * vkp + c * vkq

    # Extract, then sort ascending by eigenvalue with a 3-element selection
    # sort. The smallest eigenvalue's vector is the surface normal.
    var result = Eigen3()
    var vals = Array[Float64, 3](fill=0.0)
    var order = Array[Int, 3](fill=0)
    for i in range(3):
        vals[i] = a[3 * i + i]
        order[i] = i

    for i in range(3):
        var best = i
        for j in range(i + 1, 3):
            if vals[order[j]] < vals[order[best]]:
                best = j
        var tmp = order[i]
        order[i] = order[best]
        order[best] = tmp

    for k in range(3):
        var src = order[k]
        result.values[k] = vals[src]
        for r in range(3):
            result.vectors[3 * k + r] = v[3 * r + src]
    return result^


def invert_sym3(m: Sym3, ridge: Float64) -> Sym3:
    """Invert a symmetric 3x3 via cofactors, after ridge regularization.

    Returns the (also symmetric) inverse. The ridge keeps this defined when the
    driven terrain is uniform enough to make the covariance rank-deficient.
    """
    var xx = m.xx + ridge
    var yy = m.yy + ridge
    var zz = m.zz + ridge
    var xy = m.xy
    var xz = m.xz
    var yz = m.yz

    var c00 = yy * zz - yz * yz
    var c01 = xz * yz - xy * zz
    var c02 = xy * yz - xz * yy
    var det = xx * c00 + xy * c01 + xz * c02

    # A determinant this small means the ridge itself is doing all the work;
    # fall back to a scaled identity rather than emitting infinities.
    if abs(det) < 1e-18:
        var fallback = 1.0 / (ridge + 1e-12)
        return Sym3(fallback, 0.0, 0.0, fallback, 0.0, fallback)

    var inv_det = 1.0 / det
    return Sym3(
        c00 * inv_det,
        c01 * inv_det,
        c02 * inv_det,
        (xx * zz - xz * xz) * inv_det,
        (xz * xy - xx * yz) * inv_det,
        (xx * yy - xy * xy) * inv_det,
    )


# ===----------------------------------------------------------------------===
# Stage 2: derived per-cell terrain attributes
# ===----------------------------------------------------------------------===


struct TerrainCell(ImplicitlyCopyable, Movable):
    """The geometric attributes STONE extracts per cell."""

    var observed: Bool
    var count: Int
    var elevation: Float64
    """Mean z of the returns in the cell, meters."""
    var slope: Float64
    """Angle between the fitted surface normal and vertical, radians."""
    var roughness: Float64
    """RMS residual to the fitted plane, meters. sqrt of the smallest
    eigenvalue of the scatter matrix."""
    var step: Float64
    """Vertical extent (z_max - z_min) within the cell, meters.

    Separates a rough-but-drivable gravel patch from a curb or trench edge
    that happens to share its roughness."""
    var normal: Vec3

    def __init__(out self):
        self.observed = False
        self.count = 0
        self.elevation = 0.0
        self.slope = 0.0
        self.roughness = 0.0
        self.step = 0.0
        self.normal = Vec3(0.0, 0.0, 1.0)


def finalize_cell(acc: CellAccumulator) -> TerrainCell:
    """Turn accumulated moments into terrain attributes via a plane fit."""
    var cell = TerrainCell()
    cell.count = acc.count
    if acc.count < MIN_POINTS_PER_CELL:
        return cell

    var n = Float64(acc.count)
    var mx = acc.sum_x / n
    var my = acc.sum_y / n
    var mz = acc.sum_z / n

    # Central second moments. Subtracting the mean here (rather than
    # accumulating centered) is the numerically sloppy route, but LiDAR
    # coordinates are single-digit-meter magnitudes so it is well within
    # Float64 headroom.
    var scatter = Sym3(
        acc.sum_xx / n - mx * mx,
        acc.sum_xy / n - mx * my,
        acc.sum_xz / n - mx * mz,
        acc.sum_yy / n - my * my,
        acc.sum_yz / n - my * mz,
        acc.sum_zz / n - mz * mz,
    )

    var eig = jacobi_eigen(scatter)
    var normal = eig.vector(0)

    # Orient the normal upward so slope is unambiguous.
    if normal.z < 0.0:
        normal = Vec3(-normal.x, -normal.y, -normal.z)

    var nz = normal.z
    if nz > 1.0:
        nz = 1.0
    if nz < -1.0:
        nz = -1.0

    cell.observed = True
    cell.elevation = mz
    cell.normal = normal
    cell.slope = acos(abs(nz))
    var lambda_min = eig.values[0]
    if lambda_min < 0.0:
        lambda_min = 0.0
    cell.roughness = sqrt(lambda_min)
    cell.step = acc.z_max - acc.z_min
    return cell


struct CellFeatures(ImplicitlyCopyable, Movable):
    """The 3-D feature vector traversability is decided in."""

    var slope: Float64
    var roughness: Float64
    var step: Float64

    def __init__(out self, slope: Float64, roughness: Float64, step: Float64):
        self.slope = slope
        self.roughness = roughness
        self.step = step


def features_of(cell: TerrainCell) -> CellFeatures:
    return CellFeatures(cell.slope, cell.roughness, cell.step)


# ===----------------------------------------------------------------------===
# The grid
# ===----------------------------------------------------------------------===


struct TerrainGrid(Movable):
    """Dense 2.5-D terrain grid: accumulate, then finalize."""

    var spec: GridSpec
    var accumulators: List[CellAccumulator]
    var cells: List[TerrainCell]

    def __init__(out self, spec: GridSpec):
        self.spec = spec
        self.accumulators = List[CellAccumulator](
            length=spec.cell_count(), fill=CellAccumulator()
        )
        self.cells = List[TerrainCell](
            length=spec.cell_count(), fill=TerrainCell()
        )

    def ingest(mut self, points: List[Vec3]):
        """Stage 1. Bin one sweep. Call repeatedly to aggregate a sequence --
        this is the offboard advantage: every pass over the same ground adds
        viewpoints, and nothing has to run in real time."""
        for i in range(len(points)):
            var p = points[i]
            var idx = self.spec.index_of(p.x, p.y)
            if idx >= 0:
                self.accumulators[idx].add(p)

    def merge(mut self, patch: CellPatch) raises:
        """Stage 1, in pieces. Adds a patch built elsewhere, such as one sweep
        binned on another machine, as if its points had been `ingest`ed here.
        Merging the same patches in the same order always gives the same
        grid; see `CellAccumulator.merge` for how it compares to `ingest`."""
        if not self.spec.same_as(patch.spec):
            raise Error("stone: a patch on another grid can't be merged")
        for i in range(len(patch.indices)):
            self.accumulators[patch.indices[i]].merge(patch.accumulators[i])

    def finalize(mut self):
        """Stage 2. Fit every observed cell."""
        for i in range(len(self.accumulators)):
            self.cells[i] = finalize_cell(self.accumulators[i])


# ===----------------------------------------------------------------------===
# Stages 3 and 4: fit from the trajectory, then propagate
# ===----------------------------------------------------------------------===


struct TraversabilityModel(Movable):
    """A Gaussian over feature space, fitted to terrain the robot drove on.

    This is the core of STONE's annotation-free claim. Nobody labels anything:
    the machine's own trajectory is the positive set, and everything else is
    scored by how far it sits from that distribution.
    """

    var fitted: Bool
    var support: Int
    """Number of traversed cells the fit is based on."""
    var mean: CellFeatures
    var inv_cov: Sym3

    def __init__(out self):
        self.fitted = False
        self.support = 0
        self.mean = CellFeatures(0.0, 0.0, 0.0)
        self.inv_cov = Sym3(1.0, 0.0, 0.0, 1.0, 0.0, 1.0)

    def squared_mahalanobis(self, f: CellFeatures) -> Float64:
        var d0 = f.slope - self.mean.slope
        var d1 = f.roughness - self.mean.roughness
        var d2 = f.step - self.mean.step
        # d^T S^-1 d, expanded for the symmetric case.
        return (
            d0 * d0 * self.inv_cov.xx
            + d1 * d1 * self.inv_cov.yy
            + d2 * d2 * self.inv_cov.zz
            + 2.0 * d0 * d1 * self.inv_cov.xy
            + 2.0 * d0 * d2 * self.inv_cov.xz
            + 2.0 * d1 * d2 * self.inv_cov.yz
        )


def traversed_cells(grid: TerrainGrid, trajectory: List[Vec3]) -> List[Int]:
    """Stage 3a. Cell indices the robot's own path crossed.

    Deduplicated, because a slow-moving machine parks hundreds of poses in one
    cell and would otherwise dominate the fitted distribution.
    """
    var seen = List[Bool](length=grid.spec.cell_count(), fill=False)
    var out = List[Int]()
    for i in range(len(trajectory)):
        var pose = trajectory[i]
        var idx = grid.spec.index_of(pose.x, pose.y)
        if idx < 0:
            continue
        if seen[idx]:
            continue
        if not grid.cells[idx].observed:
            continue
        seen[idx] = True
        out.append(idx)
    return out^


def fit_model(grid: TerrainGrid, trajectory: List[Vec3]) -> TraversabilityModel:
    """Stage 3b. Mean and inverse covariance of the driven terrain."""
    var model = TraversabilityModel()
    var idxs = traversed_cells(grid, trajectory)
    var n = len(idxs)
    model.support = n
    # Need more samples than dimensions for the covariance to mean anything.
    if n < 4:
        return model^

    var inv_n = 1.0 / Float64(n)
    var m0 = 0.0
    var m1 = 0.0
    var m2 = 0.0
    for i in range(n):
        var f = features_of(grid.cells[idxs[i]])
        m0 += f.slope
        m1 += f.roughness
        m2 += f.step
    m0 *= inv_n
    m1 *= inv_n
    m2 *= inv_n

    var c00 = 0.0
    var c01 = 0.0
    var c02 = 0.0
    var c11 = 0.0
    var c12 = 0.0
    var c22 = 0.0
    for i in range(n):
        var f = features_of(grid.cells[idxs[i]])
        var d0 = f.slope - m0
        var d1 = f.roughness - m1
        var d2 = f.step - m2
        c00 += d0 * d0
        c01 += d0 * d1
        c02 += d0 * d2
        c11 += d1 * d1
        c12 += d1 * d2
        c22 += d2 * d2

    # Bessel correction: this is a sample covariance, and the support can be
    # small on a short trajectory.
    var denom = 1.0 / Float64(n - 1)
    # Sample covariance, then raise the diagonal to the sensor noise floor.
    # See the *_NOISE_VAR docs: without this the fit is degenerate on uniform
    # terrain and the labeler rejects ground identical to what it just drove.
    var cov = Sym3(
        max(c00 * denom, SLOPE_NOISE_VAR),
        c01 * denom,
        c02 * denom,
        max(c11 * denom, ROUGHNESS_NOISE_VAR),
        c12 * denom,
        max(c22 * denom, STEP_NOISE_VAR),
    )

    model.mean = CellFeatures(m0, m1, m2)
    model.inv_cov = invert_sym3(cov, COVARIANCE_RIDGE)
    model.fitted = True
    return model^


def label_grid(
    grid: TerrainGrid,
    model: TraversabilityModel,
    threshold: Float64,
) -> List[Int]:
    """Stage 4. Propagate traversability beyond the driven trajectory.

    Returns one LABEL_* per cell. Unobserved cells stay UNKNOWN rather than
    defaulting to non-traversable: "we have not looked there" and "we looked and
    it is impassable" are different claims, and collapsing them is how an
    auto-labeler quietly teaches a planner to be a coward.
    """
    var labels = List[Int](length=grid.spec.cell_count(), fill=LABEL_UNKNOWN)
    if not model.fitted:
        return labels^
    for i in range(len(grid.cells)):
        var cell = grid.cells[i]
        if not cell.observed:
            continue
        var d2 = model.squared_mahalanobis(features_of(cell))
        if d2 <= threshold:
            labels[i] = LABEL_TRAVERSABLE
        else:
            labels[i] = LABEL_NON_TRAVERSABLE
    return labels^


# ===----------------------------------------------------------------------===
# Pipeline entry point
# ===----------------------------------------------------------------------===


struct StoneLabels(Movable):
    """Result of one auto-labeling pass."""

    var labels: List[Int]
    var support: Int
    """Traversed cells the model was fitted from. Low support means a label set
    that should not be trusted -- surface it rather than burying it."""
    var traversable: Int
    var non_traversable: Int
    var unknown: Int

    def __init__(out self, var labels: List[Int], support: Int):
        var t = 0
        var nt = 0
        var u = 0
        for i in range(len(labels)):
            var l = labels[i]
            if l == LABEL_TRAVERSABLE:
                t += 1
            elif l == LABEL_NON_TRAVERSABLE:
                nt += 1
            else:
                u += 1
        self.labels = labels^
        self.support = support
        self.traversable = t
        self.non_traversable = nt
        self.unknown = u


def auto_label(
    spec: GridSpec,
    points: List[Vec3],
    trajectory: List[Vec3],
    threshold: Float64 = CHI2_3DOF_95,
) -> StoneLabels:
    """Run the whole STONE pipeline over one aggregated cloud and trajectory.

    No human labels, no network, no checkpoint -- the trajectory is the only
    supervision.
    """
    var grid = TerrainGrid(spec)
    grid.ingest(points)
    grid.finalize()
    var model = fit_model(grid, trajectory)
    var labels = label_grid(grid, model, threshold)
    return StoneLabels(labels^, model.support)
