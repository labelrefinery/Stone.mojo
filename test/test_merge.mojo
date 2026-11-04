"""Building a grid from patches, against streaming the same points."""

from std.math import abs, cos, sin
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from stone import CellAccumulator, CellPatch, GridSpec, TerrainGrid, Vec3


def _sweep(seed: Int, n: Int) -> List[Vec3]:
    """Deterministic points over and past a 20 m grid: some fall outside."""
    var out = List[Vec3](capacity=n)
    for i in range(n):
        var a = Float64(i * 7 + seed * 131) * 0.0137
        var r = Float64((i * 31 + seed * 17) % 1300) / 100.0
        out.append(
            Vec3(10.0 + r * cos(a), 10.0 + r * sin(a), 0.3 * sin(a * 3.0))
        )
    return out^


def _spec() -> GridSpec:
    return GridSpec(0.0, 0.0, 1.0, 20, 20)


def _close(a: Float64, b: Float64) -> Bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def test_patch_is_sparse_and_sorted() raises:
    var patch = CellPatch.of(_spec(), _sweep(1, 2000))
    assert_true(len(patch) > 0)
    assert_true(len(patch) < _spec().cell_count())
    for i in range(1, len(patch.indices)):
        assert_true(patch.indices[i - 1] < patch.indices[i])
    for i in range(len(patch)):
        assert_true(patch.accumulators[i].count > 0)


def test_merged_equals_streamed() raises:
    """Count, z_min and z_max exactly; the sums to rounding."""
    var streamed = TerrainGrid(_spec())
    var merged = TerrainGrid(_spec())
    for s in range(5):
        var pts = _sweep(s, 3000)
        streamed.ingest(pts)
        merged.merge(CellPatch.of(_spec(), pts))
    for i in range(_spec().cell_count()):
        ref a = streamed.accumulators[i]
        ref b = merged.accumulators[i]
        assert_equal(a.count, b.count)
        assert_equal(a.z_min, b.z_min)
        assert_equal(a.z_max, b.z_max)
        assert_true(_close(a.sum_z, b.sum_z))
        assert_true(_close(a.sum_xx, b.sum_xx))
        assert_true(_close(a.sum_yz, b.sum_yz))
    streamed.finalize()
    merged.finalize()
    for i in range(_spec().cell_count()):
        assert_equal(streamed.cells[i].observed, merged.cells[i].observed)
        assert_equal(streamed.cells[i].step, merged.cells[i].step)


def test_merge_is_deterministic() raises:
    var patches = List[CellPatch]()
    for s in range(4):
        patches.append(CellPatch.of(_spec(), _sweep(s, 2500)))
    var a = TerrainGrid(_spec())
    var b = TerrainGrid(_spec())
    for p in patches:
        a.merge(p)
    for p in patches:
        b.merge(p)
    for i in range(_spec().cell_count()):
        assert_equal(a.accumulators[i].sum_xy, b.accumulators[i].sum_xy)
        assert_equal(a.accumulators[i].sum_zz, b.accumulators[i].sum_zz)


def test_merge_into_empty_and_from_empty() raises:
    var acc = CellAccumulator()
    var other = CellAccumulator()
    other.add(Vec3(1.0, 2.0, -3.0))
    acc.merge(other)
    assert_equal(acc.count, 1)
    assert_equal(acc.z_min, -3.0)
    assert_equal(acc.z_max, -3.0)
    acc.merge(CellAccumulator())
    assert_equal(acc.count, 1)
    assert_equal(acc.z_min, -3.0)


def test_other_grid_refused() raises:
    var grid = TerrainGrid(_spec())
    var patch = CellPatch.of(GridSpec(0.5, 0.0, 1.0, 20, 20), _sweep(1, 100))
    with assert_raises(contains="another grid"):
        grid.merge(patch)
    assert_false(_spec().same_as(GridSpec(0.0, 0.0, 1.0, 20, 21)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
