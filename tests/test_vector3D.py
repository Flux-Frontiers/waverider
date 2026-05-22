"""Tests for Vector3D class and associated module-level functions."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from waverider.vector3D import (
    Vector3D,
    calc_angle,
    calc_dihedral,
    calculate_bond_angle,
    distance3d,
    rms_difference,
)

_tol = 1e-10


class TestVector3DInit:
    def test_xyz_init(self):
        v = Vector3D(1.0, 2.0, 3.0)
        assert v[0] == pytest.approx(1.0)
        assert v[1] == pytest.approx(2.0)
        assert v[2] == pytest.approx(3.0)

    def test_list_init(self):
        v = Vector3D([4.0, 5.0, 6.0])
        assert_allclose(v.get_array(), [4.0, 5.0, 6.0])

    def test_array_init(self):
        arr = np.array([7.0, 8.0, 9.0])
        v = Vector3D(arr)
        assert_allclose(v.get_array(), arr)

    def test_bad_length_raises(self):
        with pytest.raises((ValueError, Exception)):
            Vector3D([1.0, 2.0])

    def test_repr(self):
        v = Vector3D(1.0, 2.0, 3.0)
        r = repr(v)
        assert "1.00" in r and "2.00" in r and "3.00" in r


class TestVector3DArithmetic:
    def test_add_vectors(self):
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        r = v1 + v2
        assert_allclose(r.get_array(), [5, 7, 9])

    def test_sub_vectors(self):
        v1 = Vector3D(4, 5, 6)
        v2 = Vector3D(1, 2, 3)
        r = v1 - v2
        assert_allclose(r.get_array(), [3, 3, 3])

    def test_neg(self):
        v = Vector3D(1, -2, 3)
        r = -v
        assert_allclose(r.get_array(), [-1, 2, -3])

    def test_truediv(self):
        v = Vector3D(2, 4, 6)
        r = v / 2
        assert_allclose(r.get_array(), [1, 2, 3])

    def test_dot_product_via_mul(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        assert v1 * v2 == pytest.approx(0.0)

    def test_dot_product_nonzero(self):
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        assert v1 * v2 == pytest.approx(32.0)

    def test_cross_product_via_pow(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        r = v1**v2
        assert_allclose(r.get_array(), [0, 0, 1], atol=_tol)

    def test_cross_product_anticommutes(self):
        v1 = Vector3D(1, 2, 3)
        v2 = Vector3D(4, 5, 6)
        c1 = (v1**v2).get_array()
        c2 = (v2**v1).get_array()
        assert_allclose(c1, -c2, atol=_tol)

    def test_scalar_mul_via_pow(self):
        v = Vector3D(1, 2, 3)
        r = v**3
        assert_allclose(r.get_array(), [3, 6, 9])


class TestVector3DGeometry:
    def test_magnitude_unit_vector(self):
        v = Vector3D(1, 0, 0)
        assert v.magnitude() == pytest.approx(1.0)

    def test_magnitude_general(self):
        v = Vector3D(3, 4, 0)
        assert v.magnitude() == pytest.approx(5.0)

    def test_magnitude_squared(self):
        v = Vector3D(1, 2, 2)
        assert v.magnitude_squared() == pytest.approx(9.0)

    def test_normalize_in_place(self):
        v = Vector3D(3, 0, 0)
        v.normalize()
        assert v.magnitude() == pytest.approx(1.0)
        assert v[0] == pytest.approx(1.0)

    def test_normalized_returns_copy(self):
        v = Vector3D(0, 4, 0)
        n = v.normalized()
        assert n.magnitude() == pytest.approx(1.0)
        assert v.magnitude() == pytest.approx(4.0)  # original unchanged

    def test_angle_with_orthogonal(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 1, 0)
        assert v1.angle_with(v2) == pytest.approx(90.0, abs=1e-6)

    def test_angle_with_parallel(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(2, 0, 0)
        assert v1.angle_with(v2) == pytest.approx(0.0, abs=1e-6)

    def test_angle_with_antiparallel(self):
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(-1, 0, 0)
        assert v1.angle_with(v2) == pytest.approx(180.0, abs=1e-6)


class TestVector3DAccessors:
    def test_getitem(self):
        v = Vector3D(1, 2, 3)
        assert v[0] == 1.0
        assert v[1] == 2.0
        assert v[2] == 3.0

    def test_setitem(self):
        v = Vector3D(1, 2, 3)
        v[1] = 99.0
        assert v[1] == 99.0

    def test_contains(self):
        v = Vector3D(1, 2, 3)
        assert 2.0 in v
        assert 99.0 not in v

    def test_copy_is_independent(self):
        v1 = Vector3D(1, 2, 3)
        v2 = v1.copy()
        v2[0] = 99.0
        assert v1[0] == pytest.approx(1.0)

    def test_get_array_is_copy(self):
        v = Vector3D(1, 2, 3)
        arr = v.get_array()
        arr[0] = 99.0
        assert v[0] == pytest.approx(1.0)


class TestCalcAngle:
    def test_right_angle(self):
        # L-shaped: v1-(0,0,0)-v3, angle at origin = 90°
        v1 = Vector3D(1, 0, 0)
        v2 = Vector3D(0, 0, 0)
        v3 = Vector3D(0, 1, 0)
        assert calc_angle(v1, v2, v3) == pytest.approx(90.0, abs=1e-6)

    def test_straight_angle(self):
        v1 = Vector3D(-1, 0, 0)
        v2 = Vector3D(0, 0, 0)
        v3 = Vector3D(1, 0, 0)
        assert calc_angle(v1, v2, v3) == pytest.approx(180.0, abs=1e-6)


class TestCalcDihedral:
    def test_known_dihedral(self):
        v1 = Vector3D(0.0, 0.0, 0.0)
        v2 = Vector3D(1.0, 0.0, 0.0)
        v3 = Vector3D(1.0, 1.0, 1.0)
        v4 = Vector3D(1.0, 1.0, 2.0)
        assert abs(calc_dihedral(v1, v2, v3, v4)) == pytest.approx(90.0, abs=0.5)

    def test_planar_dihedral_zero(self):
        # All points in xy-plane — dihedral should be ~0
        v1 = Vector3D(0, 1, 0)
        v2 = Vector3D(0, 0, 0)
        v3 = Vector3D(1, 0, 0)
        v4 = Vector3D(1, 1, 0)
        result = calc_dihedral(v1, v2, v3, v4)
        assert abs(result) < 5.0  # allow some numerical slack for planar case


class TestDistance3D:
    def test_unit_distance(self):
        p1 = Vector3D(1, 0, 0)
        p2 = Vector3D(0, 0, 0)
        assert distance3d(p1, p2) == pytest.approx(1.0)

    def test_known_distance(self):
        p1 = Vector3D(1, 0, 0)
        p2 = Vector3D(0, 1, 0)
        assert distance3d(p1, p2) == pytest.approx(math.sqrt(2), rel=1e-6)

    def test_zero_distance(self):
        p = Vector3D(3, 4, 5)
        assert distance3d(p, p) == pytest.approx(0.0)


class TestRmsDifference:
    def test_identical_arrays(self):
        a = np.array([1.0, 2.0, 3.0])
        assert rms_difference(a, a) == pytest.approx(0.0)

    def test_known_rms(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        # diffs = [3, 4], squares = [9, 16], mean = 12.5, sqrt = 3.535...
        assert rms_difference(a, b) == pytest.approx(math.sqrt(12.5), rel=1e-6)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            rms_difference(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rms_difference(np.array([]), np.array([]))


class TestCalculateBondAngle:
    def test_right_angle(self):
        atom1 = [1, 0, 0]
        atom2 = [0, 0, 0]
        atom3 = [0, 1, 0]
        assert calculate_bond_angle(atom1, atom2, atom3) == pytest.approx(90.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        # Degenerate: atom1 == atom2 → zero vector → returns 0.0
        result = calculate_bond_angle([0, 0, 0], [0, 0, 0], [1, 0, 0])
        assert result == pytest.approx(0.0)
