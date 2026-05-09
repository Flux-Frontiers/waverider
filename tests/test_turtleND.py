"""Tests for TurtleND - verifies N-dimensional turtle behavior
and consistency with Turtle3D for 3D operations."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from waverider.turtle3D import Turtle3D
from waverider.turtleND import TurtleND

_tolerance = 1e-8


class TestTurtleNDInit(unittest.TestCase):
    def test_init_3d(self):
        t = TurtleND(3)
        assert_allclose(t.position, [0, 0, 0])
        assert_allclose(t.heading, [1, 0, 0])
        assert_allclose(t.left, [0, 1, 0])
        assert_allclose(t.up, [0, 0, 1])

    def test_init_5d(self):
        t = TurtleND(5)
        assert_allclose(t.position, np.zeros(5))
        assert_allclose(t.frame, np.eye(5))

    def test_init_2d(self):
        t = TurtleND(2)
        assert_allclose(t.heading, [1, 0])
        assert_allclose(t.left, [0, 1])

    def test_init_rejects_1d(self):
        with self.assertRaises(ValueError):
            TurtleND(1)

    def test_up_rejects_2d(self):
        t = TurtleND(2)
        with self.assertRaises(ValueError):
            _ = t.up


class TestTurtleNDMove(unittest.TestCase):
    def test_move_forward(self):
        t = TurtleND(3)
        t.move(10.0)
        assert_allclose(t.position, [10, 0, 0], atol=_tolerance)

    def test_move_backward(self):
        t = TurtleND(3)
        t.move(10.0)
        t.move(-5.0)
        assert_allclose(t.position, [5, 0, 0], atol=_tolerance)

    def test_move_5d(self):
        t = TurtleND(5)
        t.move(3.0)
        expected = np.zeros(5)
        expected[0] = 3.0
        assert_allclose(t.position, expected, atol=_tolerance)

    def test_move_after_turn(self):
        t = TurtleND(3)
        t.turn(90)
        t.move(5.0)
        assert_allclose(t.position, [0, 5, 0], atol=_tolerance)

    def test_recording(self):
        t = TurtleND(3)
        t.recording = True
        t.move(1.0)
        t.move(1.0)
        assert len(t.tape) == 2
        assert_allclose(t.tape[0], [1, 0, 0], atol=_tolerance)
        assert_allclose(t.tape[1], [2, 0, 0], atol=_tolerance)


class TestTurtleNDTurn(unittest.TestCase):
    def test_turn_right_90(self):
        t = TurtleND(3)
        t.turn(90)
        assert_allclose(t.heading, [0, 1, 0], atol=_tolerance)

    def test_turn_left_90(self):
        t = TurtleND(3)
        t.turn(-90)
        assert_allclose(t.heading, [0, -1, 0], atol=_tolerance)

    def test_turn_360_identity(self):
        t = TurtleND(3)
        t.turn(360)
        assert_allclose(t.heading, [1, 0, 0], atol=_tolerance)

    def test_turn_2d(self):
        t = TurtleND(2)
        t.turn(90)
        assert_allclose(t.heading, [0, 1], atol=_tolerance)
        t.move(5.0)
        assert_allclose(t.position, [0, 5], atol=_tolerance)


class TestTurtleNDRoll(unittest.TestCase):
    def test_roll_90(self):
        t = TurtleND(3)
        t.roll(90)
        assert_allclose(t.left, [0, 0, 1], atol=_tolerance)
        assert_allclose(t.up, [0, -1, 0], atol=_tolerance)

    def test_roll_preserves_heading(self):
        t = TurtleND(3)
        t.roll(45)
        assert_allclose(t.heading, [1, 0, 0], atol=_tolerance)

    def test_roll_rejects_2d(self):
        t = TurtleND(2)
        with self.assertRaises(ValueError):
            t.roll(90)


class TestTurtleNDPitch(unittest.TestCase):
    def test_pitch_90(self):
        t = TurtleND(3)
        t.pitch(90)
        assert_allclose(t.heading, [0, 0, -1], atol=_tolerance)
        assert_allclose(t.up, [1, 0, 0], atol=_tolerance)

    def test_pitch_preserves_left(self):
        t = TurtleND(3)
        t.pitch(30)
        assert_allclose(t.left, [0, 1, 0], atol=_tolerance)


class TestTurtleNDYaw(unittest.TestCase):
    def test_yaw_90(self):
        """Yaw(90) should give heading=[0,1,0], left=[-1,0,0]."""
        t = TurtleND(3)
        t.yaw(90)
        assert_allclose(t.heading, [0, 1, 0], atol=_tolerance)
        assert_allclose(t.left, [-1, 0, 0], atol=_tolerance)


class TestTurtleND3DConsistency(unittest.TestCase):
    """Verify that TurtleND(3) produces the same results as Turtle3D."""

    def _compare(self, t3d: Turtle3D, tnd: TurtleND):
        assert_allclose(tnd.position, t3d._position, atol=_tolerance)
        assert_allclose(tnd.heading, t3d._heading, atol=_tolerance)
        assert_allclose(tnd.left, t3d._left, atol=_tolerance)
        assert_allclose(tnd.up, t3d._up, atol=_tolerance)

    def test_move_consistency(self):
        t3 = Turtle3D()
        tn = TurtleND(3)
        t3.move(7.5)
        tn.move(7.5)
        self._compare(t3, tn)

    def test_turn_consistency(self):
        t3 = Turtle3D()
        tn = TurtleND(3)
        t3.turn(45)
        tn.turn(45)
        self._compare(t3, tn)

    def test_roll_consistency(self):
        t3 = Turtle3D()
        tn = TurtleND(3)
        t3.roll(60)
        tn.roll(60)
        self._compare(t3, tn)

    def test_pitch_consistency(self):
        t3 = Turtle3D()
        tn = TurtleND(3)
        t3.pitch(90)
        tn.pitch(90)
        self._compare(t3, tn)

    def test_yaw_consistency(self):
        t3 = Turtle3D()
        tn = TurtleND(3)
        t3.yaw(90)
        tn.yaw(90)
        self._compare(t3, tn)

    def test_compound_operations(self):
        """A complex sequence of operations should match between 3D and ND."""
        t3 = Turtle3D()
        tn = TurtleND(3)

        ops = [
            ("move", 5.0),
            ("turn", 30),
            ("move", 3.0),
            ("pitch", 45),
            ("roll", 60),
            ("move", 2.0),
            ("yaw", 110),
            ("move", 1.5),
            ("turn", -20),
            ("pitch", 15),
            ("roll", 240),
        ]

        for op, val in ops:
            getattr(t3, op)(val)
            getattr(tn, op)(val)

        self._compare(t3, tn)


class TestTurtleNDCoordinateTransforms(unittest.TestCase):
    def test_to_local_at_origin(self):
        t = TurtleND(3)
        local = t.to_local([3, 4, 5])
        assert_allclose(local, [3, 4, 5], atol=_tolerance)

    def test_to_global_at_origin(self):
        t = TurtleND(3)
        glob = t.to_global([3, 4, 5])
        assert_allclose(glob, [3, 4, 5], atol=_tolerance)

    def test_roundtrip(self):
        """to_global(to_local(v)) should return v."""
        t = TurtleND(3)
        t.move(5)
        t.turn(37)
        t.pitch(22)
        t.roll(15)
        t.move(3)

        point = np.array([7.0, -2.0, 4.0])
        local = t.to_local(point)
        recovered = t.to_global(local)
        assert_allclose(recovered, point, atol=_tolerance)

    def test_roundtrip_5d(self):
        """Roundtrip in 5D."""
        t = TurtleND(5)
        t.rotate(45, 0, 1)
        t.rotate(30, 2, 3)
        t.rotate(60, 1, 4)
        t.move(10)

        point = np.array([1, 2, 3, 4, 5], dtype="d")
        local = t.to_local(point)
        recovered = t.to_global(local)
        assert_allclose(recovered, point, atol=_tolerance)

    def test_3d_consistency_to_local(self):
        """to_local should match Turtle3D."""
        t3 = Turtle3D()
        tn = TurtleND(3)

        t3.move(5)
        tn.move(5)
        t3.turn(30)
        tn.turn(30)

        point = np.array([7.0, 3.0, -1.0])
        local_3d = t3.to_local(point)
        local_nd = tn.to_local(point)
        assert_allclose(local_nd, local_3d, atol=_tolerance)

    def test_3d_consistency_to_global(self):
        """to_global should match Turtle3D."""
        t3 = Turtle3D()
        tn = TurtleND(3)

        t3.move(5)
        tn.move(5)
        t3.turn(30)
        tn.turn(30)

        local = np.array([2.0, -1.0, 3.0])
        global_3d = t3.to_global(local)
        global_nd = tn.to_global(local)
        assert_allclose(global_nd, global_3d, atol=_tolerance)


class TestTurtleNDHighDim(unittest.TestCase):
    def test_rotate_5d(self):
        """Rotating in the (3,4) plane should only affect basis[3] and basis[4]."""
        t = TurtleND(5)
        t.rotate(90, 3, 4)
        # heading, left, up should be unchanged
        assert_allclose(t.heading, [1, 0, 0, 0, 0], atol=_tolerance)
        assert_allclose(t.left, [0, 1, 0, 0, 0], atol=_tolerance)
        assert_allclose(t.up, [0, 0, 1, 0, 0], atol=_tolerance)
        # basis[3] rotated toward basis[4]
        assert_allclose(t.basis(3), [0, 0, 0, 0, 1], atol=_tolerance)
        assert_allclose(t.basis(4), [0, 0, 0, -1, 0], atol=_tolerance)

    def test_frame_stays_orthonormal(self):
        """After many rotations the frame should remain orthonormal."""
        t = TurtleND(7)
        for i in range(6):
            for j in range(i + 1, 7):
                t.rotate(17 * (i + 1) * (j + 1), i, j)

        # Check orthonormality: frame @ frame.T should be identity
        product = t.frame @ t.frame.T
        assert_allclose(product, np.eye(7), atol=1e-6)

    def test_move_in_high_dim(self):
        """Move after rotating heading into a higher dimension."""
        t = TurtleND(5)
        t.rotate(90, 0, 3)  # heading now points along axis 3
        t.move(4.0)
        expected = np.zeros(5)
        expected[3] = 4.0
        assert_allclose(t.position, expected, atol=_tolerance)

    def test_orthonormalize(self):
        t = TurtleND(4)
        # Slightly perturb the frame
        t._frame[0] *= 1.001
        t._frame[1] += 0.001 * t._frame[0]
        t.orthonormalize()
        product = t.frame @ t.frame.T
        assert_allclose(product, np.eye(4), atol=1e-10)


class TestTurtleNDMisc(unittest.TestCase):
    def test_copy_coords(self):
        t1 = TurtleND(4)
        t1.move(5)
        t1.rotate(45, 0, 1)
        t2 = TurtleND(4)
        t2.copy_coords(t1)
        assert_allclose(t2.position, t1.position, atol=_tolerance)
        assert_allclose(t2.frame, t1.frame, atol=_tolerance)

    def test_copy_rejects_different_ndim(self):
        t1 = TurtleND(3)
        t2 = TurtleND(5)
        with self.assertRaises(ValueError):
            t2.copy_coords(t1)

    def test_reset(self):
        t = TurtleND(4)
        t.move(10)
        t.rotate(30, 0, 1)
        t.reset()
        assert_allclose(t.position, np.zeros(4))
        assert_allclose(t.frame, np.eye(4))

    def test_repr(self):
        t = TurtleND(3, name="Test")
        s = repr(t)
        assert "Test" in s
        assert "3D" in s

    def test_rotate_bad_indices(self):
        t = TurtleND(3)
        with self.assertRaises(IndexError):
            t.rotate(45, 0, 5)
        with self.assertRaises(ValueError):
            t.rotate(45, 1, 1)


if __name__ == "__main__":
    unittest.main()
