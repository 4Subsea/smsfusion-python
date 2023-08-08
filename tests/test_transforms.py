import numpy as np

from smsfusion import _transforms


class Test__angular_matrix_from_euler:
    def test_pure_roll(self):
        angular_matrix = _transforms._angular_matrix_from_euler(
            np.radians((30.0, 0.0, 0.0))
        )

        angular_matrix_expected = np.array(
            [[1.000, 0.000, 0.000], [0.000, 0.866, -0.500], [0.000, 0.500, 0.866]]
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[0], angular_matrix_expected, decimal=3
        )

    def test_pure_pitch(self):
        angular_matrix = _transforms._angular_matrix_from_euler(
            np.radians((0.0, 30.0, 0.0))
        )

        angular_matrix_expected = np.array(
            [[1.000, 0.000, 0.577], [0.000, 1.000, 0.000], [0.000, 0.000, 1.155]]
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[0], angular_matrix_expected, decimal=3
        )

    def test_pure_yaw(self):
        angular_matrix = _transforms._angular_matrix_from_euler(
            np.radians((0.0, 0.0, 30.0))
        )

        angular_matrix_expected = np.eye(3)

        np.testing.assert_array_almost_equal(
            angular_matrix[0], angular_matrix_expected, decimal=3
        )

    def test_references(self):
        # case 1
        angular_matrix = _transforms._angular_matrix_from_euler(
            np.radians((30.0, 15.0, 20.0))
        )

        angular_matrix_expected = np.array(
            [
                [1.0, 0.1339746, 0.23205081],
                [0.0, 0.8660254, -0.5],
                [0.0, 0.51763809, 0.89657547],
            ]
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[0], angular_matrix_expected, decimal=3
        )

        # case 2
        angular_matrix = _transforms._angular_matrix_from_euler(
            np.radians((15.0, 45.0, 5.0))
        )

        angular_matrix_expected = np.array(
            [
                [1.0, 0.25881905, 0.96592583],
                [0.0, 0.96592583, -0.25881905],
                [0.0, 0.3660254, 1.3660254],
            ]
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[0], angular_matrix_expected, decimal=3
        )

    def test_vectorized(self):
        euler_vector = np.array(
            [
                np.radians((30.0, 15.0, 20.0)),
                np.radians((15.0, 45.0, 5.0)),
                np.radians((25.0, 125.0, -35.0)),
            ]
        )
        angular_matrix = _transforms._angular_matrix_from_euler(euler_vector)

        angular_matrix_expected_0 = np.array(
            [
                [1.0, 0.1339746, 0.23205081],
                [0.0, 0.8660254, -0.5],
                [0.0, 0.51763809, 0.89657547],
            ]
        )

        angular_matrix_expected_1 = np.array(
            [
                [1.0, 0.25881905, 0.96592583],
                [0.0, 0.96592583, -0.25881905],
                [0.0, 0.3660254, 1.3660254],
            ]
        )

        angular_matrix_expected_2 = np.array(
            [
                [1.0, -0.60356143, -1.29434166],
                [0.0, 0.90630779, -0.42261826],
                [0.0, -0.73681245, -1.58009941],
            ]
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[0], angular_matrix_expected_0, decimal=3
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[1], angular_matrix_expected_1, decimal=3
        )

        np.testing.assert_array_almost_equal(
            angular_matrix[2], angular_matrix_expected_2, decimal=3
        )


class Test__rot_matrix_from_euler:
    """
    Reference cases are obtained from https://www.andre-gaschler.com/rotationconverter/
    and https://github.com/gaschler/rotationconverter.

    Note that Euler angles ZYX is selected, which corresponds to the rotation
    matrix defined in this package. (ZYX active rotaiton -> XYZ passive rotation).
    """

    def test_pure_roll(self):
        rot_matrix = _transforms._rot_matrix_from_euler(np.radians((30.0, 0.0, 0.0)))

        rot_matrix_expected = np.array(
            [[1.000, 0.000, 0.000], [0.000, 0.866, -0.500], [0.000, 0.500, 0.866]]
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[0], rot_matrix_expected, decimal=3
        )

    def test_pure_pitch(self):
        rot_matrix = _transforms._rot_matrix_from_euler(np.radians((0.0, 30.0, 0.0)))

        rot_matrix_expected = np.array(
            [[0.866, 0.000, 0.500], [0.000, 1.000, 0.000], [-0.500, 0.000, 0.866]]
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[0], rot_matrix_expected, decimal=3
        )

    def test_pure_yaw(self):
        rot_matrix = _transforms._rot_matrix_from_euler(np.radians((0.0, 0.0, 30.0)))

        rot_matrix_expected = np.array(
            [[0.866, -0.500, 0.000], [0.500, 0.866, 0.000], [0.000, 0.000, 1.000]]
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[0], rot_matrix_expected, decimal=3
        )

    def test_references(self):
        # case 1
        rot_matrix = _transforms._rot_matrix_from_euler(np.radians((30.0, 15.0, 20.0)))

        rot_matrix_expected = np.array(
            [
                [0.9076734, -0.1745930, 0.3816364],
                [0.3303661, 0.8580583, -0.3931846],
                [-0.2588190, 0.4829629, 0.8365163],
            ]
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[0], rot_matrix_expected, decimal=3
        )

        # case 2
        rot_matrix = _transforms._rot_matrix_from_euler(np.radians((15.0, 45.0, 5.0)))

        rot_matrix_expected = np.array(
            [
                [0.7044160, 0.0981303, 0.7029712],
                [0.0616284, 0.9782008, -0.1983057],
                [-0.7071068, 0.1830127, 0.6830127],
            ]
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[0], rot_matrix_expected, decimal=3
        )

    def test_vectorized(self):
        euler_vector = np.array(
            [
                np.radians((30.0, 15.0, 20.0)),
                np.radians((15.0, 45.0, 5.0)),
                np.radians((25.0, 125.0, -35.0)),
            ]
        )
        rot_matrix = _transforms._rot_matrix_from_euler(euler_vector)

        rot_matrix_expected_0 = np.array(
            [
                [0.9076734, -0.1745930, 0.3816364],
                [0.3303661, 0.8580583, -0.3931846],
                [-0.2588190, 0.4829629, 0.8365163],
            ]
        )

        rot_matrix_expected_1 = np.array(
            [
                [0.7044160, 0.0981303, 0.7029712],
                [0.0616284, 0.9782008, -0.1983057],
                [-0.7071068, 0.1830127, 0.6830127],
            ]
        )

        rot_matrix_expected_2 = np.array(
            [
                [-0.4698463, 0.8034179, 0.3657378],
                [0.3289899, 0.5438383, -0.7720140],
                [-0.8191521, -0.2424039, -0.5198368],
            ]
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[0], rot_matrix_expected_0, decimal=3
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[1], rot_matrix_expected_1, decimal=3
        )

        np.testing.assert_array_almost_equal(
            rot_matrix[2], rot_matrix_expected_2, decimal=3
        )
