import unittest

import numpy as np

from controlsys import DiscreteStateSpace


class TestDiscreteStateSpace(unittest.TestCase):
    def test_sys_siso(self):
        A = np.matrix('0.0, 1.0; -0.5, -0.5')
        B = np.matrix('0.0; 1.0')
        C = np.matrix('0.0, 1.0')
        D = np.matrix('0.3')
        return A, B, C, D

    def test_sys_simo(self):
        A = np.matrix('0.0, 1.0; -0.5, -0.5')
        B = np.matrix('0.0; 1.0')
        C = np.matrix('0.0, 1.0; 0.2, -2.0')
        D = np.matrix('0.3; -0.3')
        return A, B, C, D

    def test_step_response_calculation(self):
        """Step response generation
        """
        # SISO
        A, B, C, D = self.test_sys_siso()
        expected_step = np.array([0.3, 1.3, 0.8, 0.55, 0.925, 0.8625])
        N = len(expected_step)

        dss = DiscreteStateSpace(A, B, C, D)
        actual_step = dss.step(t_final=N, plot=False).tolist()[0]
        np.testing.assert_array_almost_equal(expected_step, actual_step)

        # SIMO
        A, B, C, D = self.test_sys_simo()
        expected_step = np.array([
            [0.3, 1.3, 0.8, 0.55, 0.925, 0.8625],
            [-0.3, -2.3, -1.1, -0.7, -1.5, -1.3]])

        dss = DiscreteStateSpace(A, B, C, D)
        actual_step = dss.step(t_final=N, plot=False).tolist()
        np.testing.assert_array_almost_equal(expected_step, actual_step)

    def test_system_matrices_error_check(self):
        """Invalid system matrices raise errors
        """

    def test_impulse_response(self):
        """Impulse response generation
        """


# TODO:
# - Discrete state-space systems with sampling times
