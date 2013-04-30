import numpy as np

try:
    import vincent
    assert vincent
except ImportError:
    vincent = None


# If no time vector or final time is given when generating step/impulse
# responses, the final time is _t_end_scale / min(abs(real(poles))), rounded
# up to the nearest sample.
_t_end_scale = 6.0

# If the minimum pole of a system is less than this, then the default number
# of samples is used to compute a step/impulse response.
_min_response_eig = 1e-12

# If the minimum pole of a system is too small, then this many samples are
# used to construct a discrete-time step/impulse response.
_discrete_default_response_samples = 50

# If no time vector is given when generating step/impulse responses for
# continuous-time systems, this many samples are used.
_N_continuous_time_samples = 200


def stairs(line_data):
    """Given a list/ndarray of data, return data in "stairs" format, so that
    lines plotted with the new data would appear as the old data sampled via
    ZOH."""
    pass


class DiscreteStateSpace(object):
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self.__poles = [p for p in np.linalg.eigvals(self.A)]
        self.__order = len(self.poles)
        self.__input_dim = self.B.shape[1]
        self.__output_dim = self.C.shape[0]

    def step(self, t_final=None, plot=True):
        """Plot the step response of a discrete-time LTI system.

        Parameters
        ----------
        t_final: scalar
            End time for response calculation. If None, then the default is
            used.*
        plot: boolean, default True
            If True, plot the response.
        """
        if plot:
            raise NotImplementedError('plotting not yet implemented')

        if t_final:
            N = t_final
        else:
            min_pole = min(abs(np.real(self.poles)))
            if abs(min_pole) < _min_response_eig:
                N = _discrete_default_response_samples
            else:
                N = int(np.ceil(_t_end_scale / min_pole))

        response = np.zeros((self.output_dim, N))
        response[:, 0] = np.array(self.D).flatten()
        for i in xrange(1, N):
            response[:, i] = response[:, i - 1] + np.array(
                self.C * np.linalg.matrix_power(self.A, i - 1) * self.B
            ).flatten()
        return response

    @property
    def poles(self):
        return self.__poles

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def output_dim(self):
        return self.__output_dim
