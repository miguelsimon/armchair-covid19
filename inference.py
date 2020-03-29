import unittest

import numpy as np
import scipy.stats
from numpy import ndarray

from sir import SIR, Beta, Recoveries


def get_death_ll(deaths: ndarray, alpha: float, recoveries: ndarray) -> float:
    means = recoveries * alpha
    variances = recoveries * alpha * (1 - alpha) + 0.001
    scales = np.sqrt(variances)
    log_lls = scipy.stats.norm.logpdf(deaths, loc=means, scale=scales)
    return np.sum(log_lls)


class DeathLLGivenModel:
    def __init__(self, deaths: ndarray):
        self.deaths = deaths

    def __call__(
        self,
        alpha: float,
        state_0: ndarray,
        beta_0: float,
        beta_1: float,
        t_lock: float,
        gamma: float,
    ) -> float:

        beta = Beta(beta_0, beta_1, t_lock)
        sir = SIR(beta, gamma, state_0)

        rec = Recoveries(sir)
        recoveries_list = []
        for i, _ in enumerate(self.deaths):
            recoveries_list.append(rec(i))
        recoveries = np.array(recoveries_list)

        return get_death_ll(self.deaths, alpha, recoveries)


class Test(unittest.TestCase):
    def test_get_death_ll(self):
        # Check that the normal approximation is correct

        deaths = np.array([100])
        recoveries = np.array([1000])
        alpha = 0.1

        log_ll = get_death_ll(deaths, alpha, recoveries)

        binomial_ll = scipy.stats.binom.logpmf(deaths[0], recoveries[0], alpha)

        error = np.abs(log_ll - binomial_ll)
        self.assertTrue(error < 0.01)

    def test_DeathLLGivenModel(self):
        R0 = 2.5
        gamma = 1 / 20.0
        b = R0 * gamma
        state0 = np.array([6.55 * 1000000, 1000, 0])

        beta = Beta(b, b, 1)
        sir = SIR(beta, gamma, state0)
        rec = Recoveries(sir)
        alpha = 0.1
        deaths_list = []

        for i in range(200):
            deaths_list.append(alpha * rec(i))
        deaths = np.array(deaths_list)

        m = DeathLLGivenModel(deaths)

        lls = []
        for test_b in [b, 0.05, 0.3]:
            lls.append(m(alpha, state0, test_b, test_b, 0, gamma))

        # b should maximize log likelihood
        self.assertEqual(np.argmax(lls), 0)
