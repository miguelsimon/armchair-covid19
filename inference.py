import unittest
from typing import NamedTuple, Tuple

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


class Coord(NamedTuple):
    infected: float
    alpha: float
    beta_0: float
    beta_1: float
    gamma: float


class Spec(NamedTuple):
    """
    Full specification for an inference problem

    Parameters
    ----------

    population : float
        Total population for the region being studied
    deaths : ndarray
        daily death counts due to disease
    b_infected : Tuple[float]
        lower and upper bounds on the possible infected population on day 0
    b_alpha : Tuple[float]
        lu bounds on infection fatality rate
    b_beta_0 : Tuple[float]
        lu bounds on SIR beta prior to lockdown
    b_beta_1 : Tuple[float]
        lu bounds on SIR beta after lockdown
    t_lock : float
        time lockdown went into effect
    b_gamma : Tuple[float]
        lu bounds on SIR gamma
    """

    population: float
    deaths: ndarray
    b_infected: Tuple[float, float]
    b_alpha: Tuple[float, float]
    b_beta_0: Tuple[float, float]
    b_beta_1: Tuple[float, float]
    t_lock: float
    b_gamma: Tuple[float, float]

    def sample(self) -> Coord:
        uniform = np.random.uniform

        return Coord(
            infected=uniform(*self.b_infected),
            alpha=uniform(*self.b_alpha),
            beta_0=uniform(*self.b_beta_0),
            beta_1=uniform(*self.b_beta_1),
            gamma=uniform(*self.b_gamma),
        )

    def mcmc_propose(self, coord: Coord) -> Coord:
        def proposal(x: float, bounds: Tuple[float, float]) -> float:
            """
            Perturb by a delta sampled from a gaussian at 1/50 scale of the bounds
            """
            scale = (bounds[1] - bounds[0]) / 50
            delta = np.random.normal() * scale
            new_x = x + delta
            return np.clip(new_x, bounds[0], bounds[1])

        return Coord(
            infected=proposal(coord.infected, self.b_infected),
            alpha=proposal(coord.alpha, self.b_alpha),
            beta_0=proposal(coord.beta_0, self.b_beta_0),
            beta_1=proposal(coord.beta_1, self.b_beta_1),
            gamma=proposal(coord.gamma, self.b_gamma),
        )


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

    def test_Spec(self):
        spec = Spec(
            population=6.55 * 1000000,
            deaths=np.array([0, 0, 0, 0, 1]),
            b_infected=(1.0, 100000.0),
            b_alpha=(0.0001, 0.01),
            b_beta_0=(0.05, 0.4),
            b_beta_1=(0.05, 0.4),
            t_lock=4.0,
            b_gamma=(1 / 40.0, 1 / 10.0),
        )

        coord = spec.sample()
        print(coord, spec.mcmc_propose(coord))
