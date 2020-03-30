import unittest
from typing import List, NamedTuple, Tuple

import numpy as np
import scipy.stats
from numpy import ndarray

from sir import SIR, Beta, Recoveries


def get_death_ll(
    deaths: ndarray, alpha: float, recoveries: ndarray, fuzz: float
) -> float:
    """

    Normal approximation to binomial log probability.

    assumes deaths[i] ~ B(recoveries[i], alpha)

    Parameters
    ----------

    deaths : ndarray
        daily death counts
    alpha : float
        probability that a recovery is a death
    recoveries : ndarray
        daily recovery counts

    Returns
    -------

    float
        sum of log probabilities for all days

    """

    means = recoveries * alpha
    variances = recoveries * alpha * (1 - alpha)
    scales = np.sqrt(variances) + fuzz
    log_lls = scipy.stats.norm.logpdf(deaths, loc=means, scale=scales)
    return np.sum(log_lls)


class DeathLLGivenModel:
    """
    Calculates the log likelihood of observed deaths given the model parameters
    """

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
        fuzz: float,
    ) -> float:

        beta = Beta(beta_0, beta_1, t_lock)
        sir = SIR(beta, gamma, state_0)

        rec = Recoveries(sir)

        recoveries = rec.apply_to_span(len(self.deaths))

        return get_death_ll(self.deaths, alpha, recoveries, fuzz)


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
    fuzz : float
        add this to the scale of the Normal approximation of the binomial
    """

    population: float
    deaths: ndarray
    b_infected: Tuple[float, float]
    b_alpha: Tuple[float, float]
    b_beta_0: Tuple[float, float]
    b_beta_1: Tuple[float, float]
    t_lock: float
    b_gamma: Tuple[float, float]
    fuzz: float

    def sample(self) -> Coord:
        """
        Return a random sample from the prior
        """
        uniform = np.random.uniform

        return Coord(
            infected=uniform(*self.b_infected),
            alpha=uniform(*self.b_alpha),
            beta_0=uniform(*self.b_beta_0),
            beta_1=uniform(*self.b_beta_1),
            gamma=uniform(*self.b_gamma),
        )

    def mcmc_propose(self, coord: Coord) -> Coord:
        """
        Sample from the Metropolis-Hastings jump distribution given a coord
        """

        def proposal(x: float, bounds: Tuple[float, float]) -> float:
            """
            Perturb by a delta sampled from a gaussian at scale dependent on the bounds
            """
            scale = (bounds[1] - bounds[0]) / 100
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


class MCMCSampler:
    """
    Trivial Metropolis-Hastings MCMC sampler.

    Parameters
    ----------

    spec : Spec
        inference problem specification

    """

    def __init__(self, spec: Spec):
        self.spec = spec

        coord = spec.sample()

        self.samples: List[Coord] = [coord]
        self.logps: List[float] = [self.get_logp(coord)]

    def get_logp(self, coord: Coord) -> float:
        state0 = np.array([self.spec.population - coord.infected, coord.infected, 0])

        m = DeathLLGivenModel(self.spec.deaths)

        return m(
            coord.alpha,
            state0,
            coord.beta_0,
            coord.beta_1,
            self.spec.t_lock,
            coord.gamma,
            self.spec.fuzz,
        )

    def step(self) -> bool:
        """
        Attempt mcmc step, returns whether the proposal was accepted or rejected
        """

        current = self.samples[-1]
        current_logp = self.logps[-1]

        proposed = self.spec.mcmc_propose(current)
        proposed_logp = self.get_logp(proposed)

        log_acceptance_ratio = proposed_logp - current_logp
        acceptance_ratio = np.exp(log_acceptance_ratio)

        if np.random.random() <= acceptance_ratio:
            self.samples.append(proposed)
            self.logps.append(proposed_logp)
            return True
        else:
            self.samples.append(current)
            self.logps.append(current_logp)
            return False


class Predict:
    def __init__(self, spec: Spec):
        self.spec = spec

    def predict(self, coord: Coord) -> ndarray:
        state0 = np.array([self.spec.population - coord.infected, coord.infected, 0])
        beta = Beta(coord.beta_0, coord.beta_1, self.spec.t_lock)
        sir = SIR(beta, coord.gamma, state0)

        rec = Recoveries(sir)

        recoveries = rec.apply_to_span(365)
        predicted_mean_deaths = recoveries * coord.alpha
        return predicted_mean_deaths


class Test(unittest.TestCase):
    def test_get_death_ll(self):
        # Check that the normal approximation is correct

        deaths = np.array([100])
        recoveries = np.array([1000])
        alpha = 0.1

        log_ll = get_death_ll(deaths, alpha, recoveries, fuzz=0.0)

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

        recoveries = rec.apply_to_span(200)
        deaths = alpha * recoveries

        m = DeathLLGivenModel(deaths)

        fuzz = 1.0

        lls = []
        for test_b in [b, 0.05, 0.3]:
            lls.append(m(alpha, state0, test_b, test_b, 0, gamma, fuzz))

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
            fuzz=1.0,
        )

        coord = spec.sample()
        print(coord, spec.mcmc_propose(coord))

    def test_MCMCSampler(self):
        spec = Spec(
            population=6.55 * 1000000,
            deaths=np.array([0, 0, 0, 0, 1]),
            b_infected=(1.0, 100000.0),
            b_alpha=(0.0001, 0.01),
            b_beta_0=(0.05, 0.4),
            b_beta_1=(0.05, 0.4),
            t_lock=4.0,
            b_gamma=(1 / 40.0, 1 / 10.0),
            fuzz=1.0,
        )

        sampler = MCMCSampler(spec)

        for _ in range(10):
            sampler.step()

        print(sampler.logps)

    def test_Predict(self):
        spec = Spec(
            population=6.55 * 1000000,
            deaths=np.array([0, 0, 0, 0, 1]),
            b_infected=(1.0, 100000.0),
            b_alpha=(0.0001, 0.01),
            b_beta_0=(0.05, 0.4),
            b_beta_1=(0.05, 0.4),
            t_lock=4.0,
            b_gamma=(1 / 40.0, 1 / 10.0),
            fuzz=1.0,
        )

        predict = Predict(spec)
        coord = spec.sample()
        print(predict.predict(coord))
