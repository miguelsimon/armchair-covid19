import unittest

import numpy as np
from numpy import ndarray


def step(beta: float, gamma: float, state: ndarray, dt: float):
    """

    Parameters
    ----------

    beta : float
    gamma : float
    state : ndarray
        3-vector of SIR: susceptible, infected, recovered

    Returns
    -------

    ndarray
        3 - dimensional array of S, I, R

    Taken from https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model_without_vital_dynamics

    """

    S, I, R = state
    N = np.sum(state)

    dS = dt * (-beta * I * S / N)
    dI = dt * (beta * I * S / N - gamma * I)
    dR = dt * (gamma * I)

    delta = np.array([dS, dI, dR])

    return state + delta


class Beta:
    """
    Callable that gives beta value for a given time t.

    Parameters
    ----------

    beta_0 : float
        pre-lockdown beta
    beta_1 : float
        post-lockdown beta
    t_lock : float
        time at which lockdown occurred

    """

    def __init__(self, beta_0: float, beta_1: float, t_lock: float):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.t_lock = t_lock

    def __call__(self, t: float) -> float:
        if t < self.t_lock:
            return self.beta_0
        else:
            return self.beta_1


class SIR:
    """
    SIR model to yield susceptible, infected, recovered
    populations as a function of time.

    Parameters
    ----------

    beta : Beta
        Callable returning SIR model beta for time t
    gamma : float
        SIR model gamma
    state_0 : ndarray
        3, - shaped array of initial S, I, R populations

    Caches a year's worth of simulation and interpolates when called.
    """

    def __init__(self, beta: Beta, gamma: float, state_0: ndarray):
        assert len(state_0) == 3

        self.beta = beta
        self.gamma = gamma
        self.state_0 = state_0

        cur = state_0.copy()

        res = [state_0]
        ts = [0.0]
        dt = 1 / 24.0

        while ts[-1] < 365:
            cur = step(self.beta(ts[-1]), self.gamma, cur, dt)
            res.append(cur.copy())
            ts.append(ts[-1] + dt)

        self.s, self.i, self.r = np.array(res).transpose()

        self.ts = np.array(ts)

    def get_s(self, ts: ndarray) -> ndarray:
        return np.interp(ts, self.ts, self.s)

    def get_i(self, ts: ndarray) -> ndarray:
        return np.interp(ts, self.ts, self.i)

    def get_r(self, ts: ndarray) -> ndarray:
        return np.interp(ts, self.ts, self.r)


class Recoveries:
    """
    Calculate recoveries at day i

    Parameters
    ----------

    sim : SIR
        SIR model
    """

    def __init__(self, sim: SIR):
        self.sim = sim

    def __call__(self, day: int) -> float:
        prev, cur = self.sim.get_r([day - 1, day])
        return cur - prev


class Test(unittest.TestCase):
    def test_beta(self):
        b = Beta(1, 0, 10)
        self.assertEqual(b(3), 1)
        self.assertEqual(b(15), 0)

    def test_SIR(self):
        R0 = 2.5
        gamma = 1 / 20.0
        b = R0 * gamma
        state0 = np.array([6.55 * 1000000, 1000, 0])

        beta = Beta(b, b, 1)
        sir = SIR(beta, gamma, state0)
        ts = np.array([0, 1, 2])
        susceptible = sir.get_s(ts)
        self.assertEqual(len(susceptible), len(ts))

    def test_recoveries(self):
        R0 = 2.5
        gamma = 1 / 20.0
        b = R0 * gamma
        state0 = np.array([6.55 * 1000000, 1000, 0])

        beta = Beta(b, b, 1)
        sir = SIR(beta, gamma, state0)
        rec = Recoveries(sir)
        print(rec(0))
        print(rec(70))
