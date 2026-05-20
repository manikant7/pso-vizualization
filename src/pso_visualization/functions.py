import numpy as np


def rastrigin(x):
    x = np.asarray(x)
    n = x.shape[-1] if x.ndim > 0 else 1
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def sphere(x):
    x = np.asarray(x)
    return np.sum(x**2, axis=-1)


def ackley(x):
    x = np.asarray(x)
    n = x.shape[-1] if x.ndim > 0 else 1
    sum_sq = np.sum(x**2, axis=-1)
    sum_cos = np.sum(np.cos(2 * np.pi * x), axis=-1)
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + np.e + 20


def rosenbrock_constrained(x):
    x = np.asarray(x, dtype=float)
    xy = x[..., :2]
    xv = xy[..., 0]
    yv = xy[..., 1]
    f = (1 - xv) ** 2 + 100 * (yv - xv ** 2) ** 2
    violation = xv ** 2 + yv ** 2 - 2
    penalty = np.where(violation > 0, 1e6 * violation ** 2, 0.0)
    return f + penalty
