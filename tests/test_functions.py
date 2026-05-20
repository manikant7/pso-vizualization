import numpy as np
import pytest
from pso_visualization.functions import rastrigin, sphere, ackley


class TestRastrigin:
    def test_global_minimum_at_origin(self):
        assert rastrigin(0, 0) == pytest.approx(0.0)

    def test_symmetric(self):
        assert rastrigin(1, 2) == pytest.approx(rastrigin(-1, -2))
        assert rastrigin(0.5, -0.5) == pytest.approx(rastrigin(-0.5, 0.5))

    def test_always_non_negative(self):
        rng = np.random.default_rng(42)
        xs = rng.uniform(-5.12, 5.12, 100)
        ys = rng.uniform(-5.12, 5.12, 100)
        for x, y in zip(xs, ys):
            assert rastrigin(x, y) >= 0

    def test_vectorized(self):
        x = np.array([0, 1, -1])
        y = np.array([0, -1, 1])
        result = rastrigin(x, y)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)


class TestSphere:
    def test_global_minimum_at_origin(self):
        assert sphere(0, 0) == pytest.approx(0.0)

    def test_known_value(self):
        assert sphere(3, 4) == pytest.approx(25.0)

    def test_vectorized(self):
        x = np.array([1, 2])
        y = np.array([1, 2])
        result = sphere(x, y)
        np.testing.assert_array_almost_equal(result, [2.0, 8.0])


class TestAckley:
    def test_global_minimum_at_origin(self):
        assert ackley(0, 0) == pytest.approx(0.0, abs=1e-10)

    def test_always_non_negative(self):
        rng = np.random.default_rng(42)
        xs = rng.uniform(-5, 5, 100)
        ys = rng.uniform(-5, 5, 100)
        for x, y in zip(xs, ys):
            assert ackley(x, y) >= 0
