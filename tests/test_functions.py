import numpy as np
import pytest
from pso_visualization.functions import rastrigin, sphere, ackley


class TestRastrigin:
    def test_global_minimum_at_origin(self):
        assert rastrigin(np.array([0, 0])) == pytest.approx(0.0)

    def test_symmetric(self):
        assert rastrigin(np.array([1, 2])) == pytest.approx(rastrigin(np.array([-1, -2])))
        assert rastrigin(np.array([0.5, -0.5])) == pytest.approx(rastrigin(np.array([-0.5, 0.5])))

    def test_always_non_negative(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(-5.12, 5.12, (100, 2))
        for p in points:
            assert rastrigin(p) >= 0

    def test_vectorized(self):
        points = np.array([[0, 0], [1, -1], [-1, 1]])
        result = rastrigin(points)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)

    def test_high_dimensional(self):
        assert rastrigin(np.zeros(20)) == pytest.approx(0.0)
        assert rastrigin(np.zeros(20)) < rastrigin(np.ones(20))


class TestSphere:
    def test_global_minimum_at_origin(self):
        assert sphere(np.array([0, 0])) == pytest.approx(0.0)

    def test_known_value(self):
        assert sphere(np.array([3, 4])) == pytest.approx(25.0)

    def test_vectorized(self):
        points = np.array([[1, 1], [2, 2]])
        result = sphere(points)
        np.testing.assert_array_almost_equal(result, [2.0, 8.0])

    def test_high_dimensional(self):
        assert sphere(np.zeros(20)) == pytest.approx(0.0)
        assert sphere(np.ones(20)) == pytest.approx(20.0)


class TestAckley:
    def test_global_minimum_at_origin(self):
        assert ackley(np.array([0, 0])) == pytest.approx(0.0, abs=1e-10)

    def test_always_non_negative(self):
        rng = np.random.default_rng(42)
        points = rng.uniform(-5, 5, (100, 2))
        for p in points:
            assert ackley(p) >= 0

    def test_high_dimensional(self):
        assert ackley(np.zeros(20)) == pytest.approx(0.0, abs=1e-10)
