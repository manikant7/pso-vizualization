import numpy as np
import pytest
from pso_visualization.optimizer import ParticleSwarmOptimizer
from pso_visualization.functions import sphere, rastrigin


class TestParticleSwarmOptimizer:
    def test_initialization(self):
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=10)
        assert pso.positions.shape == (10, 2)
        assert pso.velocities.shape == (10, 2)
        assert len(pso.personal_best_scores) == 10
        assert len(pso.history) == 1

    def test_initialization_high_dim(self):
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=10, n_dims=20)
        assert pso.positions.shape == (10, 20)
        assert pso.velocities.shape == (10, 20)

    def test_particles_within_bounds(self):
        pso = ParticleSwarmOptimizer(sphere, (-3, 3), n_particles=20)
        assert np.all(pso.positions >= -3)
        assert np.all(pso.positions <= 3)

    def test_step_records_history(self):
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=10)
        pso.step()
        pso.step()
        assert len(pso.history) == 3

    def test_global_best_improves_or_stays(self):
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=20)
        initial_best = pso.global_best_score
        for _ in range(50):
            pso.step()
        assert pso.global_best_score <= initial_best

    def test_run_method(self):
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=20)
        position, score = pso.run(50)
        assert position.shape == (2,)
        assert score <= pso.personal_best_scores.max()
        assert len(pso.history) == 51

    def test_converges_on_sphere(self):
        np.random.seed(42)
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=30, w=0.5, c1=1.5, c2=1.5)
        pso.run(200)
        assert pso.global_best_score < 0.1

    def test_positions_clipped_to_bounds(self):
        pso = ParticleSwarmOptimizer(sphere, (-2, 2), n_particles=10)
        pso.velocities = np.full((10, 2), 100.0)
        pso.step()
        assert np.all(pso.positions >= -2)
        assert np.all(pso.positions <= 2)

    def test_personal_best_updates(self):
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=10)
        initial_personal_bests = pso.personal_best_scores.copy()
        for _ in range(20):
            pso.step()
        assert np.all(pso.personal_best_scores <= initial_personal_bests)

    def test_20d_converges(self):
        np.random.seed(42)
        pso = ParticleSwarmOptimizer(sphere, (-5, 5), n_particles=60, n_dims=20, w=0.5, c1=1.5, c2=1.5)
        pso.run(300)
        assert pso.global_best_score < 1.0
