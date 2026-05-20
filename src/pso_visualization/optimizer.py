import numpy as np


class ParticleSwarmOptimizer:
    def __init__(self, func, bounds, n_particles=30, w=0.5, c1=1.5, c2=1.5):
        self.func = func
        self.bounds = bounds
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

        low, high = bounds
        self.positions = np.random.uniform(low, high, (n_particles, 2))
        self.velocities = np.random.uniform(-1, 1, (n_particles, 2))

        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([func(x, y) for x, y in self.positions])

        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

        self.history = [self.positions.copy()]

    def step(self):
        r1 = np.random.random((self.n_particles, 2))
        r2 = np.random.random((self.n_particles, 2))

        cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
        social = self.c2 * r2 * (self.global_best_position - self.positions)
        self.velocities = self.w * self.velocities + cognitive + social

        self.positions += self.velocities

        low, high = self.bounds
        self.positions = np.clip(self.positions, low, high)

        scores = np.array([self.func(x, y) for x, y in self.positions])

        improved = scores < self.personal_best_scores
        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_scores[improved] = scores[improved]

        best_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[best_idx] < self.global_best_score:
            self.global_best_position = self.personal_best_positions[best_idx].copy()
            self.global_best_score = self.personal_best_scores[best_idx]

        self.history.append(self.positions.copy())

    def run(self, n_iterations):
        for _ in range(n_iterations):
            self.step()
        return self.global_best_position, self.global_best_score
