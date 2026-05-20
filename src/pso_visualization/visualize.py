import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

from .optimizer import ParticleSwarmOptimizer
from .functions import rastrigin


def main():
    bounds = (-5.12, 5.12)
    n_iterations = 100
    pso = ParticleSwarmOptimizer(rastrigin, bounds, n_particles=40, w=0.6, c1=1.8, c2=1.8)
    pso.run(n_iterations)

    fig, ax = plt.subplots(figsize=(8, 8))

    x = np.linspace(*bounds, 300)
    y = np.linspace(*bounds, 300)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(X, Y)

    ax.contourf(X, Y, Z, levels=50, cmap="viridis", norm=LogNorm())
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)
    ax.set_aspect("equal")
    ax.set_title("Particle Swarm Optimization — Rastrigin Function", fontsize=13)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    particles = ax.scatter([], [], c="red", s=25, edgecolors="white", linewidths=0.5, zorder=5)
    best_marker = ax.scatter([], [], c="yellow", s=120, marker="*", edgecolors="black", linewidths=0.8, zorder=6)
    iteration_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                             verticalalignment="top", color="white",
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6))

    def init():
        particles.set_offsets(np.empty((0, 2)))
        best_marker.set_offsets(np.empty((0, 2)))
        iteration_text.set_text("")
        return particles, best_marker, iteration_text

    def update(frame):
        pos = pso.history[frame]
        particles.set_offsets(pos)

        scores = np.array([rastrigin(x, y) for x, y in pos])
        best_idx = np.argmin(scores)
        best_pos = pos[best_idx]
        best_marker.set_offsets([best_pos])

        iteration_text.set_text(f"Iteration: {frame}/{n_iterations}\nBest: {scores[best_idx]:.4f}")
        return particles, best_marker, iteration_text

    anim = FuncAnimation(fig, update, frames=len(pso.history), init_func=init,
                         interval=80, blit=True, repeat=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
