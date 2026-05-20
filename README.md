# PSO Visualization

Animated visualization of the Particle Swarm Optimization (PSO) algorithm running on 2D benchmark functions.

Particles swarm across a contour plot, converging toward the global minimum over successive iterations.

## Benchmark Functions

| Function  | Search Domain       | Global Minimum |
|-----------|---------------------|----------------|
| Rastrigin | [-5.12, 5.12]       | f(0, 0) = 0   |
| Sphere    | [-5, 5]             | f(0, 0) = 0   |
| Ackley    | [-5, 5]             | f(0, 0) = 0   |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Usage

Run the animated visualization:

```bash
python -m pso_visualization.visualize
```

Or use the console entry point after installing the package:

```bash
pso-viz
```

Use the optimizer in your own code:

```python
from pso_visualization import ParticleSwarmOptimizer, rastrigin

pso = ParticleSwarmOptimizer(rastrigin, bounds=(-5.12, 5.12), n_particles=40)
best_position, best_score = pso.run(n_iterations=100)
print(f"Best: {best_score:.6f} at {best_position}")
```

## Tests

```bash
pytest
```

## Project Structure

```
pso_vizualization/
├── src/
│   └── pso_visualization/
│       ├── __init__.py
│       ├── functions.py      # Benchmark objective functions
│       ├── optimizer.py       # PSO algorithm
│       └── visualize.py       # Matplotlib animation
├── tests/
│   ├── test_functions.py
│   └── test_optimizer.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
