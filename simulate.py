from functools import partial
from multiprocessing import Pool
from collections import Counter
import numpy as np
from tqdm import tqdm


def simulate_brownian_motion_fast(
    particle_populations, dt=0.01, radius=0.01, max_steps=1e4, D=3
):
    N = sum(particle_populations.values())
    particles = np.random.rand(N, D)
    buffer = np.zeros_like(particles)
    sentinel = np.random.rand(D)

    for step in range(int(max_steps)):
        sentinel += np.random.normal(size=D) * dt
        particles += np.random.normal(size=(N, D)) * dt

        np.clip(particles, 0, 1, out=buffer)
        particles, buffer = buffer, particles

        collisions = np.argwhere(
            np.linalg.norm(sentinel - particles, axis=1) < 2 * radius
        )
        if len(collisions):
            i = collisions[0, 0]
            for color, n in particle_populations.items():
                i -= n
                if i <= 0:
                    return color
    return None


def run_experiment(i, experiment):
    return experiment()


def run_many(N, experiment):
    results = []
    try:
        with Pool() as pool:
            for r in tqdm(
                pool.imap_unordered(
                    partial(run_experiment, experiment=experiment), range(N)
                ),
                total=N,
            ):
                results.append(r)
    except KeyboardInterrupt:
        pass
    return results


# Example usage
population = {
    "red": 40,
    "blue": 45,
    "pink": 15,
}
D = 2
N = 100_000

"""
collision_with = simulate_brownian_motion_fast(population)
print(f"One off sentinel collided with particle of color: {collision_with}")
"""

experiment = Counter(
    run_many(
        N,
        partial(
            simulate_brownian_motion_fast,
            population,
            D=D,
        ),
    )
)
N_unended = experiment.pop(None, 0)
N_actual = sum(experiment.values())

print(f"Collision counts after {N_actual} completed runs")
for color, count in experiment.most_common():
    print(f"\t{color} -> {count} collisions ({count/N_actual*100:0.2f}%)")
print(f"Number of experiments that didn't conclude: {N_unended}")
