import numpy as np

def simple_function(point):
    x = point[0]
    return 2*x**2 + 4*x + 1

def simple_gradient(point):
    x = point[0]
    return np.array([4*x + 4,])

########################################################################################
# Functions to implement the PSO algorithm
# It is designed for academic purpose
# Various details have been omitted and no program optimization has done.
# Refer to https://induraj2020.medium.com/implementing-particle-swarm-optimization-in-python-c59278bc5846
def pso(cost_func, dim=2, num_particles=30, max_iter=100, w=0.5, c1=1, c2=2):
    # Initialize particles and velocities
    particles = np.random.uniform(-5.12, 5.12, (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    # Initialize the best positions and fitness values
    best_positions = np.copy(particles)
    best_fitness = np.array([cost_func(p) for p in particles])
    swarm_best_position = best_positions[np.argmin(best_fitness)]
    swarm_best_fitness = np.min(best_fitness)

    # Iterate through the specified number of iterations, updating the velocity and position of each particle at each iteration
    for i in range(max_iter):
        # Update velocities
        r1 = np.random.uniform(0, 1, (num_particles, dim))
        r2 = np.random.uniform(0, 1, (num_particles, dim))
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (swarm_best_position - particles)

        # Update positions
        particles += velocities

        # Evaluate fitness of each particle
        fitness_values = np.array([cost_func(p) for p in particles])

        # Update best positions and fitness values
        improved_indices = np.where(fitness_values < best_fitness)
        best_positions[improved_indices] = particles[improved_indices]
        best_fitness[improved_indices] = fitness_values[improved_indices]
        if np.min(fitness_values) < swarm_best_fitness:
            swarm_best_position = particles[np.argmin(fitness_values)]
            swarm_best_fitness = np.min(fitness_values)

    # Return the best solution found by the PSO algorithm
    return swarm_best_position, swarm_best_fitness
########################################################################################

if __name__ == '__main__':
    # Define the dimensions of the problem
    dim = 1

    # Run the PSO algorithm on the Rastrigin function
    solution, fitness = pso(simple_function, dim=dim)

    # Print the solution and fitness value
    print('Solution:', solution)
    print('Fitness:', fitness)