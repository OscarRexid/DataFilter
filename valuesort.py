import numpy as np

class Particle():
    def __init__(self, position, cost):
        self.position = position
        self.cost = cost
        self.best_position = self.position
        self.best_cost = self.cost

class PSO():
    def __init__(self, particle_matrix, cost_list):
        self.particles = [Particle(particle_matrix[i], cost_list[i]) for i in range(len(cost_list))]

    def optimize(self):
        global_best_cost = min([particle.best_cost for particle in self.particles])
        global_best_position = self.particles[np.argmin([particle.best_cost for particle in self.particles])].position

        return global_best_position, global_best_cost

# Assume we have the following sensor data and cost values
particle_matrix = np.array([[1,2,3],[4,5,6],[7,8,9],[17,834,329],[37,3284,429],[57,428,9],[527,8,9]])  # 100 particles in 2D
print(particle_matrix)

cost_list = np.array([3,1,3,42,5,2,1])  # Cost for each particle
print(cost_list)

pso = PSO(particle_matrix, cost_list)
best_position, best_cost = pso.optimize()
print(f'Best position: {best_position}, Best cost: {best_cost}')

