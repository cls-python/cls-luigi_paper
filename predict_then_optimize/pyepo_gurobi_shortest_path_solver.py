import pyepo
import numpy as np
from tqdm import tqdm


class PyEPOGurobiShortestPathSolver:
    def __init__(self, grid_size):
        self.optimizer = pyepo.model.grb.shortestPathModel(grid_size)

    def _get_solutions_and_objective_values(self, costs):

        sols = []
        objs = []
        pbar = tqdm(costs)
        print("Getting solutions and objective values...")
        for c in pbar:
            self.optimizer.setObj(c)
            sol, obj = self.optimizer.solve()
            sols.append(sol)
            objs.append([obj])

        return np.array(sols), np.array(objs)



