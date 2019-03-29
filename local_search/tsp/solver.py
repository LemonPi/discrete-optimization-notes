#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import time
from collections import namedtuple
import itertools
import random

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def pairwise_dist(points):
    cache = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = length(points[i], points[j])
            cache[(i, j)] = d
            cache[(j, i)] = d

    return cache


def tour_length(order, dists):
    obj = dists[(order[-1], order[0])]
    for index in range(len(order) - 1):
        obj += dists[(order[index], order[index + 1])]
    return obj


def two_opt(order, i, j):
    """Modify visit order by switching directions from i -> j to i <- j"""
    order[i:j] = order[j - 1:i - 1:-1]


def brute_force(points, dists):
    # enumerate all possible rearrangements (scales combinatorially)
    # can use as ground truth for smaller problems
    solution = None
    lowest_cost = float('inf')
    # can also disallow rotational symmetry by fixing 0 as the first node (but only saves 1 element)
    indices = range(len(points))
    for candidate in itertools.permutations(indices):
        cost = tour_length(candidate, dists)
        if cost < lowest_cost:
            lowest_cost = cost
            solution = candidate

    return solution, True


def greedy(points, dists):
    solution = list(range(len(points)))
    candidate = solution[:]
    lowest_cost = tour_length(solution, dists)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(solution) - 2):
            for j in range(i + 2, len(solution)):
                new_candidate = candidate[:]
                two_opt(new_candidate, i, j)
                cost = tour_length(new_candidate, dists)
                if cost < lowest_cost:
                    lowest_cost = cost
                    improved = True
                    solution = new_candidate
        candidate = solution
    return solution, False


def tabu_search(points):
    solution = range(len(points))
    # TODO implement taboo search
    return solution, False


def random_indices(N):
    """2 random indices between [1,N) that are more than 1 apart"""
    while True:
        a = random.randint(1, N - 1)
        b = random.randint(1, N - 1)
        if math.fabs(a - b) > 1:
            return min(a, b), max(a, b)


class BestSolutionsCache:
    def __init__(self, num):
        # number of solutions to keep
        self.n = num
        self.q = []
        self.keys = set()

    def add(self, cost, sol):
        if len(self.q) < self.n:
            self.q.append((cost, sol[:]))
            self.q.sort(key=lambda x: x[0])
            self.keys.add(cost)
            return True
        elif cost < self.q[-1][0]:
            if cost not in self.keys:
                self.keys.discard(self.q[-1][0])
                self.keys.add(cost)
                self.q[-1] = (cost, sol[:])
                self.q.sort(key=lambda x: x[0])
                return True
        return False


def simulated_annealing(points, dists, start_solution=None, anneal=0.99, timeout=600):
    start = time.perf_counter()

    N = len(points)
    if start_solution:
        solution = start_solution[:]
    else:
        solution = list(range(N))
        # initialize solution with greedy 2-opt
        if N < 120:
            solution, lowest_cost = greedy(points, dists)

    lowest_cost = tour_length(solution, dists)
    print("Starting simulated annealing with {}".format(lowest_cost))

    # parameters
    max_T = lowest_cost / 4
    T = max_T
    M = 5
    ACTUAL_IMPROVEMENT_ITER = int(10000000 / N)
    REXPLORE_WITHOUT_IMPROVEMENT = 100
    prev_sols = BestSolutionsCache(M)
    prev_sols.add(lowest_cost, solution)

    k = 0
    k_since_last_timeout_check = 0
    k_since_last_improvement = 0
    k_since_actual_improvement = 0
    while True:
        if k - k_since_last_timeout_check > 500:
            k_since_last_timeout_check = k
            elapsed = time.perf_counter() - start
            print(elapsed)
            if elapsed > timeout:
                break

        i, j = random_indices(N)

        candidate = solution[:]
        two_opt(candidate, i, j)

        cost = tour_length(candidate, dists)
        if cost < lowest_cost:
            solution = candidate
            lowest_cost = cost
            k_since_last_improvement = k
            if prev_sols.add(cost, solution):
                k_since_actual_improvement = k
            print("Moved to {}".format(lowest_cost))
        else:
            # maybe we're done and can't make any improvements
            if k - k_since_actual_improvement > ACTUAL_IMPROVEMENT_ITER:
                print("Can't make progress so give up")
                break
            # maybe need to do some reheating or re-exploration
            if k - k_since_last_improvement > REXPLORE_WITHOUT_IMPROVEMENT:
                T += max_T / 16
                # go back to a previous solution
                m = random.randint(0, len(prev_sols.q) - 1)
                lowest_cost, solution = prev_sols.q[m]
                k_since_last_improvement = k
                print("Going back to {}".format(lowest_cost))
            else:
                # take it with some probability
                d = cost - lowest_cost
                p = math.exp(-d / T)
                if random.random() < p:
                    solution = candidate
                    lowest_cost = cost
                    prev_sols.add(cost, solution)
                    print("Moved to {} with p {}".format(lowest_cost, p))

        T *= anneal
        k += 1

    return prev_sols.q[0][1], False


def lower_bound(points):
    # TODO create a path length lower bound for use as heuristic for example in branch and bound
    # one way to do this is to create a minimum spanning tree then double all edges
    return 0


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # cache pairwise distances
    dists = pairwise_dist(points)

    # build a trivial solution
    start = time.perf_counter()
    # visit the nodes in the order they appear in the file
    # solution, optimality = brute_force(points, dists)
    # solution = range(0, nodeCount)
    # solution, optimality = greedy(points, dists)
    solution, optimality = simulated_annealing(points, dists)

    end = time.perf_counter()
    print("Took {} seconds".format(end - start))

    # calculate the length of the tour
    obj = tour_length(solution, dists)

    # prepare the solution in the specified output format
    output_data = '%.4f' % obj + ' ' + str(int(optimality)) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        output = solve_it(input_data)
        print(output)
        with open('sol', 'w') as out:
            print(output, file=out)
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')
