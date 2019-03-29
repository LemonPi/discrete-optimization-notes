#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import time
from collections import namedtuple
import itertools

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


def simulated_annealing(points):
    solution = range(len(points))
    # TODO implement simulated annealing
    return solution, False


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
    solution, optimality = greedy(points, dists)

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
