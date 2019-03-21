#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
import itertools

Point = namedtuple("Point", ['x', 'y'])


def length(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def tour_length(points, order):
    obj = length(points[order[-1]], points[order[0]])
    for index in range(len(order) - 1):
        obj += length(points[order[index]], points[order[index + 1]])
    return obj


def brute_force(points):
    # enumerate all possible rearrangements (scales combinatorially)
    # can use as ground truth for smaller problems
    solution = None
    lowest_cost = float('inf')
    # can also disallow rotational symmetry by fixing 0 as the first node (but only saves 1 element)
    indices = range(len(points))
    for candidate in itertools.permutations(indices):
        cost = tour_length(points, candidate)
        if cost < lowest_cost:
            lowest_cost = cost
            solution = candidate

    return solution, True


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

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution, optimality = brute_force(points)
    # solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = tour_length(points, solution)

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(int(optimality)) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')