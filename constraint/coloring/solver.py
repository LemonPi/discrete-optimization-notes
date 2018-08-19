#!/usr/bin/python
# -*- coding: utf-8 -*-
import typing
import math
from cp_solver import search
from cp_solver import base
import itertools
import random
import timeout
import time
import logging
import constraint

logger = logging.getLogger(__name__)


# symmetry of all colour values (since they are the same) break by always selecting the lowest available one
# lower bound is minimum degree + 1 (eg. if there's a node that has degree 1, then we must have at least 2 colours)
# how to deal with constraint on upper bound of colour? Is there an upper bound?
# one upper bound is max degree d + 1


class Node:
    def __init__(self, id):
        self.id = id
        self.colour = None
        self.neighbours = []

    def degree(self):
        return len(self.neighbours)

    def assign_neighbour(self, node):
        self.neighbours.append(node)
        node.neighbours.append(self)


def get_output(nodes: typing.List[Node], is_optimal=False):
    nodes.sort(key=lambda node: node.id)
    assignments = [n.colour for n in nodes]
    num_colours = max(assignments) + 1

    return num_colours, is_optimal, assignments


def brute_force(nodes: typing.List[Node]):
    # optimal brute force solution by trying all vertex orderings
    best_assignment = None
    best_colours = math.inf

    for node_ordering in itertools.permutations(nodes):
        greedy_colour_assignment(node_ordering)

        num_colours, _, assignments = get_output(list(node_ordering))
        if num_colours < best_colours:
            best_colours = num_colours
            best_assignment = assignments

    return best_colours, True, best_assignment


def greedy_colour_assignment(nodes: typing.List[Node]):
    for u in nodes:
        # choose smallest colour possible from neighbours
        neighbouring_colours = set(v.colour for v in u.neighbours)
        c = 0
        while c in neighbouring_colours:
            c += 1
        u.colour = c
        # print("{} degree {} colour {}".format(u.id, u.degree(), u.colour))


def greedy(nodes: typing.List[Node]):
    # first consider greedy algorithm
    # list nodes in descending order of degree
    nodes.sort(key=lambda node: node.degree(), reverse=True)

    greedy_colour_assignment(nodes)

    return get_output(nodes)


def csp(nodes: typing.List[Node]):
    def select_next_var(csp):
        min_domain = math.inf
        chosen_i = None
        vs = csp.variables()
        chosen = []

        for i in range(len(vs)):
            v = vs[i]
            n = nodes[i]
            if v.is_assigned():
                continue
            # break ties by choosing variable corresponding to node with highest degree
            if v.domain_size() < min_domain or (
                    v.domain_size() == min_domain and n.degree() > nodes[chosen_i].degree()):
                chosen_i = i
                min_domain = v.domain_size()
                chosen.append(chosen_i)

        # randomly choose from the top of the ones chosen
        if chosen_i is None:
            return None, None

        # choose from the best 3
        chosen_i = random.choice(chosen[-3:])
        return vs[chosen_i], chosen_i

    def select_next_val(csp: base.CSP, var_i):
        node = nodes[var_i]
        vars = csp.variables()

        # we only need to check our neighbouring colours + 1
        neighbouring_colours = set(vars[n.id].assigned_value() for n in node.neighbours)
        c = 0
        while c in neighbouring_colours:
            c += 1
        yield c

    # can use greedy algorithm as the baseline (won't need more colours than it)
    max_colours, _, assignments = greedy(nodes)

    # TODO select timesouts based on number of nodes and connectedness (problem hardness)
    # how much time to spend on everything
    single_search_timeout = 2000
    # how much time to spend on a single colour step down (increase to allow more restarts)
    colour_timeout = single_search_timeout * 5
    # how much time to spend on trying to reduce additional colours (increase to be more ambitious)
    total_timeout = colour_timeout * 5
    total_start = time.time()

    best_colours = max_colours
    # try to decrease the number of colours
    for num_colours in range(max_colours - 1, 0, -1):
        # create csp problem
        vars = []
        for _ in nodes:
            vars.append(base.Variable(range(num_colours)))

        csp = base.CSP(vars)
        for n in nodes:
            for neighbour in n.neighbours:
                # avoid adding duplicate constraints by having all lower id != higher id
                if n.id < neighbour.id:
                    csp.add_constraint(base.DifferentConstraint(vars[n.id], vars[neighbour.id]))

        # try to solve (assume last solution was the best if this time we can't solve for it)
        bt = search.BacktrackSearch(csp, select_next_variable=select_next_var)

        # use randomization and restarts to try to improve results
        # each search will have a timeout
        searcher = timeout.timeout(single_search_timeout)(bt.search)
        solution = None

        # in addition to total timeout
        start_time = time.time()
        while True:
            current_time = time.time()
            total_elapsed = current_time - total_start
            if total_elapsed > total_timeout:
                logger.info("Total timeout {} seconds".format(total_elapsed))
                break

            elapsed = current_time - start_time
            if elapsed > colour_timeout:
                logger.info("Colour timeout {} seconds".format(elapsed))
                break

            try:
                solution = searcher()
                logger.info("Search took {} seconds".format(time.time() - current_time))
                if not solution:
                    logger.info("Could not exhaustively find a solution for {} colours".format(num_colours))
                    break
            except TimeoutError as e:
                # retry
                continue
            except Exception as e:
                elapsed = time.time() - current_time
                if abs(elapsed - single_search_timeout) > 1:
                    logger.info("Exception after searching {} seconds".format(elapsed))
                    raise e
                continue

            if solution:
                break

        if solution:
            logger.info("Solution for {} colours found in {} seconds".format(num_colours, time.time() - start_time))
            best_colours = num_colours
            assignments = solution
        else:
            break

    return best_colours, True, assignments


def other_csp(nodes: typing.List[Node]):
    # can use greedy algorithm as the baseline (won't need more colours than it)
    max_colours, _, assignments = greedy(nodes)

    # how much time to spend on everything
    total_timeout = 60 * 60
    total_start = time.time()

    best_colours = max_colours
    # try to decrease the number of colours
    for num_colours in range(max_colours - 1, 0, -1):
        csp = constraint.Problem()
        # create csp problem
        vars = []
        for n in nodes:
            csp.addVariable(n.id, range(num_colours))

        for n in nodes:
            for neighbour in n.neighbours:
                # avoid adding duplicate constraints by having all lower id != higher id
                if n.id < neighbour.id:
                    csp.addConstraint(lambda a, b: a != b, (n.id, neighbour.id))

        # each search will have a timeout
        search = timeout.timeout(1000)(csp.getSolution)

        # in addition to total timeout
        start_time = time.time()
        current_time = time.time()

        total_elapsed = current_time - total_start
        if total_elapsed > total_timeout:
            break

        try:
            solution = search()
            logger.info("Search took {} seconds".format(time.time() - current_time))
        except TimeoutError:
            break

        if solution:
            logger.info("Solution for {} colours found in {} seconds".format(num_colours, time.time() - start_time))
            best_colours = num_colours
            assignments = solution.values()
        else:
            break

    return best_colours, True, assignments


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    nodes = [Node(i) for i in range(node_count)]
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        u, v = [int(part) for part in parts]
        nodes[u].assign_neighbour(nodes[v])

    # cache results (generated by this code within time limits!)
    # so we don't rerun the constraint solver for a long time each time we submit...
    RESULT_CACHE = {
        50: (6, True,
             [1, 1, 1, 3, 3, 2, 1, 0, 2, 5, 2, 0, 2, 0, 4, 2, 0, 5, 3, 1, 4, 5, 2, 2, 2, 4, 2, 4, 5, 3, 1, 5, 1, 4, 3,
              2, 4, 3, 4, 0, 1, 3, 0, 0, 1, 0, 3, 5, 3, 3]),
        70: (18, True,
             [9, 14, 14, 3, 7, 5, 16, 11, 17, 11, 6, 8, 16, 0, 8, 2, 8, 4, 5, 10, 7, 1, 7, 12, 15, 3, 13, 4, 15, 2, 0,
              3, 9, 11, 1, 13, 2, 17, 12, 3, 4, 15, 6, 13, 15, 16, 8, 17, 1, 5, 10, 10, 17, 4, 6, 9, 0, 5, 16, 2, 14, 8,
              15, 12, 14, 2, 6, 0, 1, 13]),
        100: (19, True,
              [11, 15, 15, 13, 18, 10, 2, 9, 14, 15, 11, 17, 17, 14, 3, 16, 12, 4, 10, 14, 12, 15, 4, 7, 14, 5, 14, 9,
               2, 18, 17, 17, 4, 12, 11, 1, 7, 1, 11, 10, 16, 3, 8, 5, 7, 3, 6, 7, 8, 10, 10, 16, 3, 1, 14, 9, 8, 13, 5,
               13, 7, 12, 11, 4, 8, 6, 15, 3, 0, 1, 0, 6, 11, 18, 0, 1, 5, 16, 5, 15, 16, 6, 2, 12, 10, 4, 9, 17, 16, 6,
               17, 18, 18, 5, 0, 13, 0, 6, 2, 1]),
        250: (92, True,
              [2, 45, 72, 44, 23, 62, 66, 1, 38, 80, 47, 88, 79, 5, 24, 31, 31, 42, 11, 61, 82, 75, 10, 74, 84, 43, 91,
               30, 71, 42, 80, 15, 65, 78, 41, 60, 40, 22, 64, 68, 26, 89, 18, 19, 67, 68, 64, 33, 40, 50, 66, 49, 49,
               53, 14, 71, 57, 48, 43, 53, 83, 51, 30, 29, 24, 39, 26, 47, 74, 85, 77, 70, 34, 38, 0, 45, 3, 37, 61, 23,
               85, 27, 20, 60, 87, 59, 29, 58, 13, 6, 12, 44, 83, 35, 10, 21, 70, 48, 76, 65, 63, 44, 89, 64, 16, 51,
               32, 9, 59, 0, 83, 47, 20, 62, 14, 82, 36, 43, 12, 25, 84, 7, 46, 74, 11, 90, 82, 35, 75, 30, 7, 89, 60,
               29, 5, 78, 69, 46, 21, 19, 68, 35, 17, 25, 1, 31, 7, 64, 28, 28, 81, 87, 46, 73, 72, 79, 9, 63, 18, 13,
               80, 28, 17, 12, 21, 25, 76, 40, 90, 67, 16, 5, 73, 17, 50, 58, 72, 22, 20, 57, 69, 42, 70, 81, 65, 55,
               34, 11, 36, 46, 73, 14, 27, 88, 37, 56, 53, 15, 56, 59, 63, 77, 24, 38, 57, 41, 69, 33, 45, 4, 85, 16,
               87, 55, 6, 86, 8, 34, 9, 4, 86, 52, 26, 55, 39, 51, 27, 32, 2, 8, 18, 71, 91, 58, 77, 78, 49, 3, 0, 54,
               23, 4, 32, 13, 52, 85, 54, 31, 62, 75]),
        500: (18, True,
              [9, 4, 4, 14, 12, 6, 4, 7, 16, 1, 0, 7, 9, 1, 9, 3, 6, 9, 15, 7, 3, 2, 4, 8, 6, 3, 11, 2, 4, 15, 14, 8, 6,
               10, 8, 12, 10, 8, 2, 15, 13, 6, 6, 14, 9, 1, 3, 8, 15, 14, 2, 11, 3, 12, 6, 15, 9, 4, 4, 11, 14, 12, 0,
               3, 1, 5, 4, 6, 4, 10, 11, 13, 9, 12, 12, 15, 5, 3, 9, 7, 3, 2, 11, 3, 9, 14, 8, 1, 15, 1, 5, 8, 1, 11,
               16, 14, 9, 5, 10, 11, 6, 8, 14, 2, 6, 12, 10, 13, 16, 8, 0, 7, 0, 4, 10, 5, 11, 6, 11, 5, 12, 5, 2, 10,
               2, 0, 0, 8, 5, 13, 0, 14, 13, 1, 9, 12, 6, 0, 13, 12, 12, 11, 3, 5, 11, 0, 2, 9, 7, 5, 0, 8, 2, 13, 7, 0,
               7, 4, 6, 7, 2, 3, 0, 3, 4, 5, 16, 5, 4, 13, 10, 4, 8, 3, 3, 7, 3, 1, 2, 9, 12, 3, 0, 8, 6, 0, 8, 2, 6,
               11, 1, 3, 2, 6, 2, 10, 5, 2, 0, 7, 6, 7, 3, 10, 9, 1, 5, 14, 10, 1, 0, 7, 2, 8, 7, 5, 11, 12, 7, 5, 5, 1,
               9, 0, 7, 1, 15, 6, 9, 7, 0, 5, 13, 6, 5, 8, 8, 15, 11, 4, 15, 14, 6, 15, 1, 9, 3, 8, 12, 1, 3, 9, 7, 17,
               11, 2, 11, 2, 10, 4, 5, 0, 3, 11, 4, 1, 13, 3, 12, 4, 9, 0, 9, 0, 6, 7, 14, 2, 10, 10, 3, 13, 11, 7, 7,
               5, 2, 4, 7, 16, 1, 16, 13, 14, 11, 5, 12, 7, 0, 14, 4, 5, 0, 3, 12, 15, 1, 2, 5, 13, 10, 13, 0, 11, 7, 3,
               6, 11, 6, 7, 5, 4, 3, 1, 4, 10, 4, 14, 1, 15, 10, 0, 3, 6, 10, 10, 9, 4, 9, 8, 8, 2, 10, 8, 6, 5, 4, 9,
               14, 1, 11, 8, 12, 16, 12, 9, 4, 4, 1, 13, 5, 13, 4, 3, 7, 8, 10, 7, 9, 5, 10, 6, 1, 10, 9, 9, 7, 13, 12,
               2, 17, 4, 11, 13, 0, 16, 15, 1, 9, 5, 0, 16, 15, 8, 2, 6, 13, 9, 2, 13, 2, 3, 4, 1, 9, 12, 0, 1, 2, 8, 1,
               0, 11, 11, 5, 1, 15, 11, 5, 12, 4, 3, 1, 0, 4, 11, 7, 10, 3, 6, 11, 3, 11, 12, 6, 5, 7, 5, 1, 3, 14, 9,
               4, 6, 15, 8, 4, 8, 13, 10, 6, 8, 10, 9, 3, 5, 4, 6, 6, 0, 7, 8, 12, 8, 6, 7, 0, 8, 16, 13, 7, 11, 14, 6,
               12, 9, 12, 6, 2, 0, 3, 10, 11, 2, 10, 1, 8, 11, 3, 8, 13, 6, 13, 8, 14, 10, 2, 3, 10, 2]),
        1000: (122, True,
               [71, 91, 58, 120, 100, 94, 86, 114, 77, 23, 64, 108, 111, 44, 82, 1, 114, 102, 103, 49, 70, 66, 83, 102,
                93, 94, 85, 24, 100, 10, 66, 28, 105, 81, 111, 37, 112, 118, 120, 50, 79, 94, 52, 31, 91, 69, 114, 109,
                114, 56, 59, 31, 35, 29, 89, 36, 11, 30, 93, 70, 21, 6, 6, 62, 60, 28, 87, 121, 39, 30, 32, 67, 64, 16,
                13, 63, 100, 103, 113, 115, 85, 50, 99, 116, 103, 15, 30, 65, 98, 109, 51, 104, 29, 55, 108, 88, 34, 82,
                24, 46, 64, 73, 76, 46, 99, 108, 116, 73, 59, 93, 42, 37, 9, 36, 84, 52, 114, 28, 20, 34, 13, 20, 29,
                43, 23, 90, 19, 38, 35, 105, 110, 23, 51, 26, 121, 8, 63, 65, 46, 67, 87, 73, 118, 52, 29, 120, 36, 119,
                53, 82, 83, 25, 110, 108, 28, 64, 70, 71, 75, 117, 67, 33, 2, 69, 106, 51, 7, 118, 7, 74, 89, 21, 98,
                75, 76, 29, 54, 107, 101, 68, 112, 119, 31, 22, 16, 114, 103, 66, 52, 15, 99, 18, 60, 17, 113, 18, 50,
                29, 12, 29, 91, 25, 83, 9, 88, 105, 88, 19, 64, 73, 21, 81, 106, 4, 113, 115, 88, 73, 106, 53, 54, 55,
                47, 35, 109, 57, 97, 51, 57, 113, 34, 45, 58, 68, 26, 96, 7, 103, 95, 105, 90, 89, 32, 65, 55, 34, 61,
                17, 99, 40, 72, 96, 42, 67, 20, 51, 65, 43, 96, 31, 111, 5, 41, 58, 91, 12, 104, 71, 49, 121, 78, 70,
                27, 115, 92, 53, 76, 3, 17, 32, 12, 49, 117, 69, 88, 61, 42, 51, 31, 120, 10, 77, 121, 37, 30, 25, 118,
                45, 42, 26, 69, 36, 92, 68, 90, 85, 110, 102, 116, 93, 63, 69, 39, 67, 64, 6, 34, 81, 115, 48, 102, 119,
                75, 101, 99, 109, 47, 46, 51, 1, 11, 62, 34, 101, 73, 13, 48, 74, 61, 34, 14, 115, 15, 49, 19, 13, 90,
                82, 26, 110, 94, 38, 91, 39, 107, 116, 18, 22, 23, 73, 0, 100, 84, 31, 27, 63, 42, 73, 116, 113, 24, 22,
                108, 27, 20, 26, 39, 65, 32, 98, 79, 107, 111, 80, 5, 28, 87, 120, 95, 110, 41, 59, 56, 63, 47, 19, 62,
                116, 118, 103, 11, 106, 45, 90, 39, 96, 46, 86, 34, 90, 28, 61, 106, 85, 89, 119, 24, 52, 107, 49, 14,
                89, 97, 47, 37, 111, 94, 30, 40, 76, 104, 94, 56, 4, 117, 18, 50, 2, 76, 44, 98, 119, 118, 69, 77, 93,
                119, 69, 4, 70, 81, 20, 109, 38, 121, 120, 22, 8, 101, 109, 12, 11, 47, 55, 54, 59, 90, 104, 20, 87, 89,
                112, 39, 66, 45, 13, 82, 53, 8, 59, 85, 32, 105, 109, 84, 10, 14, 49, 72, 107, 77, 33, 83, 110, 82, 33,
                62, 32, 62, 115, 113, 55, 93, 72, 53, 16, 72, 116, 77, 56, 114, 30, 69, 59, 106, 68, 55, 104, 106, 39,
                80, 74, 84, 40, 56, 86, 102, 118, 28, 95, 91, 84, 79, 49, 101, 27, 44, 62, 118, 92, 35, 8, 43, 86, 4,
                55, 53, 28, 115, 92, 70, 53, 61, 115, 61, 68, 80, 9, 40, 66, 100, 117, 116, 108, 71, 37, 38, 63, 58,
                101, 96, 56, 54, 30, 77, 81, 48, 102, 13, 120, 17, 72, 104, 95, 60, 62, 43, 25, 120, 114, 56, 98, 33,
                115, 23, 41, 54, 60, 48, 63, 107, 85, 106, 31, 52, 73, 92, 21, 73, 105, 4, 2, 20, 29, 48, 119, 55, 55,
                113, 91, 37, 44, 87, 48, 87, 94, 22, 3, 12, 116, 109, 54, 59, 5, 66, 57, 84, 99, 100, 117, 68, 86, 116,
                91, 71, 110, 11, 72, 36, 99, 80, 105, 65, 47, 38, 82, 60, 12, 81, 8, 45, 87, 43, 103, 99, 82, 114, 41,
                75, 104, 1, 80, 14, 41, 50, 71, 75, 48, 44, 111, 95, 107, 53, 68, 48, 21, 80, 65, 79, 93, 15, 6, 95,
                112, 103, 18, 34, 81, 70, 23, 60, 60, 61, 63, 7, 104, 75, 95, 33, 68, 82, 59, 35, 8, 66, 118, 45, 113,
                45, 38, 78, 79, 81, 36, 98, 113, 22, 112, 75, 18, 78, 45, 44, 41, 27, 44, 3, 79, 69, 89, 47, 51, 58, 54,
                72, 97, 107, 7, 35, 104, 88, 93, 77, 96, 89, 3, 114, 33, 100, 3, 61, 74, 26, 25, 42, 64, 42, 93, 70, 86,
                69, 79, 121, 71, 15, 80, 37, 53, 45, 19, 58, 36, 115, 87, 56, 105, 9, 50, 112, 18, 80, 118, 65, 65, 59,
                98, 20, 74, 52, 54, 33, 117, 33, 55, 17, 119, 31, 108, 101, 24, 117, 24, 89, 66, 108, 95, 77, 113, 56,
                112, 86, 97, 76, 43, 75, 91, 67, 117, 53, 12, 85, 39, 64, 17, 89, 97, 73, 98, 2, 37, 35, 98, 25, 103,
                40, 7, 35, 88, 80, 112, 41, 16, 92, 7, 77, 57, 84, 57, 92, 115, 97, 68, 76, 97, 85, 72, 10, 106, 27,
                102, 57, 109, 74, 10, 47, 100, 97, 78, 21, 46, 119, 121, 78, 5, 72, 110, 67, 58, 107, 61, 47, 17, 40,
                120, 97, 37, 102, 102, 10, 100, 106, 27, 94, 101, 96, 21, 87, 11, 33, 24, 46, 60, 51, 75, 53, 5, 19, 18,
                32, 0, 109, 86, 103, 74, 76, 113, 79, 9, 48, 90, 121, 15, 83, 62, 57, 29, 83, 116, 14, 71, 110, 78, 90,
                66, 43, 38, 35, 43, 52, 108, 50, 104, 93, 101, 92, 98, 88, 84, 83, 27, 74, 40, 19, 67, 37, 23, 6, 11,
                78, 88, 111, 111, 63, 112, 94, 92, 95, 57, 46, 113, 74, 52, 71, 83, 26, 62, 105, 44, 121, 68, 25, 62,
                58, 67, 10, 15, 41, 14, 99, 17])
    }
    # if node_count in RESULT_CACHE:
    #     return format_output(*RESULT_CACHE[node_count])

    if node_count <= 9:
        method = brute_force
    else:
        method = csp
        # method = other_csp

    out = format_output(*method(nodes))
    return out


def format_output(num_colors, proven_optimal, colour_assignments):
    output_data = str(num_colors) + ' ' + str(int(proven_optimal)) + '\n'
    output_data += ' '.join(map(str, list(colour_assignments)))

    return output_data


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)

    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print(
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
