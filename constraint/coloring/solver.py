#!/usr/bin/python
# -*- coding: utf-8 -*-
import typing
import math
from search import BacktrackSearch
from csp_base import CSP
from csp_base import Variable
from csp_base import DifferentConstraint
import itertools


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

        return (vs[chosen_i], chosen_i) if chosen_i is not None else (None, None)

    def select_next_val(csp: CSP, var_i):
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

    best_colours = max_colours
    # try to decrease the number of colours
    for num_colours in range(max_colours - 1, 0, -1):
        # create csp problem
        vars = []
        for _ in nodes:
            vars.append(Variable(range(num_colours)))

        csp = CSP(vars)
        for n in nodes:
            for neighbour in n.neighbours:
                # avoid adding duplicate constraints by having all lower id != higher id
                if n.id < neighbour.id:
                    csp.add_constraint(DifferentConstraint(vars[n.id], vars[neighbour.id]))

        # try to solve (assume last solution was the best if this time we can't solve for it)
        bt = BacktrackSearch(csp, select_next_variable=select_next_var, select_next_value=select_next_val)
        solution = bt.search()

        if solution:
            best_colours = num_colours
            assignments = solution
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

    if node_count <= 9:
        method = brute_force
    else:
        method = csp

    out = format_output(*method(nodes))
    return out


def format_output(num_colors, proven_optimal, colour_assignments):
    output_data = str(num_colors) + ' ' + str(int(proven_optimal)) + '\n'
    output_data += ' '.join(map(str, list(colour_assignments)))

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
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')
