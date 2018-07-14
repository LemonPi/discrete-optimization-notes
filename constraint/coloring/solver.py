#!/usr/bin/python
# -*- coding: utf-8 -*-
import typing


# symmetry of all colour values (since they are the same) break by always selecting the lowest available one
# lower bound is minimum degree + 1 (eg. if there's a node that has degree 1, then we must have at least 2 colours)
# how to deal with constraint on upper bound of colour? Is there an upper bound?
# one upper bound is max degree d + 1

class Variable:
    def __init__(self, domain=()):
        self._domain = domain
        self._value = None

    def domain_size(self):
        return len(self._domain) if self._value is None else 1

    def assign_value(self, value):
        self._value = value


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


def greedy(nodes: typing.List[Node]):
    # first consider greedy algorithm
    # list nodes in descending order of degree
    nodes.sort(key=lambda node: node.degree(), reverse=True)

    for u in nodes:
        # choose smallest colour possible from neighbours
        neighbouring_colours = set(v.colour for v in u.neighbours)
        c = 0
        while c in neighbouring_colours:
            c += 1
        u.colour = c

        # print("{} degree {} colour {}".format(u.id, u.degree(), u.colour))

    # retrieve output
    nodes.sort(key=lambda node: node.id)
    assignments = [n.colour for n in nodes]
    num_colours = max(assignments) + 1
    lower_bound = min((n.degree() for n in nodes)) + 1

    return num_colours, num_colours == lower_bound, assignments


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

    method = greedy

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
