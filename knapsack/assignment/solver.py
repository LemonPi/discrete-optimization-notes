#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import sys
import statistics
import copy
import time

Item = namedtuple("Item", ['index', 'value', 'weight'])


def greedy(items, capacity):
    # order by value density and take in order
    value = 0
    weight = 0
    taken = [0] * len(items)
    items = sorted(items, key=lambda item: item.value / item.weight, reverse=True)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return value, 0, taken


def dp(items, capacity):
    # each col is an item and each row is an increase in capacity
    # print(len(items), capacity)
    table = [[0] * (len(items) + 1) for i in range(capacity + 1)]
    # table[k][j] gives optimal value for knapsack with capacity k and items up to j=item.index+1
    for item in items:
        for k in range(capacity + 1):
            j = item.index + 1
            if item.weight <= k:
                # max of not selecting item j (O(k,j-1)) and selecting it (O(k-w_j,j-1) + v_j)
                table[k][j] = max(table[k][j - 1], table[k - item.weight][j - 1] + item.value)
            else:
                # can't select it
                table[k][j] = table[k][j - 1]

    # for line in table:
    #     print(line)
    # need to backtrace for whether items are taken or not
    value = table[capacity][len(items)]
    taken = [0] * len(items)

    row = capacity
    for j in reversed(range(1, len(table[0]))):
        if table[row][j] == table[row][j - 1]:
            # not selected
            pass
        else:
            index = j - 1
            taken[index] = 1
            row -= items[index].weight

    return value, 1, taken


def bnb(items, capacity):
    # branch and bound solution
    # we sort items first so that we essentially branch based on heuristic
    items = sorted(items, key=lambda item: item.value / item.weight, reverse=True)
    num_items = len(items)

    def heuristic(i, k, v):
        # admissible/optimistic evaluation of max score starting with item i, remaining weight k, and value v
        # note if heuristic not admissible then can't guarantee optimality
        while i < num_items and items[i].weight <= k:
            k -= items[i].weight
            v += items[i].value
            i += 1
        if i < num_items:
            # add remainder (linear relaxation)
            v += items[i].value * k / items[i].weight

        return v

    # OK to assume lowest value is 0
    best_value = 0
    taken = [0] * num_items
    best_taken = [0] * num_items

    def recurse(i, k, v):
        nonlocal best_value
        nonlocal taken
        nonlocal best_taken

        if v > best_value:
            best_value = v
            best_taken = copy.deepcopy(taken)

        # leaf node no more selections
        if i == num_items - 1:
            return

        # go left (select item)
        if items[i].weight <= k:
            taken[items[i].index] = 1
            recurse(i + 1, k - items[i].weight, v + items[i].value)
            taken[items[i].index] = 0

        # check if we can prune tree
        if heuristic(i + 1, k, v) > best_value:
            # only check if it's possible to beat found solution
            recurse(i + 1, k, v)

    recurse(0, capacity, 0)
    return best_value, 1, best_taken


def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i - 1, int(parts[0]), int(parts[1])))

    # problems seem to go up to 10k
    sys.setrecursionlimit(20000)
    # choose optimization method

    # determine how useful the heuristic will be
    value_densities = [item.value / item.weight for item in items]
    heuristic_usefulness = statistics.stdev(value_densities)

    # print(heuristic_usefulness)
    # DP is fastest for small inputs O(Kn)
    method = dp
    if item_count * capacity > 3e7:
        method = bnb
        # B&B only fast if heuristic is useful
        # threshold was experimentally determined
        if heuristic_usefulness < 0.03:
            method = greedy

    # value, proven_optimal, taken = method(items, capacity)
    # print("dp")
    # print(format_output(*dp(items, capacity)))
    # print("greedy")
    # print(format_output(*greedy(items, capacity)))
    # print("\nbnb")

    # start = time.time()
    out = format_output(*method(items, capacity))
    # print("took {}s".format(time.time() - start))
    # prepare the solution in the specified output format
    return out


def format_output(value, proven_optimal, taken):
    output_data = str(value) + ' ' + str(proven_optimal) + '\n'
    output_data += ' '.join(map(str, taken))
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
            'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')
