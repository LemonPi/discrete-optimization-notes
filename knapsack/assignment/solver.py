#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple

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
    table = [[0] * (len(items) + 1) for i in range(capacity+1)]
    # table[k][j] gives optimal value for knapsack with capacity k and items up to j=item.index+1
    for item in items:
        for k in range(capacity+1):
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

    # choose optimization method
    method = dp
    # default to greedy if problem size is too large
    if capacity * item_count > 3e7:
        method = greedy

    value, provenOptimal, taken = method(items, capacity)

    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(provenOptimal) + '\n'
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
