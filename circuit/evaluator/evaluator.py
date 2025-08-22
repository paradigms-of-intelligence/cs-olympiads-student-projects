#!/usr/bin/env python3
"""
⚠️ WARNING: This code was automatically translated from C++ to Python by an AI tool.
It may contain inaccuracies, inefficient constructs, or logic errors introduced 
during translation. Please review carefully and test thoroughly before using 
in production.
"""

import sys
import struct
from config import *

# helper to read signed 32-bit little-endian integer (matches typical C++ behaviour on little-endian machines)
def read_int32_t(f):
    data = f.read(4)
    return struct.unpack('<i', data)[0]

class AndNot_network:
    def __init__(self):
        self.N = 0
        self.O = 0
        self.value = []          # list of bools, 1..N used
        self.C_1 = []            # list of ints, index 0..N
        self.C_2 = []
        self.result_nodes = []   # ordered ids of final network outputs
        self.toposorted = []     # list sized N+1; toposorted[i] = node id to compute at step i

    def init(self):
        with open(TRAINED_NETWORK_2GATES_FILE, 'rb') as graphinput:
            self.N = read_int32_t(graphinput)
            # allocate arrays sized N+1, indexed 0..N
            self.value = [False] * (self.N + 1)
            self.C_1 = [0] * (self.N + 1)
            self.C_2 = [0] * (self.N + 1)
            self.toposorted = [0] * (self.N + 1)

            # read nodes for indices INPUT_NODES+1 .. N inclusive
            for i in range(INPUT_NODES + 1, self.N + 1):
                self.toposorted[i] = read_int32_t(graphinput)
                self.C_1[i] = read_int32_t(graphinput)
                self.C_2[i] = read_int32_t(graphinput)

            self.O = read_int32_t(graphinput)
            self.result_nodes = [0] * self.O
            for i in range(self.O):
                self.result_nodes[i] = read_int32_t(graphinput)

    def input_into(self, inputs):
        """Set input node values from a 0-based iterable `inputs` of length INPUT_NODES."""
        # C++ used 1..INPUT_NODES mapping: value[1] = in[0], ...
        for i in range(1, INPUT_NODES + 1):
            self.value[i] = bool(inputs[i - 1])

    def getvalue(self, id_):
        """Return boolean value for a (possibly negated) id, matching the C++ logic exactly."""
        c = False
        if id_ < 0:
            # negated reference
            assert -id_ <= self.N
            c = not self.value[-id_]
        elif abs(id_) == ALWAYS_TRUE:
            c = True
        elif abs(id_) == ALWAYS_FALSE:
            c = False
        else:
            assert id_ <= self.N
            c = self.value[id_]
        return c

    def calculatenetwork(self):
        """Compute all internal node values in toposorted order (matching C++)."""
        # loop i = INPUT_NODES+1 .. N
        for i in range(INPUT_NODES + 1, self.N + 1):
            target_id = self.toposorted[i]
            # compute AND of the two connections
            self.value[target_id] = self.getvalue(self.C_1[i]) & self.getvalue(self.C_2[i])

    def guess(self, correct):
        """
        Determine network's 'guess' score for a single sample.
        Implements the same logic as C++:
          - groups outputs into 10 categories (category = O / 10)
          - counts active nodes per category
          - sorts categories by active count descending
          - computes g/cnt where g is count of ties that include the correct category
        NOTE: kept the same order of comparisons inside the final while loop to mirror C++ behavior.
        """
        g = 0.0
        cnt = 0.0
        category = self.O // 10  # integer division, same as C++ `int category = O/10`

        # category_ids is list of (on_count, category_index)
        category_ids = [(0, 0) for _ in range(10)]
        for counter in range(10):
            on = 0
            for k in range(category):
                idx = self.result_nodes[counter * category + k]
                if self.value[idx]:
                    on += 1
            category_ids[counter] = (on, counter)

        # sort descending by (on_count, category_index) to match sort(..., greater<...>) in C++
        category_ids.sort(reverse=True)

        maximum = category_ids[0][0]
        i = 0
        # THIS replicates the C++ `while (category_ids[i].first == maximum && i < 10)`
        # which checks the .first before checking bounds — kept here to be faithful.
        while category_ids[i][0] == maximum and i < 10:
            cnt += 1.0
            if category_ids[i][1] == correct:
                g += 1.0
            i += 1

        # Return a float (g/cnt). If cnt==0 this will raise ZeroDivisionError (reflects unexpected inputs).
        return g / cnt


def make_test(net: AndNot_network, infile):
    """
    Read one sample from text `infile`, feed into `net`, compute network, and return net.guess(correct).
    Mirrors the C++ make_test: read one line of bits, then one line with correct label.
    """
    # read input image line (C++ getline removes newline)
    input_image = infile.readline()
    input_image = input_image.rstrip('\n')

    # build input boolean list of length INPUT_NODES
    input_values = [False] * INPUT_NODES
    for i in range(INPUT_NODES):
        # assume line length >= INPUT_NODES; identical to C++ indexing
        input_values[i] = (input_image[i] == '1')

    net.input_into(input_values)
    net.calculatenetwork()

    correct_line = infile.readline()
    correct_line = correct_line.rstrip('\n')
    # convert first char to digit
    correct_digit = ord(correct_line[0]) - ord('0')
    return net.guess(correct_digit)


def evaluate_network():
    # create and initialize network structure
    net = AndNot_network()
    net.init()

    # open test text input
    test_input = open(TESTDATA_DATA_FILE, 'r', encoding='utf-8')

    num = 0.0
    for i in range(TEST_SIZE):
        if i % 500 == 0:
            print(f"Tested {i}/{TEST_SIZE}")
        num += make_test(net, test_input)

    print(f"Tested {TEST_SIZE}/{TEST_SIZE}")
    accuracy = 100.0 * num / float(TEST_SIZE)
    print(f"{accuracy}% accuracy")

    test_input.close()

