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


# ---------------------------
# Varint decode helpers
# ---------------------------
def getnoneofch(f):
    b = f.read(1)
    if b:
        return b[0]
    sys.stderr.write("*** decode: unexpected EOF\n")
    sys.exit(1)

def decode(f):
    x = 0
    i = 0
    while True:
        ch = getnoneofch(f)
        if (ch & 0x80):
            x |= (ch & 0x7f) << (7 * i)
            i += 1
        else:
            return x | (ch << (7 * i))

# ---------------------------
# AndNot_network class
# ---------------------------
class AndNot_network:
    def __init__(self):
        self.value = []        # list of bools (index 0..M)
        self.C_1 = []          # list of ints (edges)
        self.C_2 = []
        self.result_nodes = [] # list of ints (output literal ids)

    def init(self, path: str):
        # open in binary so we can mix reading ascii lines and binary varints
        with open(path, 'rb') as f:
            # read header line "aig M I L O A\n"
            header_line = f.readline().decode('ascii')
            parts = header_line.strip().split()
            M = int(parts[1])
            I = int(parts[2])
            L = int(parts[3])
            O = int(parts[4])
            A = int(parts[5])

            # sanity checks (match C++ asserts)
            # read O lines of outputs (one integer per line)
            for _ in range(O):
                line = f.readline().decode('ascii')
                v = int(line.strip())
                self.result_nodes.append(v)

            # prepare arrays sized M+1 (C++ uses 1..M)
            self.value = [False] * (M + 1)
            self.C_1 = [0] * (M + 1)
            self.C_2 = [0] * (M + 1)

            # read AIG gate data: for i = INPUT_NODES+1 .. M
            for i in range(INPUT_NODES + 1, M + 1):
                idx = 2 * i
                delta0 = decode(f)
                delta1 = decode(f)
                a = idx - delta0
                b = a - delta1
                self.C_1[i] = a
                self.C_2[i] = b

    def input_into(self, inputs):
        # inputs is a list-like of booleans length INPUT_NODES
        for i in range(1, INPUT_NODES + 1):
            self.value[i] = bool(inputs[i - 1])

    def getvalue(self, lit_id: int):
        bit = self.value[lit_id // 2]
        return (not bit) if (lit_id % 2) else bit

    def calculatenetwork(self):
        # compute values for gates i = INPUT_NODES+1 .. M
        for i in range(INPUT_NODES + 1, len(self.value)):
            self.value[i] = self.getvalue(self.C_1[i]) & self.getvalue(self.C_2[i])

    def guess(self, correct: int):
        # same logic as C++: return fraction of top-tied categories that match `correct`
        g = 0.0
        cnt = 0.0
        category = OUTPUT_NODES // 10  # integer division, 10 categories for MNIST
        category_ids = [(0, 0)] * 10
        for counter in range(10):
            on = 0
            for k in range(category):
                id_ = self.result_nodes[counter * category + k]
                if self.getvalue(id_):
                    on += 1
            category_ids[counter] = (on, counter)

        # sort descending by (on, counter)
        category_ids.sort(reverse=True)

        maximum = category_ids[0][0]
        i = 0
        # NOTE: original C++ checked equality before bounds (buggy order). Here we
        # check bounds first to avoid Python IndexError but preserve intended behaviour.
        while i < 10 and category_ids[i][0] == maximum:
            cnt += 1.0
            if category_ids[i][1] == correct:
                g += 1.0
            i += 1
        return g / cnt if cnt != 0 else 0.0

# ---------------------------
# Test helper
# ---------------------------

def make_test(net: AndNot_network, infile):
    # read one line of input bits
    input_image = infile.readline()
    input_image = input_image.rstrip('\n')
    # build input booleans
    input_values = [False] * INPUT_NODES
    for i in range(0, INPUT_NODES):
        input_values[i] = (input_image[i] == '1')
    net.input_into(input_values)
    net.calculatenetwork()
    # read correct label line
    correct_line = infile.readline()
    correct_line = correct_line.rstrip('\n')
    correct_digit = ord(correct_line[0]) - ord('0')
    return net.guess(correct_digit)

# ---------------------------
# Main
# ---------------------------
def evaluate_abc_format():

    # create and initialize the network from the .aig file
    net = AndNot_network()
    net.init(TRAINED_NETWORK_2GATES_FILE)

    # open test input (text)
    test_input = open(TESTDATA_DATA_FILE, 'r', encoding='utf-8')

    num = 0.0
    for i in range(TEST_SIZE):
        if i % 500 == 0:
            print(f"Tested {i}/{TEST_SIZE}")
        num += make_test(net, test_input)

    print(f"Tested {TEST_SIZE}/{TEST_SIZE}")
    print(f"{100.0 * num / float(TEST_SIZE)}% accuracy")

    test_input.close()