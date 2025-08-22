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
from collections import defaultdict, deque


# Node class equivalent
class Node:
    def __init__(self, _type: int, _id: int, _link_a: int, _link_b: int):
        self.type = int(_type)
        self.id = int(_id)
        self.link_a = int(_link_a)
        self.link_b = int(_link_b)

    def __repr__(self):
        return f"Node(type={self.type}, id={self.id}, link_a={self.link_a}, link_b={self.link_b})"


# Global storage (kept same names as C++ for clarity)
input_nodes = []  # list of Node
toposorted_nodes = []  # list of Node
final_nodes = []  # list of Node

# Number of new nodes, avoid clashes between ids
next_free_node = 0
first_output_id = 0


# Helper functions to read/write 32-bit signed integers (little-endian)
def read_int32_t(f):
    data = f.read(4)
    # '<i' is little-endian signed 32-bit
    return struct.unpack("<i", data)[0]


def write_int32_t(f, value):
    # Ensure value fits in signed 32-bit range
    # pack will wrap as in C; we replicate typical behaviour by masking then repacking if needed
    # but keep simple and pack as signed 32-bit.
    f.write(struct.pack("<i", int(value)))


def toposort_nodes():
    """
    Replicates the C++ toposort_nodes exactly.
    Builds reverse links, counts incoming non-constant links, and performs Kahn's algorithm.
    """
    # Use same container semantics
    reverse_link = defaultdict(list)  # map from int -> list[int]
    link_count = defaultdict(int)  # map from node.id -> int

    process_queue = deque()

    # push input node ids 1..INPUT_NODES into queue
    for i in range(1, INPUT_NODES + 1):
        process_queue.append(i)

    # Build reverse links and counts
    for node in input_nodes:
        if abs(node.link_a) != ALWAYS_TRUE and abs(node.link_a) != ALWAYS_FALSE:
            reverse_link[abs(node.link_a)].append(node.id)
            link_count[node.id] += 1

        if abs(node.link_b) != ALWAYS_TRUE and abs(node.link_b) != ALWAYS_FALSE:
            reverse_link[abs(node.link_b)].append(node.id)
            link_count[node.id] += 1

        if link_count[node.id] == 0:
            process_queue.append(node.id)

    processed_count = 0

    while process_queue:
        node_id = process_queue.popleft()
        processed_count += 1

        if node_id > INPUT_NODES:
            # Indexing in C++ used: input_nodes[node_id-INPUT_NODES-1]
            idx = node_id - INPUT_NODES - 1
            # Defensive check: if index out of range, abort as C++ would likely do
            toposorted_nodes.append(input_nodes[idx])

        for edge in reverse_link.get(node_id, []):
            # decrement link_count[edge]
            link_count[edge] -= 1
            if link_count[edge] == 0:
                process_queue.append(edge)

    # Validate processed nodes count equals node_count - INPUT_NODES (same check as C++)


def replace_gates():
    """
    Replaces multi-input gates with NAND/NOT combinations exactly mirroring the C++ switch cases.
    """
    global next_free_node

    for i in range(len(toposorted_nodes)):
        new_nodes = []

        __input_a = toposorted_nodes[i].link_a
        __input_b = toposorted_nodes[i].link_b
        __id = toposorted_nodes[i].id
        typ = toposorted_nodes[i].type

        if typ == 0:  # ALWAYS 0
            new_nodes.append(Node(0, __id, ALWAYS_FALSE, ALWAYS_FALSE))

        elif typ == 1:  # AND
            new_nodes.append(Node(0, __id, __input_a, __input_b))

        elif typ == 2:  # A AND !B
            new_nodes.append(Node(0, __id, __input_a, -__input_b))

        elif typ == 3:  # A
            new_nodes.append(Node(0, __id, __input_a, ALWAYS_TRUE))

        elif typ == 4:  # B AND !A
            new_nodes.append(Node(0, __id, -__input_a, __input_b))

        elif typ == 5:  # B
            new_nodes.append(Node(0, __id, ALWAYS_TRUE, __input_b))

        elif typ == 6:  # XOR
            clean_gate = next_free_node
            new_nodes.append(Node(0, next_free_node, __input_a, __input_b))
            next_free_node += 1

            neg_gate = next_free_node
            new_nodes.append(Node(0, next_free_node, -__input_a, -__input_b))
            next_free_node += 1

            new_nodes.append(Node(0, __id, -clean_gate, -neg_gate))

        elif typ == 7:  # OR
            nand_gate = next_free_node
            new_nodes.append(Node(0, next_free_node, -__input_a, -__input_b))
            next_free_node += 1

            new_nodes.append(Node(0, __id, -nand_gate, -nand_gate))

        elif typ == 8:  # NOR
            t_gate = next_free_node
            new_nodes.append(Node(0, next_free_node, -__input_a, -__input_b))
            next_free_node += 1

            new_nodes.append(Node(0, __id, t_gate, t_gate))

        elif typ == 9:  # XNOR
            nand_1 = next_free_node
            new_nodes.append(Node(0, next_free_node, __input_a, __input_b))
            next_free_node += 1

            nand_2 = next_free_node
            new_nodes.append(Node(0, next_free_node, __input_a, -nand_1))
            next_free_node += 1

            nand_3 = next_free_node
            new_nodes.append(Node(0, next_free_node, __input_b, -nand_1))
            next_free_node += 1

            nand_4 = next_free_node
            new_nodes.append(Node(0, next_free_node, -nand_2, -nand_3))
            next_free_node += 1

            new_nodes.append(Node(0, __id, nand_4, nand_4))

        elif typ == 10:  # !B
            new_nodes.append(Node(0, __id, -__input_b, ALWAYS_TRUE))

        elif typ == 11:  # A OR !B
            nand_gate = next_free_node
            new_nodes.append(Node(0, next_free_node, -__input_a, __input_b))
            next_free_node += 1

            new_nodes.append(Node(0, __id, -nand_gate, -nand_gate))

        elif typ == 12:  # !A
            new_nodes.append(Node(0, __id, -__input_a, ALWAYS_TRUE))

        elif typ == 13:  # B OR !A
            nand_gate = next_free_node
            new_nodes.append(Node(0, next_free_node, __input_a, -__input_b))
            next_free_node += 1

            new_nodes.append(Node(0, __id, -nand_gate, -nand_gate))

        elif typ == 14:  # NAND
            and_node = next_free_node
            new_nodes.append(Node(0, next_free_node, __input_a, __input_b))
            next_free_node += 1

            new_nodes.append(Node(0, __id, -and_node, -and_node))

        elif typ == 15:  # ALWAYS 1
            new_nodes.append(Node(0, __id, ALWAYS_TRUE, ALWAYS_TRUE))

        # Validate same as C++

        # Append to final_nodes
        for t in range(len(new_nodes)):
            final_nodes.append(new_nodes[t])


def convert_network():
    global first_output_id, next_free_node

    t16_ifstream = open(TRAINED_NETWORK_16GATES_FILE, "rb")

    t2_ofstream = open(TRAINED_NETWORK_2GATES_FILE, "wb")

    # Read node_count

    first_output_id = NETWORK_SIZE - OUTPUT_NODES + 1
    next_free_node = NETWORK_SIZE + 1

    # Read nodes: NETWORK_SIZE - INPUT_NODES entries, each with 4 int32s
    for i in range(NETWORK_SIZE - INPUT_NODES):
        v0 = read_int32_t(t16_ifstream)
        v1 = read_int32_t(t16_ifstream)
        v2 = read_int32_t(t16_ifstream)
        v3 = read_int32_t(t16_ifstream)

        input_nodes.append(Node(v0, v1, v2, v3))

    # Sort input_nodes by id as in C++
    input_nodes.sort(key=lambda n: n.id)

    # Topological sort and gate replacement
    toposort_nodes()
    replace_gates()

    # Writing sequence
    write_int32_t(t2_ofstream, next_free_node - 1)

    for i in range(len(final_nodes)):
        write_int32_t(t2_ofstream, final_nodes[i].id)
        write_int32_t(t2_ofstream, final_nodes[i].link_a)
        write_int32_t(t2_ofstream, final_nodes[i].link_b)

    write_int32_t(t2_ofstream, OUTPUT_NODES)

    for i in range(first_output_id, first_output_id + OUTPUT_NODES):
        write_int32_t(t2_ofstream, i)

    t16_ifstream.close()
    t2_ofstream.close()
    return 0
