#!/usr/bin/env python3
"""
⚠️ WARNING: This code was automatically translated from C++ to Python by an AI tool.
It may contain inaccuracies, inefficient constructs, or logic errors introduced 
during translation. Please review carefully and test thoroughly before using 
in production.
"""

from dataclasses import dataclass
import struct
from collections import deque
from typing import Dict, List
from config import *


def read_int32_t(f) -> int:
    """Read a 32-bit signed int from file (little-endian), same as C++ ifstream read of int32_t."""
    raw = f.read(4)
    return struct.unpack('<i', raw)[0]

@dataclass
class Node:
    type: int
    id: int
    link_a: int
    link_b: int

# Globals matching C++ file-scope variables
input_nodes: List[Node] = []
toposorted_nodes: List[Node] = []
final_nodes: List[Node] = []
node_remap: List[int] = []

node_count: int = 0
first_output_id: int = 0

# ---------------------------
# Algorithm translation
# ---------------------------

def toposort_nodes():
    global input_nodes, toposorted_nodes, node_count

    # reverse_link: node_id -> list of node ids that depend on it
    reverse_link: Dict[int, List[int]] = {}
    link_count: Dict[int, int] = {}

    process_queue = deque()

    # initial queue contains input indices 1..INPUT_NODES (inclusive)
    for i in range(1, INPUT_NODES + 1):
        process_queue.append(i)

    # Build reverse links and incoming link counts for all nodes (non-inputs)
    for node in input_nodes:
        # link_a
        if abs(node.link_a) != ALWAYS_TRUE and abs(node.link_a) != ALWAYS_FALSE:
            reverse_link.setdefault(abs(node.link_a), []).append(node.id)
            link_count[node.id] = link_count.get(node.id, 0) + 1

        # link_b
        if abs(node.link_b) != ALWAYS_TRUE and abs(node.link_b) != ALWAYS_FALSE:
            reverse_link.setdefault(abs(node.link_b), []).append(node.id)
            link_count[node.id] = link_count.get(node.id, 0) + 1

        if link_count.get(node.id, 0) == 0:
            process_queue.append(node.id)

    processed_count = 0

    while process_queue:
        node_id = process_queue.popleft()
        processed_count += 1

        if node_id > INPUT_NODES:
            # map to input_nodes index: (node_id - INPUT_NODES - 1)
            toposorted_nodes.append(input_nodes[node_id - INPUT_NODES - 1])

        for edge in reverse_link.get(node_id, []):
            link_count[edge] = link_count.get(edge, 0) - 1
            if link_count[edge] == 0:
                process_queue.append(edge)



def replace_gates():
    global final_nodes, node_remap, toposorted_nodes

    # final_nodes[0..INPUT_NODES] reserved: sized INPUT_NODES + 1 with dummy nodes
    final_nodes = [Node(0, 0, 0, 0) for _ in range(INPUT_NODES + 1)]

    def add_node(a: int, b: int) -> int:
        id_ = len(final_nodes)
        final_nodes.append(Node(0, id_, a, b))
        return 2 * id_

    # allocate node_remap to size (toposorted_nodes.size() + INPUT_NODES + 1)
    node_remap = [0] * (len(toposorted_nodes) + INPUT_NODES + 1)

    # set remap for constants and input wires
    for i in range(0, INPUT_NODES + 1):
        node_remap[i] = 2 * i

    # iterate through toposorted nodes and convert gate types
    for node in toposorted_nodes:
        input_a = node_remap[node.link_a]
        input_b = node_remap[node.link_b]
        id_ = node.id

        t = node.type

        if t == 0:  # ALWAYS 0
            node_remap[id_] = 0

        elif t == 1:  # AND
            node_remap[id_] = add_node(input_a, input_b)

        elif t == 2:  # A AND !B
            node_remap[id_] = add_node(input_a, input_b ^ 1)

        elif t == 3:  # A
            node_remap[id_] = input_a

        elif t == 4:  # B AND !A
            node_remap[id_] = add_node(input_a ^ 1, input_b)

        elif t == 5:  # B
            node_remap[id_] = input_b

        elif t == 6:  # XOR
            clean_gate = add_node(input_a, input_b)
            neg_gate = add_node(input_a ^ 1, input_b ^ 1)
            node_remap[id_] = add_node(clean_gate ^ 1, neg_gate ^ 1)

        elif t == 7:  # OR
            node_remap[id_] = 1 ^ add_node(input_a ^ 1, input_b ^ 1)

        elif t == 8:  # NOR
            node_remap[id_] = add_node(input_a ^ 1, input_b ^ 1)

        elif t == 9:  # XNOR
            clean_gate = add_node(input_a, input_b)
            neg_gate = add_node(input_a ^ 1, input_b ^ 1)
            node_remap[id_] = 1 ^ add_node(clean_gate ^ 1, neg_gate ^ 1)

        elif t == 10:  # !B
            node_remap[id_] = input_b ^ 1

        elif t == 11:  # A OR !B
            node_remap[id_] = 1 ^ add_node(input_a ^ 1, input_b)

        elif t == 12:  # !A
            node_remap[id_] = input_a ^ 1

        elif t == 13:  # B OR !A
            node_remap[id_] = 1 ^ add_node(input_a, input_b ^ 1)

        elif t == 14:  # NAND
            node_remap[id_] = 1 ^ add_node(input_a, input_b)

        elif t == 15:  # ALWAYS 1
            node_remap[id_] = 1


    # assign back to globals
    globals()['final_nodes'] = final_nodes
    globals()['node_remap'] = node_remap


def encode(binfile, x: int):
    """Write unsigned integer x as varint (7-bit groups with MSB continuation)"""
    while x & ~0x7f:
        ch = (x & 0x7f) | 0x80
        binfile.write(bytes([ch]))
        x >>= 7
    ch = x & 0x7f
    binfile.write(bytes([ch]))


# ---------------------------
# Main function
# ---------------------------

def convert_abc_format():
    global node_count, first_output_id, input_nodes, toposorted_nodes, final_nodes, node_remap

    fin = open(TRAINED_NETWORK_16GATES_FILE, 'rb')

    fout = open(TRAINED_NETWORK_2GATES_FILE, 'wb')  # binary, because encode writes binary bytes

    # read node_count
    node_count = read_int32_t(fin)

    first_output_id = node_count - OUTPUT_NODES + 1

    # read node_count - INPUT_NODES nodes, each node has 4 int32 fields
    input_nodes = []
    for _ in range(node_count - INPUT_NODES):
        v0 = read_int32_t(fin)
        v1 = read_int32_t(fin)
        v2 = read_int32_t(fin)
        v3 = read_int32_t(fin)

        if v0 > 16 or v0 < 0 or v1 <= INPUT_NODES:
            fin.close()
            fout.close()

        input_nodes.append(Node(v0, v1, v2, v3))

    # sort input_nodes by id ascending
    input_nodes.sort(key=lambda n: n.id)

    # run toposort and replace
    toposorted_nodes = []
    replace_nodes_before = None  # placeholder (not used), keep code shape similar
    toposort_nodes()
    replace_gates()

    # write header line: "aig M I L O A\n"
    M = len(final_nodes) - 1
    I = INPUT_NODES
    L = 0
    O = OUTPUT_NODES
    A = len(final_nodes) - INPUT_NODES - 1
    header = f"aig {M} {I} {L} {O} {A}\n"
    fout.write(header.encode('ascii'))

    # write outputs (one per line), using node_remap indices
    for i in range(first_output_id, first_output_id + OUTPUT_NODES):
        fout.write(f"{node_remap[i]}\n".encode('ascii'))

    # write AND-gate list as varint deltas (same order as final_nodes 1+INPUT_NODES ..)
    for i in range(1 + INPUT_NODES, len(final_nodes)):
        idx = 2 * i
        a = final_nodes[i].link_a
        b = final_nodes[i].link_b
        if a < b:
            a, b = b, a
        delta0 = idx - a
        delta1 = a - b
        encode(fout, delta0)
        encode(fout, delta1)

    fout.close()
    fin.close()