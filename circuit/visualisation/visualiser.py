import matplotlib.pyplot as plot
import networkx as nx
import sys

if not __name__ == "__main__": exit(0)
gate16 = False

def bytetoint(bytes) -> int:
    return int.from_bytes(bytes, "little")

G = nx.DiGraph()

size = int(sys.stdin.readline())

nodeid = 785

for line in sys.stdin:
    nums = line.split(' ')
    if(len(nums) != 2): break
    a = int(nums[0])
    b = int(nums[1])
    if a == -1 and b == -1: continue
    G.add_edges_from([(a, nodeid), (b, nodeid)])
    nodeid += 1
print(size)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, arrows=True, node_color="lightblue", node_size=2000)
plot.title("DAG")
plot.show()
