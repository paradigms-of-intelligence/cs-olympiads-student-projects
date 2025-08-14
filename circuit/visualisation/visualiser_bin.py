import matplotlib.pyplot as plot
import networkx as nx
from numpy import append

if not __name__ == "__main__": exit(0)
gate16 = input("Use 16-type graph? [1] ") == '1'

def bytetoint(bytes) -> int:
    i = int.from_bytes(bytes, "little")
    if i >= 2**31: i -= 2**32
    return i
notedge = []
yesedge = []
def addedge(linka, node) -> None:
    if linka < 0: notedge.append((-linka, node))
    else: yesedge.append((linka, node))
G = nx.DiGraph()


with open("../exec/trained_network.bin" if gate16 else "../exec/2gate_trained_network.bin", "rb") as nw:
    size = bytetoint(nw.read(4))
    print(size)
    for i in range(size):
        typ = 0
        if gate16: typ = bytetoint(nw.read(4))
        nodeid = bytetoint(nw.read(4))
        linka = bytetoint(nw.read(4))
        linkb = bytetoint(nw.read(4))
        addedge(linka, nodeid)
        addedge(linkb, nodeid)

G.add_edges_from(yesedge)
G.add_edges_from(notedge)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color="lightblue")
nx.draw_networkx_edges(G, pos, edgelist=yesedge, edge_color="black")
nx.draw_networkx_edges(G, pos, edgelist=notedge, edge_color="red")

plot.show()
